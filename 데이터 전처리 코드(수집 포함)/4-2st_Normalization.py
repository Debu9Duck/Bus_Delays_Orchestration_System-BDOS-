import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
INPUT_PATH       = Path("Normalization_input.jsonl")
STOP_LABEL_PATH  = Path("4st_output.json")
OUTPUT_PATH      = Path("Normalization_output.json")

BUCKET_SIZE  = 0.01   # 10m (km)
MIN_SAMPLES  = 2      # 버킷 최소 샘플 수
MIN_COMBO_N  = 2      # 콤보당 최소 로그 수 (미달 시 해당 콤보 제외)

# ── 승하차 시간 보정 상수 ──────────────────────
# 실제 데이터 돌린 후 출력되는 dwell 갭 통계를 보고 아래 값을 튜닝하세요.
#   p10이 5초 미만  → DWELL_LOW_THRESHOLD 올리기
#   p90이 60초 초과 → DWELL_HIGH_THRESHOLD 낮추기
#   평균값이 도메인 기대치(예: 15~20초)와 다르면 ADD/SUBTRACT_CONST 조정
DWELL_HIGH_THRESHOLD = 60    # 초 초과 시 → 비정상적으로 긴 갭 (신호 대기 등 포함)
DWELL_LOW_THRESHOLD  = 5     # 초 미만 시 → 비정상적으로 짧은 갭 (데이터 누락 등)
DWELL_SUBTRACT_CONST = 30    # 긴 갭에서 뺄 값 (초) — 순수 승하차 시간 추정
DWELL_ADD_CONST      = 15    # 짧은 갭에 더할 값 (초) — 누락된 대기 시간 보정
DWELL_MAX_VALID_GAP  = 300   # 초 초과 시 → 같은 날 운행이 아닌 것으로 보고 완전 제외
#   └ 실제 정류장 정차가 5분(300s)을 넘으면 다른 날/운행 회차로 간주
#     데이터 보고 조정 가능 (예: 종점 회차 대기가 길면 600으로 올리기)


# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────
def parse_time(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d%H%M%S")

def get_time_slot(dt: datetime) -> str:
    h = dt.hour
    return f"{h:02d}:00~{h+1:02d}:00"

def bucket_key(dist: float) -> float:
    return round(round(dist / BUCKET_SIZE) * BUCKET_SIZE, 6)

def dist_str(dist: float) -> str:
    return f"{dist:.6f}"


# ──────────────────────────────────────────────
# 1단계: 정차 라벨 로드
# ──────────────────────────────────────────────
def load_labels(path: Path) -> dict:
    """
    4st_output.json → {slot → {seq_str → {veh_key → [패턴인덱스, ...]}}}
    """
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)

    labels = {}
    for slot, seq_dict in raw.get("log_labels", {}).items():
        labels[slot] = {}
        for seq_str, data in seq_dict.items():
            labels[slot][seq_str] = data.get("logs", {})
    return labels


# ──────────────────────────────────────────────
# 2단계: JSONL 로드
# ──────────────────────────────────────────────
def load_logs(path: Path) -> list:
    logs, errors = [], 0
    with open(path, encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors += 1
                if errors <= 5:
                    print(f"  [경고] {lineno}번째 줄 오류: {e}", file=sys.stderr)
    print(f"  로드 완료: {len(logs)}건 / 오류: {errors}건")
    return logs


# ──────────────────────────────────────────────
# 3단계: 유효성 검사
# ──────────────────────────────────────────────
def validate_log(log: dict):
    for field in ["seq", "10secDist", "10secTm", "10secSpeed", "starttime"]:
        if field not in log:
            return False, f"필드 없음: {field}"
    d, t, s = len(log["10secDist"]), len(log["10secTm"]), len(log["10secSpeed"])
    if not (d == t == s):
        return False, "배열 길이 불일치"
    if d < 2:
        return False, "포인트 수 부족"
    dists = log["10secDist"]
    for i in range(1, len(dists)):
        if dists[i] < dists[i-1] - 0.001:
            return False, f"역방향 거리 감소: idx={i}"
    return True, ""


# ──────────────────────────────────────────────
# 3.5단계: 정류장 승하차 시간 추정 (신규)
# ──────────────────────────────────────────────
def compute_dwell_map(logs: list) -> dict:
    # vehId별로 로그 모으기
    veh_logs = defaultdict(list)
    for log in logs:
        veh_logs[log["vehId"]].append(log)

    dwell_map = defaultdict(dict)   # {vehId: {seq: corrected_sec}}
    raw_gaps  = []                  # 통계 출력용

    for veh_id, vlogs in veh_logs.items():
        vlogs.sort(key=lambda x: x["seq"])

        for i in range(len(vlogs) - 1):
            cur = vlogs[i]
            nxt = vlogs[i + 1]

            # seq 연속 여부 확인 (같은 노선·같은 버스의 연속 구간만)
            if nxt["seq"] - cur["seq"] != 1:
                continue

            # endtime이 없는 경우 방어
            if "endtime" not in cur or "starttime" not in nxt:
                continue

            # 10secTm 마지막 원소(cur 구간 끝) ~ 첫 원소(nxt 구간 시작) 차이로 갭 계산
            # endtime/starttime 필드는 오차가 크므로 사용하지 않음
            cur_tm = cur.get("10secTm", [])
            nxt_tm = nxt.get("10secTm", [])
            if not cur_tm or not nxt_tm:
                continue

            end_dt   = parse_time(cur_tm[-1])
            start_dt = parse_time(nxt_tm[0])

            # 필터 1: 같은 날짜인 경우만 허용 (날짜 다르면 다른 운행 회차)
            if end_dt.date() != start_dt.date():
                continue

            gap = (start_dt - end_dt).total_seconds()

            # 필터 2: 음수 갭 제외 (데이터 오류)
            if gap < 0:
                continue

            # 필터 3: 상한 초과 시 완전 제외 (같은 운행이 아닌 것으로 판단)
            if gap > DWELL_MAX_VALID_GAP:
                continue

            raw_gaps.append(gap)

            # 이상치 보정
            if gap > DWELL_HIGH_THRESHOLD:
                corrected = gap - DWELL_SUBTRACT_CONST
            elif gap < DWELL_LOW_THRESHOLD:
                corrected = gap + DWELL_ADD_CONST
            else:
                corrected = gap

            # cur["seq"] 정류장(구간 종점)에서의 승하차 시간으로 저장
            dwell_map[veh_id][cur["seq"]] = corrected

    # 통계 요약 출력
    if raw_gaps:
        arr = np.array(raw_gaps)
        print(f"  dwell 갭 통계 ({len(arr)}쌍 / 상한 {DWELL_MAX_VALID_GAP}s 이하만) :"
              f" 평균={np.mean(arr):.1f}s"
              f" | p10={np.percentile(arr, 10):.1f}s"
              f" | p50={np.percentile(arr, 50):.1f}s"
              f" | p90={np.percentile(arr, 90):.1f}s")
        high = sum(1 for g in raw_gaps if g > DWELL_HIGH_THRESHOLD)
        low  = sum(1 for g in raw_gaps if g < DWELL_LOW_THRESHOLD)
        print(f"  보정 대상: 긴 갭({DWELL_HIGH_THRESHOLD}s 초과) {high}건"
              f" / 짧은 갭({DWELL_LOW_THRESHOLD}s 미만) {low}건")
    else:
        print("  [주의] 연속 seq 쌍을 찾지 못했습니다 — dwell 보정 없이 진행합니다.")

    return dwell_map


# ──────────────────────────────────────────────
# 4단계: 로그 1건 → 포인트 추출 + 10m 보간
# ──────────────────────────────────────────────
def extract_and_interpolate(log: dict) -> tuple:
    dists    = log["10secDist"]
    times    = log["10secTm"]
    start_dt = parse_time(times[0])

    raw = [
        (dists[i], (parse_time(times[i]) - start_dt).total_seconds())
        for i in range(len(dists))
    ]
    total_dist    = raw[-1][0]
    total_elapsed = raw[-1][1]

    points = []
    for i in range(len(raw) - 1):
        d0, e0 = raw[i]
        d1, e1 = raw[i + 1]
        span   = d1 - d0
        points.append((d0, e0))
        if span > BUCKET_SIZE:
            n = int(span / BUCKET_SIZE)
            for step in range(1, n):
                r = step / n
                points.append((d0 + span * r, e0 + (e1 - e0) * r))
    points.append(raw[-1])

    return points, total_dist, total_elapsed


# ──────────────────────────────────────────────
# 5단계: 버킷 통계
# ──────────────────────────────────────────────
def calc_stats(arr: np.ndarray) -> dict:
    return {
        "mean": round(float(np.mean(arr)),        1),  # 6자리 → 소수 1자리
        "p10":  round(float(np.percentile(arr, 10)), 1),
        "p50":  round(float(np.percentile(arr, 50)), 1),
        "p90":  round(float(np.percentile(arr, 90)), 1),
        "n":    len(arr),
    }


# ──────────────────────────────────────────────
# 6단계: 빈 버킷 선형 보간
# ──────────────────────────────────────────────
def fill_gaps(valid: dict, total_dist: float) -> list:
    if not valid:
        return {}

    valid_keys = sorted(valid.keys())
    result     = {}
    k = 0.0
    while k <= total_dist + BUCKET_SIZE:
        k_r = round(k, 6)

        if k_r in valid:
            result[dist_str(k_r)] = valid[k_r]
        else:
            left  = [vk for vk in valid_keys if vk <= k_r]
            right = [vk for vk in valid_keys if vk >= k_r]
            if left and right:
                lk, rk = left[-1], right[0]
                if lk == rk:
                    result[dist_str(k_r)] = {**valid[lk], "n": 0}
                else:
                    ratio = (k_r - lk) / (rk - lk)
                    l, r  = valid[lk], valid[rk]
                    result[dist_str(k_r)] = {
                        "mean": l["mean"] + (r["mean"] - l["mean"]) * ratio,
                        "p10":  l["p10"]  + (r["p10"]  - l["p10"])  * ratio,
                        "p50":  l["p50"]  + (r["p50"]  - l["p50"])  * ratio,
                        "p90":  l["p90"]  + (r["p90"]  - l["p90"])  * ratio,
                        "n":    0,
                    }

        k = round(k + BUCKET_SIZE, 6)

    return result


# ──────────────────────────────────────────────
# 7단계: seq 그룹 → 콤보별 정규화
# ──────────────────────────────────────────────
def normalize_seq_group(logs: list, label_map: dict, dwell_map: dict = None) -> dict:
    combo_buckets  = defaultdict(lambda: defaultdict(list))
    combo_arrivals = defaultdict(list)
    all_dists      = []
    baseline_buckets = defaultdict(list)   # ← 신규: 콤보 무관 전체 버킷
    baseline_arrivals = []                 # ← 신규: 콤보 무관 전체 arrival

    for log in logs:
        points, total_dist, total_elapsed = extract_and_interpolate(log)
        all_dists.append(total_dist)

        if dwell_map:
            veh_id = log.get("vehId", "")
            seq    = log.get("seq")
            dwell  = dwell_map.get(veh_id, {}).get(seq)
            if dwell is not None:
                total_elapsed += dwell

        veh_key   = f"{log.get('vehId', 'unknown')}__{log.get('starttime', '')}"
        indices   = label_map.get(veh_key, [])
        combo_key = str(indices)

        combo_arrivals[combo_key].append(total_elapsed)
        baseline_arrivals.append(total_elapsed)          # ← 신규

        for dist, elapsed in points:
            bk = bucket_key(dist)
            combo_buckets[combo_key][bk].append(elapsed)
            baseline_buckets[bk].append(elapsed)         # ← 신규

    total_dist_median = float(np.median(all_dists))

    # ── 기존 콤보별 통계 ───────────────────────
    combos = {}
    for combo_key, buckets in combo_buckets.items():
        n_logs = len(combo_arrivals[combo_key])
        if n_logs < MIN_COMBO_N:
            continue

        valid = {}
        for bk, samples in sorted(buckets.items()):
            if len(samples) >= MIN_SAMPLES:
                valid[bk] = calc_stats(np.array(samples))

        dist_to_elapsed = fill_gaps(valid, total_dist_median)

        arr = np.array(combo_arrivals[combo_key])
        combos[combo_key] = {
            "sample_count":    n_logs,
            "arrival": {
                "mean": float(np.mean(arr)),
                "std":  float(np.std(arr)),
                "p10":  float(np.percentile(arr, 10)),
                "p50":  float(np.percentile(arr, 50)),
                "p90":  float(np.percentile(arr, 90)),
            },
            "dist_to_elapsed": dist_to_elapsed,
        }

    # ── 신규: baseline (콤보 무관 전체 평균) ───
    baseline_valid = {}
    for bk, samples in sorted(baseline_buckets.items()):
        if len(samples) >= MIN_SAMPLES:
            baseline_valid[bk] = calc_stats(np.array(samples))

    baseline_dist_to_elapsed = fill_gaps(baseline_valid, total_dist_median)

    baseline_arr = np.array(baseline_arrivals)
    baseline = {
        "sample_count": len(baseline_arrivals),
        "arrival": {
            "mean": float(np.mean(baseline_arr)),
            "std":  float(np.std(baseline_arr)),
            "p10":  float(np.percentile(baseline_arr, 10)),
            "p50":  float(np.percentile(baseline_arr, 50)),
            "p90":  float(np.percentile(baseline_arr, 90)),
        },
        "dist_to_elapsed": baseline_dist_to_elapsed,
    }
    # ─────────────────────────────────────────

    return {
        "sample_count":            len(logs),
        "total_dist_median":       total_dist_median,
        "bucket_size_km":          BUCKET_SIZE,
        "baseline":                baseline,   # ← 신규
        "combos":                  combos,
    }


# ──────────────────────────────────────────────
# 8단계: 전체 파이프라인
# ──────────────────────────────────────────────
def preprocess(input_path: Path, label_path: Path, output_path: Path) -> None:
    print(f"\n{'='*52}")
    print(f"  정규화 전처리 시작")
    print(f"  입력  : {input_path}")
    print(f"  라벨  : {label_path}")
    print(f"  출력  : {output_path}")
    print(f"  버킷  : {BUCKET_SIZE*1000:.0f}m 단위")
    print(f"{'='*52}\n")

    print("[1/6] 정차 라벨 로드 중...")
    labels = load_labels(label_path)
    print(f"  슬롯 {len(labels)}개 로드 완료")

    print("[2/6] 로그 파일 로드 중...")
    logs = load_logs(input_path)

    print("[3/6] 유효성 검사 중...")
    valid_logs, invalid_count = [], 0
    for log in logs:
        ok, _ = validate_log(log)
        if ok:
            valid_logs.append(log)
        else:
            invalid_count += 1
    print(f"  유효: {len(valid_logs)}건 / 제외: {invalid_count}건")

    print("[4/6] 정류장 승하차 시간 추정 중...")          # ← 신규 단계
    dwell_map = compute_dwell_map(valid_logs)

    print("[5/6] 시간대·seq별 그룹핑 및 콤보별 정규화 중...")
    groups = defaultdict(lambda: defaultdict(list))
    for log in valid_logs:
        slot = get_time_slot(parse_time(log["starttime"]))
        groups[slot][log["seq"]].append(log)

    output    = {}
    total_seq = sum(len(seqs) for seqs in groups.values())
    done      = 0

    for slot in sorted(groups.keys()):
        output[slot] = {}
        label_slot   = labels.get(slot, {})

        for seq in sorted(groups[slot].keys()):
            label_map  = label_slot.get(str(seq), {})
            normalized = normalize_seq_group(            # dwell_map 전달
                groups[slot][seq], label_map, dwell_map
            )
            output[slot][str(seq)] = {"seq": seq, **normalized}
            done += 1
            print(f"  진행: {done}/{total_seq}", end="\r")
    print()

    print("[6/6] 결과 저장 중...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    file_kb = output_path.stat().st_size / 1024
    print(f"\n{'='*52}")
    print(f"  완료!")
    print(f"  시간대 수 : {len(output)}개")
    print(f"  총 seq 수 : {total_seq}개")
    print(f"  파일 크기 : {file_kb:.1f} KB")
    print(f"{'='*52}\n")

    print("  [시간대별 요약]")
    for slot in sorted(output.keys()):
        seqs = output[slot]
        sample_total = sum(v["sample_count"] for v in seqs.values())
        combo_total  = sum(len(v["combos"])  for v in seqs.values())
        print(f"  {slot}: {len(seqs)}개 seq | "
              f"{sample_total}건 샘플 | 콤보 총 {combo_total}개")


if __name__ == "__main__":
    for p in [INPUT_PATH, STOP_LABEL_PATH]:
        if not p.exists():
            print(f"오류: 파일 없음 -> {p}", file=sys.stderr)
            sys.exit(1)
    preprocess(INPUT_PATH, STOP_LABEL_PATH, OUTPUT_PATH)