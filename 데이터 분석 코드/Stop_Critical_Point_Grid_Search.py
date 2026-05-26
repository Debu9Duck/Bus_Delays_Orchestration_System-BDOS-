import json
import numpy as np

# ═══════════════════════════════════════════════════════════
#  설정값  ── 여기만 수정하세요
# ═══════════════════════════════════════════════════════════
DATA_PATH    = "4st_input.jsonl"

STOP_RANGE   = (0, 15)    # 정지 임계속도 탐색 범위 (km/h)
DEPART_RANGE = (0, 20)    # 출발 임계속도 탐색 범위 (km/h)
STEP         = 0.5        # 그리드 간격 (km/h)

MIN_GAP      = 5.0        # 최소 안전지대 (km/h)
MIN_DWELL    = 2          # 양질의 정차로 인정할 최소 연속 정차 스텝 수

# ── stop_thr 구간 보상 설정 ──────────────────────────────
# 저속 크리프(실제 정차지만 속도 > 0)를 잡기 위해
# stop_thr이 이 범위 안에 있으면 보상, 밖이면 패널티
STOP_SWEET_ZONE = (1.0, 5.0)   # 보상 구간 (km/h) ← 조정 핵심
SWEET_BONUS     = 1.05         # 구간 내 보상 배율 (1.0 초과 = 보상)
SWEET_PENALTY   = 0.15         # 구간 밖 거리당 패널티 강도

BETA            = 0.10         # depart_thr 크기 패널티 강도
# ═══════════════════════════════════════════════════════════


# ── 1. 데이터 로드 ──────────────────────────────────────────
def load_data(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── 2. stop_thr 구간 보상 계산 ──────────────────────────────
def stop_zone_weight(stop_thr):
    lo, hi = STOP_SWEET_ZONE
    if lo <= stop_thr <= hi:
        return SWEET_BONUS
    elif stop_thr < lo:
        dist = lo - stop_thr
    else:
        dist = stop_thr - hi
    return SWEET_BONUS * max(0.0, 1.0 - SWEET_PENALTY * dist)


# ── 3. 구간별 정차/출발 이벤트 추출 ────────────────────────
def extract_events(speeds, stop_thr, depart_thr):
    n_stops    = 0
    n_departs  = 0
    good_stops = 0
    in_stop    = False
    dwell      = 0

    for i in range(1, len(speeds)):
        prev = speeds[i - 1]
        curr = speeds[i]

        if not in_stop:
            if prev > stop_thr and curr <= stop_thr:
                n_stops += 1
                in_stop  = True
                dwell    = 1
        else:
            if curr <= depart_thr:
                dwell += 1
            else:
                n_departs += 1
                if dwell >= MIN_DWELL:
                    good_stops += 1
                in_stop = False
                dwell   = 0

    # 구간 종료 시 아직 정차 중
    if in_stop and dwell >= MIN_DWELL:
        good_stops += 1

    return n_stops, n_departs, good_stops


# ── 4. 전체 데이터 집계 ─────────────────────────────────────
def evaluate(records, stop_thr, depart_thr):
    total_stops   = 0
    total_departs = 0
    total_good    = 0

    for rec in records:
        s, d, g = extract_events(rec["10secSpeed"], stop_thr, depart_thr)
        total_stops   += s
        total_departs += d
        total_good    += g

    good_stop_rate = total_good    / total_stops  if total_stops > 0 else 0.0
    dep_rate       = total_departs / total_stops  if total_stops > 0 else 0.0

    if good_stop_rate + dep_rate > 0:
        f1 = 2 * good_stop_rate * dep_rate / (good_stop_rate + dep_rate)
    else:
        f1 = 0.0

    return good_stop_rate, dep_rate, f1, total_stops, total_departs, total_good


# ── 5. 그리드 서치 ──────────────────────────────────────────
def grid_search(records):
    stop_cands   = np.round(np.arange(STOP_RANGE[0],   STOP_RANGE[1]   + STEP, STEP), 2)
    depart_cands = np.round(np.arange(DEPART_RANGE[0], DEPART_RANGE[1] + STEP, STEP), 2)
    depart_max   = DEPART_RANGE[1] if DEPART_RANGE[1] > 0 else 1

    results = []

    for stop_thr in stop_cands:
        sw = stop_zone_weight(float(stop_thr))   # stop 구간 보상/패널티

        for depart_thr in depart_cands:
            gap = round(float(depart_thr) - float(stop_thr), 4)
            if gap < MIN_GAP:
                continue

            gsr, dr, f1, ns, nd, ng = evaluate(
                records, float(stop_thr), float(depart_thr)
            )

            # depart 크기 패널티 (여전히 높은 출발 임계값 억제)
            depart_norm  = float(depart_thr) / depart_max
            depart_pen   = 1.0 - BETA * depart_norm

            # 최종 점수 = F1 × stop 구간 보상 × depart 패널티
            score = f1 * sw * depart_pen

            results.append({
                "stop_thr":       float(stop_thr),
                "depart_thr":     float(depart_thr),
                "gap":            gap,
                "good_stop_rate": gsr,
                "dep_rate":       dr,
                "f1":             f1,
                "stop_weight":    sw,
                "score":          score,
                "n_stops":        ns,
                "n_departs":      nd,
                "n_good":         ng,
            })

    return sorted(results, key=lambda x: x["score"], reverse=True)


# ── 6. 출력 헬퍼 ────────────────────────────────────────────
HDR = (f"{'gap':>5}  {'stop':>6}  {'depart':>7}  "
       f"{'good_stop_rate':>14}  {'dep_rate':>9}  "
       f"{'F1':>7}  {'sw':>6}  {'score':>9}  {'정차(양질)/출발':>16}")
SEP = "-" * 90

def print_table(rows, n=15):
    print(HDR)
    print(SEP)
    for row in rows[:n]:
        ratio = f"{row['n_good']:,}({row['n_stops']:,})/{row['n_departs']:,}"
        print(
            f"{row['gap']:>5.1f}  "
            f"{row['stop_thr']:>6.1f}  "
            f"{row['depart_thr']:>7.1f}  "
            f"{row['good_stop_rate']:>14.4f}  "
            f"{row['dep_rate']:>9.4f}  "
            f"{row['f1']:>7.4f}  "
            f"{row['stop_weight']:>6.4f}  "
            f"{row['score']:>9.4f}  "
            f"{ratio:>16}"
        )


# ── 7. 메인 ─────────────────────────────────────────────────
def main():
    print(f"[데이터 로드] {DATA_PATH}")
    records = load_data(DATA_PATH)
    n_total = len(records)
    print(f"  → 총 {n_total:,}개 구간 로드 완료\n")

    lo, hi = STOP_SWEET_ZONE
    print("=" * 90)
    print("그리드 서치 v5  (Hysteresis + 양질 정차 + stop 구간 보상)")
    print(f"  정지 임계속도 탐색 범위  : {STOP_RANGE[0]} ~ {STOP_RANGE[1]} km/h  (step={STEP})")
    print(f"  출발 임계속도 탐색 범위  : {DEPART_RANGE[0]} ~ {DEPART_RANGE[1]} km/h (step={STEP})")
    print(f"  최소 안전지대 (MIN_GAP)  : {MIN_GAP} km/h")
    print(f"  양질 정차 기준 (MIN_DWELL): {MIN_DWELL} 스텝 이상 연속 정차")
    print(f"  stop 보상 구간           : {lo} ~ {hi} km/h  "
          f"(보상 ×{SWEET_BONUS}, 구간 밖 거리당 -{SWEET_PENALTY})")
    print(f"  depart 크기 패널티 β     : {BETA}")
    print("=" * 90)

    # 구간 보상 프리뷰
    print("\n[stop_thr 구간 보상 미리보기]")
    preview_stops = np.round(np.arange(0, STOP_RANGE[1] + 1, 1.0), 1)
    for s in preview_stops:
        w = stop_zone_weight(float(s))
        bar = "★" if lo <= s <= hi else ("↑보상" if s < lo else "↓패널티")
        tag = f"  ← 보상구간 [{lo}~{hi}]" if lo <= s <= hi else ""
        print(f"    stop={s:4.1f}  weight={w:.4f}  {bar}{tag}")

    all_results = grid_search(records)

    if not all_results:
        print("\n유효한 조합이 없습니다. MIN_GAP을 줄여보세요.")
        return

    best = all_results[0]
    b = best

    print("\n★ 최적 임계값 탐색 결과 ★")
    print("-" * 90)
    print(f"  정지 임계속도  (stop_threshold)      : {b['stop_thr']:.1f} km/h")
    print(f"  출발 임계속도  (depart_threshold)     : {b['depart_thr']:.1f} km/h")
    print(f"  안전지대  (gap = depart - stop)       : {b['gap']:.1f} km/h")
    print(f"  stop 구간 보상 가중치                 : {b['stop_weight']:.4f}")
    print(f"  정지 탐색률  (양질 정차 / 전체 정차)  : {b['good_stop_rate']:.4f}"
          f"  ({b['n_good']:,} / {b['n_stops']:,}건)")
    print(f"    ※ 양질 정차 = 정지→정지(×{MIN_DWELL-1}↑)→출발  패턴")
    print(f"    ※ 불량 정차 = 정지→출발  (즉시 출발, 노이즈 의심)")
    print(f"  출발 탐색률  (출발 감지 / 전체 정차)  : {b['dep_rate']:.4f}"
          f"  ({b['n_departs']:,} / {b['n_stops']:,}건)")
    print(f"  F1 균형 점수                          : {b['f1']:.4f}")
    print(f"  최종 점수 (F1 × sw × depart_pen)      : {b['score']:.4f}")
    print("-" * 90)

    # gap별 최적
    best_per_gap = {}
    for row in all_results:
        g = round(row["gap"], 1)
        if g not in best_per_gap:
            best_per_gap[g] = row
    gap_table = sorted(best_per_gap.values(), key=lambda x: x["score"], reverse=True)

    print(f"\n▶ 안전지대(gap)별 최적 조합  (상위 {min(15, len(gap_table))}개)")
    print_table(gap_table)

    # stop_thr별 최적
    best_per_stop = {}
    for row in all_results:
        s = row["stop_thr"]
        if s not in best_per_stop:
            best_per_stop[s] = row
    stop_table = sorted(best_per_stop.values(), key=lambda x: x["score"], reverse=True)

    print(f"\n▶ 정지 임계값별 최적 조합  (상위 {min(15, len(stop_table))}개)")
    print_table(stop_table)

    print("\n[참고]")
    print(f"  · STOP_SWEET_ZONE ({lo}~{hi} km/h): 저속 정차(크리프)를 잡기 위한 보상 구간")
    print(f"    stop=0.0이면 완전 정지만 감지 → 크리프 정차 누락")
    print(f"    stop이 이 구간 안에 있으면 ×{SWEET_BONUS} 보상, 밖이면 거리 비례 패널티")
    print(f"  · STOP_SWEET_ZONE 조정 방법:")
    print(f"    실제 버스 저속 크리프 속도 범위를 확인 후 lo/hi 설정")
    print(f"    예) 크리프가 주로 0.5~3 km/h → STOP_SWEET_ZONE = (0.5, 3.0)")
    print(f"  · MIN_DWELL({MIN_DWELL}): 정지→정지×{MIN_DWELL-1}→출발 이상이어야 양질 정차")
    print(f"  · MIN_GAP({MIN_GAP} km/h) 미만 조합은 진동 방지를 위해 제외")


if __name__ == "__main__":
    main()