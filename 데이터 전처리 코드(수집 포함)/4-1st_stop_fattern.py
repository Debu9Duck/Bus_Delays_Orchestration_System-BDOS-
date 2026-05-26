import json
import statistics
from collections import defaultdict

# ── 설정 ──────────────────────────────────────────
STOP_ENTER_SPEED = 5.7
STOP_EXIT_SPEED  = 13.42
INTERVAL_SEC     = 5
CLUSTER_TOL      = 0.072

MAX_SPEED        = 100.0
MIN_STOP_DUR     = 3
MAX_STOP_DUR     = 300
MIN_SAMPLE       = 3
MIN_PROB         = 0.05
# ──────────────────────────────────────────────────


def get_time_slot(timestr: str) -> str:
    h = int(timestr[8:10])
    return f"{h:02d}:00~{h+1:02d}:00"


# ──────────────────────────────────────────────────
# 정차 이벤트 탐지
# ──────────────────────────────────────────────────
def detect_stop_events(dist_list: list, speed_list: list) -> list:
    events = []
    in_stop, s_dist, s_idx = False, None, None

    for i, spd in enumerate(speed_list):
        if i == 0 or spd > MAX_SPEED:
            continue

        if not in_stop:
            if spd <= STOP_ENTER_SPEED:
                in_stop, s_dist, s_idx = True, dist_list[i], i
        else:
            if spd >= STOP_EXIT_SPEED:
                dur = (i - s_idx) * INTERVAL_SEC
                if MIN_STOP_DUR <= dur <= MAX_STOP_DUR:
                    events.append((round(s_dist, 4), round(dist_list[i-1], 4), dur))
                in_stop = False

    if in_stop:
        dur = (len(dist_list) - 1 - s_idx) * INTERVAL_SEC
        if MIN_STOP_DUR <= dur <= MAX_STOP_DUR:
            events.append((round(s_dist, 4), round(dist_list[-1], 4), dur))

    return events


# ──────────────────────────────────────────────────
# 클러스터링
# ──────────────────────────────────────────────────
def cluster_stop_events(events: list) -> list:
    clusters = []
    for s, e, dur in events:
        matched = False
        for cl in clusters:
            if abs(s - cl['_center']) <= CLUSTER_TOL:
                cl['starts'].append(s)
                cl['ends'].append(e)
                cl['durs'].append(dur)
                cl['_center'] = sum(cl['starts']) / len(cl['starts'])
                matched = True
                break
        if not matched:
            clusters.append({
                '_center': s,
                'starts':  [s],
                'ends':    [e],
                'durs':    [dur],
            })
    return clusters


# ──────────────────────────────────────────────────
# 로그 이벤트 → 패턴 인덱스 매핑
# ──────────────────────────────────────────────────
def match_events_to_patterns(events: list, patterns: list) -> list:

    matched = set()
    for s_dist, _, _ in events:
        for idx, pat in enumerate(patterns):
            if abs(s_dist - pat['startDist']) <= CLUSTER_TOL:
                matched.add(idx)
    return sorted(matched)


# ──────────────────────────────────────────────────
# 메인 파이프라인 (2패스 구조)
# ──────────────────────────────────────────────────
def build_pipeline(input_path: str, output_path: str) -> None:

    # 1패스용 수집 구조
    raw = defaultdict(lambda: defaultdict(lambda: {
        'total_dists':  [],
        'sample_count': 0,
        'all_events':   [],
        'logs':         [],   # (veh_key, per_log_events) 보관
    }))

    # ── 1패스: 데이터 수집 ────────────────────────
    with open(input_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r      = json.loads(line)
            slot   = get_time_slot(r['starttime'])
            seq    = r['seq']
            dists  = r['10secDist']
            speeds = r['10secSpeed']

            events = detect_stop_events(dists, speeds)

            bkt = raw[slot][seq]
            bkt['total_dists'].append(dists[-1] if dists else 0)
            bkt['sample_count'] += 1
            bkt['all_events'].extend(events)   # 패턴 집계용 (전체 누적)

            # 로그별 보관 (패턴 인덱스 매핑용)
            veh_key = f"{r.get('vehId', 'unknown')}__{r.get('starttime', '')}"
            bkt['logs'].append((veh_key, events))

    # ── 2패스: 패턴 확정 + 라벨링 ────────────────
    stop_patterns_out = {}
    log_labels_out    = {}

    for slot in sorted(raw.keys()):
        seqs_out    = []
        slot_labels = {}

        for seq in sorted(raw[slot].keys()):
            bkt = raw[slot][seq]
            n   = bkt['sample_count']

            if n < MIN_SAMPLE:
                continue

            total_dist = round(statistics.median(bkt['total_dists']), 4)

            # 패턴 확정 (클러스터링)
            clusters      = cluster_stop_events(bkt['all_events'])
            stop_patterns = []
            for cl in clusters:
                cnt  = len(cl['durs'])
                prob = min(cnt / n, 1.0)
                if prob < MIN_PROB:
                    continue
                stop_patterns.append({
                    "startDist":   round(sum(cl['starts']) / cnt, 4),
                    "endDist":     round(sum(cl['ends'])   / cnt, 4),
                    "probability": round(prob, 2),
                    "avgDuration": round(sum(cl['durs'])   / cnt, 1),
                })
            # probability 내림차순 -> 인덱스 0이 가장 빈번한 패턴
            stop_patterns.sort(key=lambda x: -x['probability'])

            seqs_out.append({
                "seq":          seq,
                "totalDist":    total_dist,
                "sampleCount":  n,
                "stopPatterns": stop_patterns,
            })

            # 로그별 패턴 인덱스 매핑
            log_label_map = {}
            combo_stats   = defaultdict(int)

            for veh_key, events in bkt['logs']:
                # 이 로그에서 발생한 정차가 어느 패턴(인덱스)에 해당하는지
                indices   = match_events_to_patterns(events, stop_patterns)
                combo_key = str(indices)   # "[0]", "[0, 1]", "[]"
                log_label_map[veh_key] = indices
                combo_stats[combo_key] += 1

            fast_count = sum(1 for v in log_label_map.values() if not v)
            slow_count = n - fast_count

            slot_labels[str(seq)] = {
                "logs":         log_label_map,     # preprocessor 핵심 조회 대상
                "_fast_count":  fast_count,
                "_slow_count":  slow_count,
                "_combo_stats": dict(combo_stats), # 분석/디버그용
            }

        stop_patterns_out[slot] = {"timeSlot": slot, "seqs": seqs_out}
        log_labels_out[slot]    = slot_labels

        total_pats = sum(len(s['stopPatterns']) for s in seqs_out)
        slow_total = sum(v['_slow_count'] for v in slot_labels.values())
        fast_total = sum(v['_fast_count'] for v in slot_labels.values())
        print(f"  {slot}: seq {len(seqs_out)}개 | "
              f"패턴 {total_pats}개 | "
              f"정차있음 {slow_total}건 / 정차없음 {fast_total}건")

    # ── 저장 ──────────────────────────────────────
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(
            {"stop_patterns": stop_patterns_out, "log_labels": log_labels_out},
            f, ensure_ascii=False, indent=2,
        )
    print(f"\n✓ 저장 완료: {output_path}")


# ── 실행 ──────────────────────────────────────────
INPUT  = '4st_input.jsonl'
OUTPUT = '4st_output.json'

build_pipeline(INPUT, OUTPUT)