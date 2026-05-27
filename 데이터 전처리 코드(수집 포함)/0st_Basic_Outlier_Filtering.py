import json
import numpy as np
from collections import defaultdict
from datetime import datetime

def parse_time(ts):
    return datetime.strptime(ts, "%Y%m%d%H%M%S")

input_file = "0st_input.jsonl"
output_file = "0st_output.jsonl"
route_file = "route_160.json"

with open(route_file, "r", encoding="utf-8") as f:
    route_data = json.load(f)

stations = route_data["stations"] if "stations" in route_data else route_data
dist_map = {s["seq"]: s["dist"] for s in stations}

sessions = []

count_no_data      = 0
count_backwards    = 0
count_start_error  = 0
count_end_error    = 0
count_parse_error  = 0

lines_no_data     = []
lines_backwards   = []
lines_start_error = []
lines_end_error   = []
lines_parse_error = []

with open(input_file, "r", encoding="utf-8") as f_in, \
     open(output_file, "w", encoding="utf-8") as f_out:

    for line_no, line in enumerate(f_in, start=1):
        line = line.strip()
        if not line:
            continue

        s = json.loads(line)
        dist_list = s.get("10secDist", [])
        tm_list   = s.get("10secTm", [])

        if not dist_list or not tm_list:
            count_no_data += 1
            lines_no_data.append(line_no)
            continue

        try:
            start_t = parse_time(s["starttime"])
            end_t   = parse_time(s["endtime"])
            tms     = [parse_time(t) for t in tm_list]

            is_backwards   = any(tms[i] > tms[i+1] for i in range(len(tms)-1))
            is_start_error = start_t > tms[0]
            is_end_error   = end_t < tms[-1]

            if is_backwards:
                count_backwards += 1
                lines_backwards.append(line_no)
            if is_start_error:
                count_start_error += 1
                lines_start_error.append(line_no)
            if is_end_error:
                count_end_error += 1
                lines_end_error.append(line_no)

            if is_backwards or is_start_error or is_end_error:
                continue

            sessions.append(s)
            f_out.write(json.dumps(s, ensure_ascii=False) + "\n")

        except (ValueError, IndexError):
            count_parse_error += 1
            lines_parse_error.append(line_no)
            continue

def fmt(label, count, lines):
    preview = str(lines[:10])
    suffix = "..." if len(lines) > 10 else ""
    return f"{label}{count}건  → 라인 {preview}{suffix}"

total_filtered = count_no_data + count_backwards + count_start_error + count_end_error + count_parse_error

print("=" * 50)
print("[ 필터링 결과 ]")
print("=" * 50)
print(fmt(" 데이터 누락:      ", count_no_data,     lines_no_data))
print(fmt(" 시간 역행:        ", count_backwards,   lines_backwards))
print(fmt(" starttime 오류:   ", count_start_error, lines_start_error))
print(fmt(" endtime 오류:     ", count_end_error,   lines_end_error))
print(fmt(" 파싱 오류:        ", count_parse_error, lines_parse_error))
print("-" * 50)
print(f"전체 삭제 라인 수:  {total_filtered}건  (중복 포함)")
print(f"정상 저장 라인 수:  {len(sessions)}건")
print("=" * 50)

# --- 분석 파트 (로직 유지, 출력 제거) ---
total_points = 0
valid_sessions = 0
toSect_groups = defaultdict(list)

for s in sessions:
    toSect    = s["toSect"]
    dist_list = s["10secDist"]

    valid_sessions += 1
    total_points   += len(dist_list)

    last_dist = dist_list[-1] * 1000
    toSect_groups[toSect].append(last_dist)

results = []
for toSect in sorted(toSect_groups.keys()):
    values      = toSect_groups[toSect]
    actual_dist = dist_map.get(toSect)
    if actual_dist is None:
        continue

    mean       = np.mean(values)
    std        = np.std(values)
    error      = mean - actual_dist
    error_rate = (error / actual_dist) * 100 if actual_dist else 0

    results.append({
        "toSect": toSect, "actual_dist": actual_dist, "mean": mean,
        "std": std, "error": error, "error_rate": error_rate, "count": len(values)
    })