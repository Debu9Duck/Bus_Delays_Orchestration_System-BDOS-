import json
from datetime import datetime, timedelta

FMT = "%Y%m%d%H%M%S"
FALLBACK_OFFSET_SEC = 20

input_path  = "1-1st_input.jsonl"
output_path = "1-1st_output.jsonl"

def parse_tm(s):
    return datetime.strptime(s, FMT)

def fmt_tm(dt):
    return dt.strftime(FMT)

def find_last_zero_tm(dist_list, tm_list):
    last_zero_idx = None
    for i, d in enumerate(dist_list):
        if d == 0.0:
            last_zero_idx = i
        else:
            break
    if last_zero_idx is not None:
        return tm_list[last_zero_idx]
    return None

def estimate_departure_time(dist_list, tm_list, endtime_str):
    if len(dist_list) < 1 or len(tm_list) < 1:
        return None

    MIN_SLOPE = 0.2 / 40
    MAX_SLOPE = 50 / 3600

    t_first = parse_tm(tm_list[0])
    t_last  = parse_tm(tm_list[-1])
    n       = len(tm_list)
    avg_interval = (t_last - t_first).total_seconds() / max(n - 1, 1)
    if avg_interval == 0:
        avg_interval = 10
    max_seconds_back = n * avg_interval * 2

    slope = None
    for i in range(len(dist_list) - 1):
        d0 = dist_list[i]
        d1 = dist_list[i + 1]
        t0 = parse_tm(tm_list[i])
        t1 = parse_tm(tm_list[i + 1])

        dt_sec = (t1 - t0).total_seconds()
        dd     = d1 - d0

        if dt_sec <= 0 or dd <= 0:
            continue

        actual_slope = dd / dt_sec
        slope = max(actual_slope, MIN_SLOPE)
        slope = min(slope, MAX_SLOPE)
        break

    if slope is None:
        slope = MAX_SLOPE

    d0 = dist_list[0]
    t0 = parse_tm(tm_list[0])

    seconds_back = d0 / slope
    departure    = t0 - timedelta(seconds=seconds_back)

    endtime = parse_tm(endtime_str)
    if departure >= t_first:
        return None
    if departure >= endtime:
        return None
    if seconds_back > max_seconds_back:
        return None

    return departure

def apply_fallback(original_starttime, tm_list):
    original_dt = parse_tm(original_starttime)
    candidate   = original_dt + timedelta(seconds=FALLBACK_OFFSET_SEC)
    first_tm    = parse_tm(tm_list[0])

    if candidate >= first_tm:
        return first_tm - timedelta(seconds=5), "fallback_clamp"
    return candidate, "fallback_constant"

def fix_starttime(input_path, output_path):
    corrected = fallback = zero_case = total = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            total += 1

            dist_list          = record.get("10secDist", [])
            tm_list            = record.get("10secTm", [])
            original_starttime = record.get("starttime")
            endtime            = record.get("endtime")

            new_start_dt = None

            if dist_list and tm_list and dist_list[0] == 0.0:
                last_zero_tm = find_last_zero_tm(dist_list, tm_list)
                if last_zero_tm:
                    new_start_dt = parse_tm(last_zero_tm)
                    zero_case += 1

            elif len(dist_list) < 1 or len(tm_list) < 1:
                if tm_list:
                    new_start_dt, _ = apply_fallback(original_starttime, tm_list)
                else:
                    new_start_dt = parse_tm(original_starttime) + timedelta(seconds=FALLBACK_OFFSET_SEC)
                fallback += 1

            else:
                departure = estimate_departure_time(dist_list, tm_list, endtime)
                if departure is not None:
                    new_start_dt = departure
                    corrected += 1
                else:
                    new_start_dt, _ = apply_fallback(original_starttime, tm_list)
                    fallback += 1

            if new_start_dt:
                record["starttime"] = fmt_tm(new_start_dt)

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    fix_starttime(input_path, output_path)