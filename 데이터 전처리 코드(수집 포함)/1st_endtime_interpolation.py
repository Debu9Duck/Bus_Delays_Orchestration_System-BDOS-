import json
from datetime import datetime, timedelta

with open("route_160.json", "r", encoding="utf-8") as f:
    route = json.load(f)

dist_map = {s["seq"]: s["dist"] for s in route["stations"]}

results = []

def process_record(rec):
    dist = rec["10secDist"]
    times = rec["10secTm"]

    if len(dist) == 0:
        return rec

    to_seq = rec["toSect"]
    end_dist = dist_map[to_seq] / 1000  # 목적지 누적 거리 (km)
    
    # ------------------------------------------------
    # 1. 기존 데이터 준비
    # ------------------------------------------------
    raw_dist = list(dist)
    raw_time = list(times)
    FMT = "%Y%m%d%H%M%S"

    # ------------------------------------------------
    # 2. endtime 보정 로직 (선형 보간 및 속도 제한)
    # ------------------------------------------------
    last_dist = raw_dist[-1]
    last_time_str = raw_time[-1]
    last_time_dt = datetime.strptime(last_time_str, FMT)
    
    # 잔여 거리 (km)
    remaining_dist = end_dist - last_dist
    
    # 이전 구간의 속도 계산 (보간의 기준점)
    if len(raw_dist) > 1:
        prev_d = raw_dist[-1] - raw_dist[-2]
        t_prev_0 = datetime.strptime(raw_time[-2], FMT)
        t_prev_1 = datetime.strptime(raw_time[-1], FMT)
        dt_prev = (t_prev_1 - t_prev_0).total_seconds()
        
        # 이전 속도 (km/h), dt가 0인 경우 대비
        v_prev = (prev_d / dt_prev * 3600) if dt_prev > 0 else 35.0
    else:
        v_prev = 35.0 # 이전 데이터가 없으면 중간값인 35km/h 가정

    # 요구사항: 20km/h <= v <= 50km/h 제한
    v_clamped = max(20.0, min(50.0, v_prev))
    
    # 시간 간격 계산: dt(sec) = (거리 / 속도) * 3600
    # 잔여 거리가 거의 0인 경우 최소 1초 부여
    dt_needed = (remaining_dist / v_clamped * 3600) if remaining_dist > 0 else 1
    
    # 새로운 endtime 계산
    new_endtime_dt = last_time_dt + timedelta(seconds=dt_needed)
    new_endtime_str = new_endtime_dt.strftime(FMT)

    # ------------------------------------------------
    # 3. 데이터 패딩 (0.0 및 보정된 end_dist/endtime)
    # ------------------------------------------------
    # starttime은 기존 값 유지, endtime은 계산된 값으로 교체
    new_dist = [0.0] + raw_dist + [end_dist]
    new_time = [rec["starttime"]] + raw_time + [new_endtime_str]

    rec["10secDist"] = new_dist
    rec["10secTm"] = new_time
    # 레코드 자체의 endtime 필드도 동기화 (필요 시)
    rec["endtime"] = new_endtime_str 

    return rec


total = 0
with open("1-2st_input.jsonl", "r", encoding="utf-8") as f_in, \
     open("1-2st_output.jsonl", "w", encoding="utf-8") as f_out:

    for line in f_in:
        rec = json.loads(line)
        rec = process_record(rec)
        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        total += 1

print(f"완료: {total}건 처리 → 1st_output.jsonl")

mismatch = 0

for rec in results:
    to_seq = rec["toSect"]

    expected = dist_map[to_seq] / 1000
    actual = rec["10secDist"][-1]

    if abs(expected - actual) > 1e-6:
        mismatch += 1
        print({
            "vehId": rec["vehId"],
            "seq": rec["seq"],
            "toSect": to_seq,
            "expected": expected,
            "actual": actual
        })

if mismatch == 0:
    print("ROUTE CHECK: ALL MATCH")
else:
    print(f"ROUTE CHECK: MISMATCH {mismatch}건")