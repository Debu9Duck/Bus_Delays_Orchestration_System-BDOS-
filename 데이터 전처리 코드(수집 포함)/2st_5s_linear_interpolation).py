import json
import numpy as np
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))

def to_sec(t: str) -> int:
    return int(datetime.strptime(t, "%Y%m%d%H%M%S").replace(tzinfo=KST).timestamp())

def from_sec(s) -> str:
    return datetime.fromtimestamp(int(s), tz=KST).strftime("%Y%m%d%H%M%S")

def validate_rec(rec: dict) -> None:
    tm = rec["10secTm"]
    if tm[0] != rec["starttime"]:
        raise ValueError(
            f"[검증 실패] vehId={rec['vehId']} seq={rec['seq']} | "
            f"10secTm[0]='{tm[0]}' ≠ starttime='{rec['starttime']}'"
        )
    if tm[-1] != rec["endtime"]:
        raise ValueError(
            f"[검증 실패] vehId={rec['vehId']} seq={rec['seq']} | "
            f"10secTm[-1]='{tm[-1]}' ≠ endtime='{rec['endtime']}'"
        )

def resample_5sec(rec: dict) -> dict:
    validate_rec(rec)

    t_raw = np.array([to_sec(x) for x in rec["10secTm"]], dtype=np.int64)
    d_raw = np.array(rec["10secDist"], dtype=float)

    if len(t_raw) < 2:
        return rec

    t_grid = np.arange(t_raw[0], t_raw[-1] + 5, 5, dtype=np.int64)
    t_grid = t_grid[t_grid <= t_raw[-1]]
    if t_grid[-1] != t_raw[-1]:
        t_grid = np.append(t_grid, t_raw[-1])

    d_grid = np.interp(t_grid, t_raw, d_raw)

    result = {
        k: v for k, v in rec.items()
        if k not in ("10secDist", "10secTm")
    }
    result["10secDist"] = [round(v, 6) for v in d_grid.tolist()]
    result["10secTm"]   = [from_sec(x) for x in t_grid]
    return result

# -----------------------------
# JSONL 파일 처리
# -----------------------------
INPUT_PATH  = "2st_input.jsonl"
OUTPUT_PATH = "2st_output.jsonl"

input_recs  = {}
output_recs = {}

total = 0
with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
     open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

    for line_num, line in enumerate(fin, start=1):
        line = line.strip()
        if not line:
            continue

        rec = json.loads(line)
        try:
            resampled = resample_5sec(rec)
            fout.write(json.dumps(resampled, ensure_ascii=False) + "\n")
            total += 1

            key = (rec["vehId"], rec["seq"])
            input_recs[key]  = rec
            output_recs[key] = resampled

        except ValueError as e:
            print(e)
            raise SystemExit(1)

print(f"완료: {total}건 처리 → {OUTPUT_PATH}")

# -----------------------------
# 사후 검증
# -----------------------------
ROUTE_PATH = "route_160.json"

with open(ROUTE_PATH, "r", encoding="utf-8") as f:
    route_data = json.load(f)

seq_to_dist = {s["seq"]: s["dist"] for s in route_data["stations"]}

all_ok = True

for key, in_rec in input_recs.items():
    out_rec   = output_recs[key]
    vehId     = in_rec["vehId"]
    seq       = in_rec["seq"]
    toSect    = in_rec["toSect"]

    in_start  = in_rec["10secTm"][0]
    out_start = out_rec["10secTm"][0]
    in_end    = in_rec["10secTm"][-1]
    out_end   = out_rec["10secTm"][-1]
    start_ok  = "ok" if in_start == out_start else "no"
    end_ok    = "ok" if in_end   == out_end   else "no"

    out_dist_km   = out_rec["10secDist"][-1]
    out_dist_m    = round(out_dist_km * 1000, 3)
    route_dist_m  = seq_to_dist.get(toSect)
    if route_dist_m is None:
        dist_ok = "?"
    else:
        dist_ok = "ok" if abs(out_dist_m - route_dist_m) < 1.0 else "no"

    if "no" in (start_ok, end_ok, dist_ok):
        all_ok = False
        print(f"[검증 실패] vehId={vehId} seq={seq} toSect={toSect} | "
              f"시작={start_ok} 끝={end_ok} 거리={dist_ok}")

print(f"검증 결과: {'전체 통과' if all_ok else '일부 실패 — 위 항목 확인 필요'}")