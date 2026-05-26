import sys
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# ── [FIX-8] stdout 즉시 flush ──
sys.stdout.reconfigure(line_buffering=True)

# ── 설정 ──────────────────────────────────────
API_URL      = "http://ws.bus.go.kr/api/rest/buspos/getBusPosByRtid"
BUS_ROUTE_ID = 100100033
SERVICE_KEY = os.environ["SERVICE_KEY"]

NORM_PATH     = Path("estimator_bus_travel_pattern.json")
STOP_PAT_PATH = Path("estimator_stop_patterns.json")
STATIONS_PATH = Path("estimator_stations.json")

DELAY_LOG_PATH = Path("delay_log.jsonl")
DWELL_LOG_PATH = Path("dwell_log.jsonl")

MIN_BASELINE_SAMPLE    = 3
MIN_STARTTIME_SAMPLE   = 3
STOP_THRESHOLD         = 0.5
BUCKET_SIZE            = 0.01
RECOVERY_THRESHOLD     = 600
MIN_DIST_FOR_STARTTIME = 0.15

STARTTIME_USE_DATATM_THRESHOLD = 0.05

STOP_ENTER_SPEED = 5.7
STOP_EXIT_SPEED  = 13.42

# ── [수정 21] 동적 percentile tracking 설정 ──
PCT_EMA_ALPHA_UP   = 0.3
PCT_EMA_ALPHA_DOWN = 0.25
PCT_LAG            = 10
PCT_REF_MIN        = 20.0
PCT_REF_MAX        = 75.0

# ── [수정 24] 저속/정지 구간 seg_delay 감쇠 ──
STOPPED_DELAY_FACTOR  = 0.7
STOPPED_KMH_THRESHOLD = 1.0

# ── [수정 25] 정지 중 seg_delay delta clamp ──
STOPPED_SEG_DELAY_DELTA_MAX = 3.0

# ── [수정 29] 정지 중 actual_elapsed 상한 계수 ──
STOPPED_ELAPSED_CAP_FACTOR = 1.5

# ── [수정 31] slow EMA 설정 ──────────────────
SLOW_EMA_ALPHA      = 0.05
SLOW_PCT_MIN        = 12.0
SLOW_PCT_MAX        = 86.0
ARR_FALLBACK_SPREAD = 0.15

# ── [수정 33] 미세/이상치 분기 기준 (분위수) ─
MICRO_ANOMALY_PCT_HI = 86.0
MICRO_ANOMALY_PCT_LO = 12.0

MICRO_SEG_DELTA_MAX = 2.0
MICRO_DELAY_CAP     = 60.0


# ── [수정 34] 권장 속도 설정 ─────────────────
TARGET_SPEED_MIN_DELAY = 5.0
TARGET_SPEED_MAX_KMH   = 80.0
TARGET_SPEED_MIN_KMH   = 1.0

# ── 전역 캐시 ─────────────────────────────────
_starttime_cache: dict[tuple[str, int], datetime] = {}

_last_log_cache:       dict[str, dict]     = {}
_first_seq_cache:      dict[str, int]      = {}
_prev_pos_cache:       dict[str, tuple]    = {}
_stop_state_cache:     dict[str, bool]     = {}
_prev_stop_flag_cache: dict[str, str]      = {}
_dwell_start_cache:    dict[str, datetime] = {}

_stuck_tracker: dict[str, dict] = {}
STUCK_THRESHOLD = 30

# ── [수정 18] 출력 게이팅 캐시 ───────────────
_output_enabled:  dict[str, bool] = {}
_in_stop_silence: dict[str, bool] = {}

# ── [수정 30] 회차지 처리: leg 상태 캐시 ─────
_leg_cache: dict[str, str | None] = {}

# ── [수정 21] fast percentile tracking 캐시 ──
_ema_pct_cache: dict[str, float] = {}

# ── [수정 31] slow percentile tracking 캐시 ──
_slow_ema_pct_cache: dict[str, float] = {}

# ── [수정 32/33] 미세 지연 캐시 ──────────────
_anomaly_seq_cache:    dict[str, set]        = {}
_micro_seg_cache:      dict[str, float]      = {}
_micro_cum_cache:      dict[str, float]      = {}
_micro_cum_restored:   dict[str, bool]       = {}
_micro_prev_seq_cache: dict[str, int]        = {}


# ── 유틸 ──────────────────────────────────────
def parse_time(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d%H%M%S")

def get_slot(dt: datetime) -> str:
    h = dt.hour
    return f"{h:02d}:00~{h+1:02d}:00"

def bucket_key(dist: float) -> float:
    return round(round(dist / BUCKET_SIZE) * BUCKET_SIZE, 6)

def dist_str(d: float) -> str:
    return f"{d:.6f}"

def adjacent_slots(slot: str) -> list:
    h = int(slot[:2])
    out = []
    for dh in [1, -1, 2, -2]:
        nh = h + dh
        if 0 <= nh <= 23:
            out.append(f"{nh:02d}:00~{nh+1:02d}:00")
    return out


# ── [수정 30] 회차지 전환 처리 ───────────────

def _reset_vehicle_state(veh_id: str) -> None:
    keys_to_del = [k for k in _starttime_cache if k[0] == veh_id]
    for k in keys_to_del:
        del _starttime_cache[k]

    _last_log_cache.pop(veh_id, None)
    _first_seq_cache.pop(veh_id, None)
    _prev_pos_cache.pop(veh_id, None)
    _stop_state_cache.pop(veh_id, None)
    _prev_stop_flag_cache.pop(veh_id, None)
    _dwell_start_cache.pop(veh_id, None)

    _ema_pct_cache[veh_id]      = 50.0
    _slow_ema_pct_cache[veh_id] = 50.0
    _output_enabled[veh_id]     = False
    _in_stop_silence[veh_id]    = False

    if veh_id in _stuck_tracker:
        del _stuck_tracker[veh_id]

    _anomaly_seq_cache[veh_id]    = set()
    _micro_seg_cache[veh_id]      = 0.0
    _micro_cum_cache[veh_id]      = 0.0
    _micro_cum_restored[veh_id]   = False
    _micro_prev_seq_cache.pop(veh_id, None)


def check_and_handle_turn(veh_id: str, seq: int, turn_seq: int,
                          data_tm: datetime) -> bool:
    current_leg = "outbound" if seq <= turn_seq else "inbound"
    prev_leg    = _leg_cache.get(veh_id)
    _leg_cache[veh_id] = current_leg

    if prev_leg is None:
        return False

    if prev_leg == "outbound" and current_leg == "inbound":
        ts = data_tm.strftime("%H:%M:%S")
        print(f"\n{'='*60}")
        print(f"[회차 전환] vehId={veh_id}  seq={seq}  {ts}")
        print(f"  outbound 완료 → inbound 재출발 감지")
        print(f"  → 모든 상태 초기화 + warm-up 재시작 + cum_delay=0 리셋")
        print(f"  → micro_delay도 0으로 리셋 (leg 간 미세 지연 이월 방지)")
        print(f"{'='*60}\n")
        _reset_vehicle_state(veh_id)
        return True

    return False


# ── 데이터 로드 ───────────────────────────────
def load_data():
    with open(NORM_PATH,     encoding="utf-8") as f:
        norm = json.load(f)
    with open(STOP_PAT_PATH, encoding="utf-8") as f:
        raw_stop = json.load(f)
    with open(STATIONS_PATH, encoding="utf-8") as f:
        raw_st = json.load(f)

    stations = {s["seq"]: s for s in raw_st["stations"]}
    turn_seq = next(s["seq"] for s in raw_st["stations"] if s["is_turn"])

    stop_pat = {}
    for slot, slot_data in raw_stop.get("stop_patterns", {}).items():
        stop_pat[slot] = {s["seq"]: s for s in slot_data.get("seqs", [])}

    return norm, stop_pat, stations, turn_seq


# ── 시간대 조회: norm ─────────────────────────
def get_norm_seq(norm: dict, slot: str, seq: int) -> dict | None:
    def _try(s):
        data = norm.get(s, {}).get(str(seq))
        if data is None:
            return None
        if data.get("baseline", {}).get("sample_count", 0) < MIN_BASELINE_SAMPLE:
            return None
        return data

    result = _try(slot)
    if result:
        return result
    for s in adjacent_slots(slot):
        result = _try(s)
        if result:
            return result
    return None


# ── 시간대 조회: stop_patterns ────────────────
def get_stop_seq(stop_pat: dict, slot: str, seq: int) -> dict | None:
    def _try(s):
        return stop_pat.get(s, {}).get(seq)

    result = _try(slot)
    if result:
        return result
    for s in adjacent_slots(slot):
        result = _try(s)
        if result:
            return result
    return None


# ── elapsed 조회 ──────────────────────────────
def lookup_elapsed(dist_to_elapsed: dict, dist: float,
                   pct: str = "p50") -> float | None:
    row = dist_to_elapsed.get(dist_str(bucket_key(dist)))
    return row[pct] if row else None


def get_elapsed(norm_seq: dict, combo_key: str,
                dist: float, pct: str = "p50") -> float | None:
    combo = norm_seq.get("combos", {}).get(combo_key)
    if combo:
        val = lookup_elapsed(combo["dist_to_elapsed"], dist, pct)
        if val is not None:
            return val
    bl = norm_seq.get("baseline", {})
    return lookup_elapsed(bl.get("dist_to_elapsed", {}), dist, pct)


# ── combo_key + p_stop 판별 ───────────────────
def infer_combo(seg: dict | None, sect_dist: float) -> tuple:
    if seg is None:
        return "[]", 0.0, False

    passed   = []
    p_future = 0.0
    in_zone  = False

    for idx, p in enumerate(seg["stopPatterns"]):
        already_passed = p["endDist"] <= sect_dist
        _in_zone       = p["startDist"] <= sect_dist <= p["endDist"]

        if already_passed or _in_zone:
            passed.append(idx)
        else:
            p_future += p["probability"]

        if _in_zone:
            in_zone = True

    return str(passed), min(p_future, 1.0), in_zone


# ── 정지 감지 (hysteresis) ────────────────────
def detect_stop(veh_id: str, stop_flag: str, in_zone: bool,
                current_kmh: float | None) -> bool:
    prev_stopped = _stop_state_cache.get(veh_id, False)

    if stop_flag == "1":
        _stop_state_cache[veh_id] = True
        return True

    if current_kmh is None:
        return prev_stopped

    if current_kmh >= STOP_EXIT_SPEED:
        _stop_state_cache[veh_id] = False
        return False

    if current_kmh <= STOP_ENTER_SPEED:
        if in_zone or prev_stopped:
            _stop_state_cache[veh_id] = True
            return True

    return prev_stopped


# ── 속도 계산 ─────────────────────────────────
def calc_speeds(veh_id: str, sect_dist: float, data_tm: datetime,
                expected_elapsed: float | None) -> tuple[float | None, float | None]:
    MAX_REALISTIC_KMH = 80.0

    current_kmh = None
    prev = _prev_pos_cache.get(veh_id)
    if prev is not None:
        prev_dist, prev_tm = prev
        dt_sec = (data_tm - prev_tm).total_seconds()
        dd_km  = sect_dist - prev_dist
        if dt_sec > 0 and dd_km >= 0:
            raw_kmh = dd_km / dt_sec * 3600
            if raw_kmh <= MAX_REALISTIC_KMH:
                current_kmh = raw_kmh

    _prev_pos_cache[veh_id] = (sect_dist, data_tm)

    norm_kmh = None
    if expected_elapsed and expected_elapsed > 0 and sect_dist > 0:
        norm_kmh = sect_dist / expected_elapsed * 3600

    return current_kmh, norm_kmh


# ── [수정 B] starttime 추정 ───────────────────
def get_or_estimate_starttime(veh_id: str, seq: int, sect_dist: float,
                               data_tm: datetime, norm_seq: dict,
                               combo_key: str) -> datetime | None:
    cache_key = (veh_id, seq)

    if cache_key in _starttime_cache:
        return _starttime_cache[cache_key]

    if sect_dist <= STARTTIME_USE_DATATM_THRESHOLD:
        _starttime_cache[cache_key] = data_tm
        return data_tm

    if sect_dist < MIN_DIST_FOR_STARTTIME:
        return None

    sample = norm_seq.get("baseline", {}).get("sample_count", 0)
    if sample < MIN_STARTTIME_SAMPLE:
        return None

    elapsed_p10 = get_elapsed(norm_seq, combo_key, sect_dist, "p10")
    elapsed_p50 = get_elapsed(norm_seq, combo_key, sect_dist, "p50")
    elapsed_p90 = get_elapsed(norm_seq, combo_key, sect_dist, "p90")

    if elapsed_p50 is None:
        return None

    if elapsed_p10 is not None and elapsed_p90 is not None:
        lo     = min(elapsed_p10, elapsed_p90)
        hi     = max(elapsed_p10, elapsed_p90)
        margin = (hi - lo) * 0.2
        if not (lo - margin <= elapsed_p50 <= hi + margin):
            return None

    smoothed_pct = _ema_pct_cache.get(veh_id, 50.0)
    ref_pct = max(PCT_REF_MIN, min(PCT_REF_MAX, smoothed_pct - PCT_LAG))

    if elapsed_p10 is not None and elapsed_p90 is not None:
        interpolated_elapsed = _interpolate_expected(ref_pct, elapsed_p10, elapsed_p50, elapsed_p90)
        interpolated_elapsed = max(interpolated_elapsed, elapsed_p10)
    else:
        interpolated_elapsed = elapsed_p50

    starttime = data_tm - timedelta(seconds=interpolated_elapsed)
    _starttime_cache[cache_key] = starttime
    return starttime


# ── [수정 21 + 23 + 25] 동적 percentile tracking ───

def _estimate_raw_percentile(actual_elapsed: float, norm_seq: dict,
                              combo_key: str, dist: float) -> float:
    p10 = get_elapsed(norm_seq, combo_key, dist, "p10")
    p50 = get_elapsed(norm_seq, combo_key, dist, "p50")
    p90 = get_elapsed(norm_seq, combo_key, dist, "p90")

    if None in (p10, p50, p90):
        return 50.0

    if actual_elapsed <= p10:
        return 10.0
    elif actual_elapsed <= p50:
        span = p50 - p10
        if span < 1e-6:
            return 30.0
        return 10.0 + 40.0 * (actual_elapsed - p10) / span
    elif actual_elapsed <= p90:
        span = p90 - p50
        if span < 1e-6:
            return 70.0
        return 50.0 + 40.0 * (actual_elapsed - p50) / span
    else:
        return 90.0


def _update_ema_percentile(veh_id: str, raw_pct: float,
                            is_stopped: bool) -> float:
    prev_fast = _ema_pct_cache.get(veh_id, 50.0)

    if not is_stopped:
        alpha_fast = PCT_EMA_ALPHA_UP if raw_pct > prev_fast else PCT_EMA_ALPHA_DOWN
        new_fast   = alpha_fast * raw_pct + (1.0 - alpha_fast) * prev_fast
        _ema_pct_cache[veh_id] = new_fast
    else:
        new_fast = prev_fast

    if not is_stopped:
        prev_slow = _slow_ema_pct_cache.get(veh_id, 50.0)
        new_slow  = SLOW_EMA_ALPHA * raw_pct + (1.0 - SLOW_EMA_ALPHA) * prev_slow
        _slow_ema_pct_cache[veh_id] = new_slow

    return new_fast


def _interpolate_expected(ref_pct: float, p10: float,
                           p50: float, p90: float) -> float:
    ref_pct = max(10.0, min(90.0, ref_pct))
    if ref_pct <= 50.0:
        span = p50 - p10
        if span < 1e-6:
            return p10
        return p10 + (p50 - p10) * (ref_pct - 10.0) / 40.0
    else:
        span = p90 - p50
        if span < 1e-6:
            return p50
        return p50 + (p90 - p50) * (ref_pct - 50.0) / 40.0


# ── 현재 구간 딜레이 계산 ─────────────────────
MAX_SEG_DELAY_ABS = 300.0

def calc_seg_delay(veh_id: str, sect_dist: float, actual_elapsed: float,
                   norm_seq: dict, combo_key: str,
                   is_stopped: bool = False,
                   current_kmh: float | None = None,
                   norm_kmh: float | None = None
                   ) -> tuple[float, float | None, float]:
    p10 = get_elapsed(norm_seq, combo_key, sect_dist, "p10")
    p50 = get_elapsed(norm_seq, combo_key, sect_dist, "p50")
    p90 = get_elapsed(norm_seq, combo_key, sect_dist, "p90")

    if None in (p10, p50, p90):
        expected = p50
        raw_pct  = 50.0
        if expected is None:
            return 0.0, None, raw_pct
    else:
        raw_pct      = _estimate_raw_percentile(actual_elapsed, norm_seq, combo_key, sect_dist)
        smoothed_pct = _update_ema_percentile(veh_id, raw_pct, is_stopped)
        ref_pct      = max(PCT_REF_MIN, min(PCT_REF_MAX, smoothed_pct - PCT_LAG))
        expected     = _interpolate_expected(ref_pct, p10, p50, p90)
        expected     = max(expected, p10)

    raw     = actual_elapsed - expected
    clamped = max(-MAX_SEG_DELAY_ABS, min(MAX_SEG_DELAY_ABS, raw))

    if is_stopped:
        last = read_last_log(veh_id)
        if last is not None:
            prev_seg = last.get("seg_delay", 0.0)
            delta = clamped - prev_seg
            if abs(delta) > STOPPED_SEG_DELAY_DELTA_MAX:
                clamped = prev_seg + STOPPED_SEG_DELAY_DELTA_MAX * (1 if delta > 0 else -1)

    if (current_kmh is not None and norm_kmh is not None
            and norm_kmh > 0.5 and not is_stopped):
        speed_ratio = current_kmh / norm_kmh
        if speed_ratio > 1.3:
            clamped = min(clamped, 0.0)
        elif speed_ratio < 0.7 and clamped < 0:
            clamped = max(clamped, 0.0)

    return clamped, expected, raw_pct


# ── [수정 32/33] 미세 지연 계산 ──────────────

def calc_micro_seg(veh_id: str, seq: int, actual_elapsed: float,
                   norm_seq: dict, combo_key: str, sect_dist: float,
                   stop_flag: str, is_stopped: bool,
                   raw_pct: float) -> float:

    is_anomaly = (
        raw_pct > MICRO_ANOMALY_PCT_HI or raw_pct < MICRO_ANOMALY_PCT_LO
    )

    # _anomaly_seq_cache: 현재 seq의 anomaly 상태를 저장 (출력용 태그에 활용)
    if veh_id not in _anomaly_seq_cache:
        _anomaly_seq_cache[veh_id] = set()

    if is_anomaly:
        _anomaly_seq_cache[veh_id].add(seq)
        _micro_seg_cache[veh_id] = 0.0
        return 0.0
    else:
        _anomaly_seq_cache[veh_id].discard(seq)

    if stop_flag == "1":
        return _micro_seg_cache.get(veh_id, 0.0)

    p50_elapsed = get_elapsed(norm_seq, combo_key, sect_dist, "p50")
    if p50_elapsed is None or p50_elapsed <= 0:
        return _micro_seg_cache.get(veh_id, 0.0)

    raw_micro = actual_elapsed - p50_elapsed

    if is_stopped:
        prev_micro = _micro_seg_cache.get(veh_id, 0.0)
        delta = raw_micro - prev_micro
        if abs(delta) > MICRO_SEG_DELTA_MAX:
            raw_micro = prev_micro + MICRO_SEG_DELTA_MAX * (1 if delta > 0 else -1)

    _micro_seg_cache[veh_id] = raw_micro
    return raw_micro


def _flush_micro_seg(veh_id: str, prev_seq: int) -> None:
    micro_seg = _micro_seg_cache.get(veh_id, 0.0)
    prev_cum  = _micro_cum_cache.get(veh_id, 0.0)
    new_cum   = max(-MICRO_DELAY_CAP, min(MICRO_DELAY_CAP, prev_cum + micro_seg))
    _micro_cum_cache[veh_id] = new_cum
    _micro_seg_cache[veh_id] = 0.0


# ── [FIX-5] micro_cum 재시작 복원 ────────────

def _restore_micro_cum(veh_id: str) -> None:
    if _micro_cum_restored.get(veh_id, False):
        return

    _micro_cum_restored[veh_id] = True

    if not DELAY_LOG_PATH.exists():
        return

    last_micro_cum = None
    try:
        with open(DELAY_LOG_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("vehId") == veh_id:
                        val = rec.get("micro_cum")
                        if val is not None:
                            last_micro_cum = float(val)
                except (json.JSONDecodeError, ValueError):
                    continue
    except Exception:
        return

    if last_micro_cum is not None:
        clamped = max(-MICRO_DELAY_CAP, min(MICRO_DELAY_CAP, last_micro_cum))
        _micro_cum_cache[veh_id] = clamped
        print(
            f"[{datetime.now():%H:%M:%S}] [RESTORE] vehId={veh_id} "
            f"micro_cum 복원: {clamped:+.2f}초 (파일 원본: {last_micro_cum:+.2f}초)"
        )


# ── 통합 JSONL 딜레이 로그 ────────────────────
def read_last_log(veh_id: str) -> dict | None:
    cached = _last_log_cache.get(veh_id)
    if cached is not None:
        return cached

    if not DELAY_LOG_PATH.exists():
        return None
    try:
        last_record = None
        with open(DELAY_LOG_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("vehId") == veh_id:
                        last_record = rec
                except json.JSONDecodeError:
                    continue
        if last_record is not None:
            _last_log_cache[veh_id] = last_record
        return last_record
    except Exception:
        return None


def append_log(veh_id: str, record: dict) -> None:
    record["vehId"] = veh_id
    _last_log_cache[veh_id] = record
    with open(DELAY_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def confirm_last_log(veh_id: str) -> None:
    last = read_last_log(veh_id)
    if last and not last.get("confirmed", False):
        last["confirmed"] = True
        append_log(veh_id, last)


def update_delay_log(veh_id: str, seq: int, sect_dist: float,
                     seg_delay: float, data_tm: datetime,
                     stop_flag: str = "0",
                     micro_seg: float = 0.0,
                     micro_cum: float = 0.0,
                     raw_pct: float = 50.0) -> float | None:
    last      = read_last_log(veh_id)
    ts        = data_tm.strftime("%Y%m%d%H%M%S")
    first_seq = _first_seq_cache.get(veh_id)

    if last is None:
        _first_seq_cache[veh_id] = seq
        append_log(veh_id, {
            "ts": ts, "seq": seq, "sect_dist": sect_dist,
            "seg_delay": round(seg_delay, 2),
            "cum_delay": 0.0,
            "micro_seg": round(micro_seg, 2),
            "micro_cum": round(micro_cum, 2),
            "total_delay": 0.0,
            "raw_pct": round(raw_pct, 1),
            "stop_flag": stop_flag,
            "confirmed": False,
            "warmup": True,
        })
        return None

    prev_seq       = last["seq"]
    prev_seg_delay = last["seg_delay"]
    prev_cum_delay = last["cum_delay"]

    if first_seq is not None and prev_seq == first_seq and seq == first_seq:
        append_log(veh_id, {
            "ts": ts, "seq": seq, "sect_dist": sect_dist,
            "seg_delay": round(seg_delay, 2),
            "cum_delay": 0.0,
            "micro_seg": round(micro_seg, 2),
            "micro_cum": round(micro_cum, 2),
            "total_delay": round(micro_cum, 2),
            "raw_pct": round(raw_pct, 1),
            "stop_flag": stop_flag,
            "confirmed": False,
            "warmup": True,
        })
        return None

    if first_seq is not None and prev_seq == first_seq and seq != first_seq:
        confirm_last_log(veh_id)
        _output_enabled[veh_id] = True
        _ema_pct_cache[veh_id] = 50.0

        append_log(veh_id, {
            "ts": ts, "seq": seq, "sect_dist": sect_dist,
            "seg_delay": 0.0,
            "cum_delay": 0.0,
            "micro_seg": round(micro_seg, 2),
            "micro_cum": round(micro_cum, 2),
            "total_delay": round(micro_cum, 2),
            "raw_pct": round(raw_pct, 1),
            "stop_flag": stop_flag,
            "confirmed": False,
        })
        return 0.0

    if prev_seq != seq:
        confirm_last_log(veh_id)
        _ema_pct_cache[veh_id] = 50.0
        cum_delay = prev_cum_delay
        append_log(veh_id, {
            "ts": ts, "seq": seq, "sect_dist": sect_dist,
            "seg_delay": round(seg_delay, 2),
            "cum_delay": round(cum_delay, 2),
            "micro_seg": round(micro_seg, 2),
            "micro_cum": round(micro_cum, 2),
            "total_delay": round(cum_delay + micro_cum, 2),
            "raw_pct": round(raw_pct, 1),
            "stop_flag": stop_flag,
            "confirmed": False,
            "seq_transition": True,
        })
    else:
        delta     = seg_delay - prev_seg_delay
        cum_delay = prev_cum_delay + delta
        append_log(veh_id, {
            "ts": ts, "seq": seq, "sect_dist": sect_dist,
            "seg_delay": round(seg_delay, 2),
            "cum_delay": round(cum_delay, 2),
            "micro_seg": round(micro_seg, 2),
            "micro_cum": round(micro_cum, 2),
            "total_delay": round(cum_delay + micro_cum, 2),
            "raw_pct": round(raw_pct, 1),
            "stop_flag": stop_flag,
            "confirmed": False,
        })

    return cum_delay


# ── 정류장 정차 시간 로그 ─────────────────────
def update_dwell_log(veh_id: str, seq: int, stop_flag: str,
                     data_tm: datetime) -> None:
    prev_flag = _prev_stop_flag_cache.get(veh_id)
    _prev_stop_flag_cache[veh_id] = stop_flag

    if prev_flag is None:
        return

    ts = data_tm.strftime("%Y%m%d%H%M%S")

    if prev_flag == "0" and stop_flag == "1":
        _dwell_start_cache[veh_id] = data_tm
        return

    if prev_flag == "1" and stop_flag == "0":
        start_tm = _dwell_start_cache.pop(veh_id, None)
        if start_tm is None:
            return

        dwell_sec = (data_tm - start_tm).total_seconds()
        record = {
            "vehId":          veh_id,
            "seq":            seq,
            "stop_start_tm":  start_tm.strftime("%Y%m%d%H%M%S"),
            "stop_end_tm":    ts,
            "dwell_time_sec": round(dwell_sec, 1),
        }
        with open(DWELL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── 남은 구간 ETA ─────────────────────────────

def _interp_arrival(slow_pct: float, arr: dict) -> float:
    p50_val = arr.get("p50") or 0.0
    if p50_val <= 0.0:
        return 0.0

    p10_val = arr.get("p10") or (p50_val * (1.0 - ARR_FALLBACK_SPREAD))
    p90_val = arr.get("p90") or (p50_val * (1.0 + ARR_FALLBACK_SPREAD))

    return _interpolate_expected(slow_pct, p10_val, p50_val, p90_val)


def calc_eta_per_seq(veh_id: str, current_seq: int,
                     cum_delay: float, micro_cum: float,
                     norm: dict, stop_pat: dict,
                     slot: str, stations: dict) -> list:
    total_delay = cum_delay + micro_cum

    raw_slow_pct = _slow_ema_pct_cache.get(veh_id, 50.0)
    slow_pct     = max(SLOW_PCT_MIN, min(SLOW_PCT_MAX, raw_slow_pct))

    recovery_mode = total_delay > RECOVERY_THRESHOLD

    remaining = sorted(seq for seq in stations if seq > current_seq)

    seg_times = []
    for seq in remaining:
        norm_seq_data = get_norm_seq(norm, slot, seq)
        base_arrival  = 0.0
        if norm_seq_data:
            arr = norm_seq_data.get("baseline", {}).get("arrival", {})
            if recovery_mode:
                base_arrival = arr.get("p10") or arr.get("p50") or 0.0
            else:
                base_arrival = _interp_arrival(slow_pct, arr)

        seg        = get_stop_seq(stop_pat, slot, seq)
        stop_score = 0.0
        if seg:
            stop_score = sum(
                p["probability"] * p["avgDuration"]
                for p in seg["stopPatterns"]
            )

        seg_times.append({
            "seq":          seq,
            "name":         stations[seq]["name"],
            "base_arrival": base_arrival,
            "stop_score":   stop_score,
        })

    max_score = max((r["stop_score"] for r in seg_times), default=0.0)
    eps       = 1e-6

    if max_score < eps:
        for r in seg_times:
            r["weight"] = 1.0
    else:
        for r in seg_times:
            r["weight"] = 1.0 - r["stop_score"] / (max_score + eps)

    weight_sum = sum(r["weight"] for r in seg_times) or 1.0

    results    = []
    cumulative = 0.0

    for r in seg_times:
        delay_share = total_delay * (r["weight"] / weight_sum)
        cumulative += r["base_arrival"] + delay_share
        use_pct_label = "p10(recovery)" if recovery_mode else f"slow_pct={slow_pct:.1f}"
        results.append({
            "seq":         r["seq"],
            "name":        r["name"],
            "eta_sec":     round(cumulative, 1),
            "delay_share": round(delay_share, 1),
            "stop_score":  round(r["stop_score"], 2),
            "weight":      round(r["weight"], 4),
            "percentile":  use_pct_label,
        })

    return results


# ── [수정 34-2] 지연 해소 권장 속도 계산 ─────

def calc_target_speed(eta_list: list, total_delay: float,
                      stations: dict, current_seq: int,
                      norm: dict, stop_pat: dict, slot: str) -> float | None:
    if total_delay <= TARGET_SPEED_MIN_DELAY:
        return None

    remaining_seqs = sorted(seq for seq in stations if seq > current_seq)
    if not remaining_seqs:
        return None

    stop_score_map: dict[int, float] = {
        r["seq"]: r.get("stop_score", 0.0) for r in eta_list
    }

    T_remain = 0.0
    D_remain = 0.0

    for seq in remaining_seqs:
        norm_seq_data = get_norm_seq(norm, slot, seq)
        if norm_seq_data is None:
            continue

        arr = norm_seq_data.get("baseline", {}).get("arrival", {})
        base_arr = arr.get("p50") or 0.0
        T_remain += base_arr
        T_remain += stop_score_map.get(seq, 0.0)

        dist_keys = norm_seq_data.get("baseline", {}).get("dist_to_elapsed", {})
        if dist_keys:
            max_dist = max(float(k) for k in dist_keys.keys())
            D_remain += max_dist

    if D_remain <= 0 or T_remain <= 0:
        return None

    net_time = T_remain - total_delay
    if net_time <= 0:
        return None

    target_kmh = D_remain / net_time * 3600

    if target_kmh < TARGET_SPEED_MIN_KMH:
        return None

    target_kmh = min(target_kmh, TARGET_SPEED_MAX_KMH)

    return round(target_kmh, 1)


# ── 디버그 플래그 ─────────────────────────────
_DEBUG = False


# ── 메인 추정 함수 ────────────────────────────
def estimate(item: dict, norm: dict, stop_pat: dict,
             stations: dict, turn_seq: int) -> dict | None:

    veh_id    = item.get("vehId", "unknown")
    sect_dist = float(item["sectDist"])

    if item.get("isrunyn") == "0":
        return None

    tracker = _stuck_tracker.get(veh_id)
    if tracker is None:
        _stuck_tracker[veh_id] = {"last_dist": sect_dist, "count": 1}
    elif tracker["last_dist"] != sect_dist:
        tracker["last_dist"] = sect_dist
        tracker["count"] = 1
    else:
        tracker["count"] += 1

    if _stuck_tracker[veh_id]["count"] >= STUCK_THRESHOLD:
        return None

    data_tm   = parse_time(item["dataTm"])
    stop_flag = item.get("stopFlag", "0")
    seq       = int(item["sectOrd"])
    direction = "to_onsu" if seq <= turn_seq else "to_dobong"

    slot = get_slot(data_tm)

    check_and_handle_turn(veh_id, seq, turn_seq, data_tm)

    update_dwell_log(veh_id, seq, stop_flag, data_tm)

    norm_seq = get_norm_seq(norm, slot, seq)
    if norm_seq is None:
        return None

    seg = get_stop_seq(stop_pat, slot, seq)

    combo_key, p_stop, in_zone = infer_combo(seg, sect_dist)
    if stop_flag == "1":
        p_stop = 1.0

    starttime = get_or_estimate_starttime(
        veh_id, seq, sect_dist, data_tm, norm_seq, combo_key
    )

    prev_expected = None
    if starttime is not None:
        prev_expected = get_elapsed(norm_seq, combo_key, sect_dist, "p50")

    current_kmh, norm_kmh = calc_speeds(
        veh_id, sect_dist, data_tm, prev_expected,
    )

    is_stopped = detect_stop(veh_id, stop_flag, in_zone, current_kmh)

    if is_stopped and stop_flag != "1":
        current_kmh = 0.0

    raw_pct = 50.0

    if starttime is not None:
        actual_elapsed = (data_tm - starttime).total_seconds()

        if is_stopped:
            p50_elapsed = get_elapsed(norm_seq, combo_key, sect_dist, "p50")
            if p50_elapsed is not None and p50_elapsed > 0:
                actual_elapsed = min(actual_elapsed, p50_elapsed * STOPPED_ELAPSED_CAP_FACTOR)

        seg_delay, expected_elapsed, raw_pct = calc_seg_delay(
            veh_id, sect_dist, actual_elapsed, norm_seq, combo_key,
            is_stopped=is_stopped,
            current_kmh=current_kmh,
            norm_kmh=norm_kmh,
        )
        if expected_elapsed is not None and expected_elapsed > 0 and sect_dist > 0:
            norm_kmh = sect_dist / expected_elapsed * 3600
    else:
        actual_elapsed = 0.0
        seg_delay, expected_elapsed = 0.0, None

    if stop_flag == "1":
        seg_delay = 0.0
    elif is_stopped and current_kmh is not None and current_kmh < STOPPED_KMH_THRESHOLD:
        seg_delay *= STOPPED_DELAY_FACTOR

    last      = read_last_log(veh_id)
    first_seq = _first_seq_cache.get(veh_id)

    is_warmup = (
        last is None
        or (first_seq is not None and last["seq"] == first_seq and seq == first_seq)
    )

    if is_warmup:
        _micro_seg_cache[veh_id]      = 0.0
        _micro_prev_seq_cache[veh_id] = seq

        if last is None:
            _first_seq_cache[veh_id] = seq
            append_log(veh_id, {
                "ts": data_tm.strftime("%Y%m%d%H%M%S"),
                "seq": seq, "sect_dist": sect_dist,
                "seg_delay": round(seg_delay, 2),
                "cum_delay": 0.0,
                "micro_seg": 0.0, "micro_cum": 0.0, "total_delay": 0.0,
                "raw_pct": round(raw_pct, 1),
                "stop_flag": stop_flag,
                "confirmed": False, "warmup": True,
            })
        else:
            append_log(veh_id, {
                "ts": data_tm.strftime("%Y%m%d%H%M%S"),
                "seq": seq, "sect_dist": sect_dist,
                "seg_delay": round(seg_delay, 2),
                "cum_delay": 0.0,
                "micro_seg": 0.0, "micro_cum": 0.0, "total_delay": 0.0,
                "raw_pct": round(raw_pct, 1),
                "stop_flag": stop_flag,
                "confirmed": False, "warmup": True,
            })
        return None

    if starttime is None:
        return None

    _restore_micro_cum(veh_id)

    prev_micro_seq = _micro_prev_seq_cache.get(veh_id)
    if prev_micro_seq is not None and prev_micro_seq != seq:
        _flush_micro_seg(veh_id, prev_micro_seq)
    _micro_prev_seq_cache[veh_id] = seq

    if actual_elapsed > 0:
        micro_seg = calc_micro_seg(
            veh_id, seq, actual_elapsed, norm_seq, combo_key,
            sect_dist, stop_flag, is_stopped, raw_pct
        )
    else:
        micro_seg = 0.0

    micro_cum = _micro_cum_cache.get(veh_id, 0.0)

    cum_delay = update_delay_log(
        veh_id, seq, sect_dist, seg_delay, data_tm,
        stop_flag=stop_flag,
        micro_seg=micro_seg,
        micro_cum=micro_cum,
        raw_pct=raw_pct,
    )

    if cum_delay is None:
        return None

    total_delay = cum_delay + micro_cum

    eta_list = calc_eta_per_seq(
        veh_id, seq, cum_delay, micro_cum,
        norm, stop_pat, slot, stations
    )

    target_kmh: float | None = None
    if not is_stopped:
        target_kmh = calc_target_speed(
            eta_list, total_delay, stations, seq,
            norm, stop_pat, slot
        )

    raw_slow         = _slow_ema_pct_cache.get(veh_id, 50.0)
    slow_pct_clamped = max(SLOW_PCT_MIN, min(SLOW_PCT_MAX, raw_slow))

    anomaly_seqs   = _anomaly_seq_cache.get(veh_id, set())
    is_anomaly_seq = seq in anomaly_seqs

    return {
        "vehId":            veh_id,
        "starttime_est":    starttime.strftime("%Y%m%d%H%M%S"),
        "slot":             slot,
        "current_seq":      seq,
        "current_name":     stations.get(seq, {}).get("name", ""),
        "direction":        direction,
        "leg":              _leg_cache.get(veh_id, "unknown"),
        "sect_dist":        sect_dist,
        "combo_key":        combo_key,

        "seg_delay":        round(seg_delay, 1),
        "cum_delay":        round(cum_delay, 1),

        "micro_seg":        round(micro_seg, 1),
        "micro_cum":        round(micro_cum, 1),
        "is_anomaly_seq":   is_anomaly_seq,

        "total_delay":      round(total_delay, 1),

        "raw_pct":          round(raw_pct, 1),
        "slow_ema_pct":     round(raw_slow, 1),
        "slow_pct_eff":     round(slow_pct_clamped, 1),
        "fast_ema_pct":     round(_ema_pct_cache.get(veh_id, 50.0), 1),

        "p_stop":           round(p_stop, 3),
        "in_zone":          in_zone,
        "is_stopped":       is_stopped,
        "stop_flag_api":    stop_flag == "1",
        "recovery_mode":    total_delay > RECOVERY_THRESHOLD,
        "current_kmh":      round(current_kmh, 1) if current_kmh is not None else None,
        "norm_kmh":         round(norm_kmh, 1) if norm_kmh is not None else None,

        "target_kmh":       target_kmh,

        "eta_per_seq":      eta_list,
    }


# ── API 호출 ──────────────────────────────────
def fetch_bus_positions() -> list | None:
    params = {
        "busRouteId": BUS_ROUTE_ID,
        "serviceKey": SERVICE_KEY,
        "resultType": "json",
    }
    try:
        resp = requests.get(API_URL, params=params, timeout=20)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        print(f"[{datetime.now():%H:%M:%S}] [ERROR] 타임아웃 — 재시도 대기")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[{datetime.now():%H:%M:%S}] [ERROR] 요청 실패: {e}")
        return None

    try:
        body = resp.json()
    except Exception:
        print(f"[{datetime.now():%H:%M:%S}] [ERROR] JSON 파싱 실패")
        return None

    header_cd  = str(body.get("msgHeader", {}).get("headerCd", "0"))
    header_msg = body.get("msgHeader", {}).get("headerMsg", "")

    if header_cd != "0":
        print(f"[{datetime.now():%H:%M:%S}] [API ERROR] code={header_cd} msg={header_msg}")
        if header_cd == "7":
            raise _QuotaExceededError()
        return None

    items = body.get("msgBody", {}).get("itemList", [])

    if isinstance(items, list) and len(items) == 0:
        print(f"[{datetime.now():%H:%M:%S}] [WARN] itemList 비어있음 (API 정상 응답이나 데이터 없음)")

    return items


class _QuotaExceededError(Exception):
    pass


# ── 신규 버스 감지 출력 ───────────────────────
def print_new_vehicle(veh_id: str, seq: int, sect_dist: float,
                      turn_seq: int) -> None:
    leg = "outbound(도봉→온수)" if seq <= turn_seq else "inbound(온수→도봉)"
    print(f"\n{'='*60}")
    print(f"[신규 버스 감지] vehId={veh_id}  현재 seq={seq}  sectDist={sect_dist:.3f}km  leg={leg}")

    if not DELAY_LOG_PATH.exists():
        print("  → 통합 딜레이 로그 없음 (이번 세션 첫 등장)")
        print(f"{'='*60}\n")
        return

    records = []
    try:
        with open(DELAY_LOG_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("vehId") == veh_id:
                        records.append(rec)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"  → 로그 읽기 실패: {e}")
        print(f"{'='*60}\n")
        return

    if not records:
        print("  → 이번 세션 첫 등장 (로그 없음)")
        print(f"{'='*60}\n")
        return

    confirmed    = [r for r in records if r.get("confirmed")]
    ongoing      = [r for r in records if not r.get("confirmed")]
    last_ongoing = ongoing[-1] if ongoing else None

    summary_records = confirmed.copy()
    if last_ongoing:
        summary_records.append(last_ongoing)

    seen_seq: dict = {}
    for r in summary_records:
        seen_seq[r["seq"]] = r
    deduped = sorted(seen_seq.values(), key=lambda r: r["seq"])

    dist_arr = [r["sect_dist"] for r in deduped]
    seq_arr  = [r["seq"]       for r in deduped]
    print(f"  이동 이력 ({len(deduped)}개 구간):")
    print(f"    seq       : {seq_arr}")
    print(f"    sectDist  : {[round(d, 3) for d in dist_arr]}")

    last_rec   = records[-1]
    last_cum   = last_rec.get("cum_delay", 0.0)
    last_micro = last_rec.get("micro_cum", 0.0)
    last_total = last_rec.get("total_delay", last_cum)
    status     = "🔴 지연" if last_total > RECOVERY_THRESHOLD else ("🟡 소폭지연" if last_total > 30 else "🟢 정상")
    print(f"  이상치 지연 : {last_cum:+.1f}초")
    print(f"  미세 지연   : {last_micro:+.1f}초")
    print(f"  합산 지연   : {last_total:+.1f}초  {status}")
    print(f"  총 로그 수  : {len(records)}줄  (confirmed={len(confirmed)}, ongoing={len(ongoing)})")
    print(f"{'='*60}\n")


# ── [수정 18] 출력 게이팅 함수들 ─────────────
def gated_print_estimate(r: dict) -> None:
    veh_id   = r["vehId"]
    stop_api = r.get("stop_flag_api", False)

    if not _output_enabled.get(veh_id, False):
        return

    was_silent = _in_stop_silence.get(veh_id, False)

    if stop_api:
        if not was_silent:
            _in_stop_silence[veh_id] = True
            _print_stop_line(r)
        return

    if was_silent:
        _in_stop_silence[veh_id] = False

    _print_normal_line(r)


def _print_stop_line(r: dict) -> None:
    total   = r["total_delay"]
    display = 0.0 if abs(total) < 0.05 else total
    print(
        f"[{datetime.now():%H:%M:%S}] {r['vehId']:<14} "
        f"seq={r['current_seq']:>3} ({r['current_name'][:8]:<8}) "
        f"dist={r['sect_dist']:.3f}km  "
        f"seg={r['seg_delay']:+.1f}s  "
        f"cum={r['cum_delay']:+.1f}s  micro={r['micro_cum']:+.1f}s  total={display:+.1f}s"
        f"  🛑 정지(API정지)"
    )


def _print_normal_line(r: dict) -> None:
    total      = r["total_delay"]
    cum        = r["cum_delay"]
    micro      = r["micro_cum"]
    mode       = " [만회모드]" if r["recovery_mode"] else ""
    total_disp = 0.0 if abs(total) < 0.05 else total

    slow_pct = r.get("slow_pct_eff", 50.0)
    slow_tag = ""
    if slow_pct >= 62:
        slow_tag = f"  📈slow={slow_pct:.0f}"
    elif slow_pct <= 38:
        slow_tag = f"  📉slow={slow_pct:.0f}"

    raw_pct     = r.get("raw_pct", 50.0)
    anomaly_tag = ""
    if r.get("is_anomaly_seq"):
        if raw_pct > MICRO_ANOMALY_PCT_HI:
            anomaly_tag = f"  ⚠️이상치(pct={raw_pct:.0f}>{MICRO_ANOMALY_PCT_HI:.0f})"
        elif raw_pct < MICRO_ANOMALY_PCT_LO:
            anomaly_tag = f"  ⚠️이상치(pct={raw_pct:.0f}<{MICRO_ANOMALY_PCT_LO:.0f})"
        else:
            anomaly_tag = f"  ⚠️이상치(seq등록)"

    micro_tag = ""
    if abs(micro) >= 2.0:
        micro_tag = f"  🔹micro={micro:+.1f}s"

    target_tag = ""
    target_kmh = r.get("target_kmh")
    if target_kmh is not None and not r["is_stopped"]:
        norm_kmh = r.get("norm_kmh")
        if norm_kmh is not None and norm_kmh > 0:
            ratio = target_kmh / norm_kmh
            if ratio >= 1.15:
                target_tag = f"  🎯목표:{target_kmh:.1f}km/h↑"
            elif ratio <= 0.85:
                target_tag = f"  🎯목표:{target_kmh:.1f}km/h↓"
            else:
                target_tag = f"  🎯목표:{target_kmh:.1f}km/h"
        else:
            target_tag = f"  🎯목표:{target_kmh:.1f}km/h"

    base = (
        f"[{datetime.now():%H:%M:%S}] {r['vehId']:<14} "
        f"seq={r['current_seq']:>3} ({r['current_name'][:8]:<8}) "
        f"dist={r['sect_dist']:.3f}km  "
        f"seg={r['seg_delay']:+.1f}s  cum={cum:+.1f}s  "
        f"total={total_disp:+.1f}s{mode}{slow_tag}{anomaly_tag}{micro_tag}{target_tag}"
    )

    if r["is_stopped"]:
        stop_reason = "패턴+저속(0km/h)" if r.get("in_zone") else "저속정지"
        print(f"{base}  🛑 정지({stop_reason})")
        return

    if total > 0:
        cur  = r.get("current_kmh")
        norm = r.get("norm_kmh")
        spd  = ""
        if cur is not None and norm is not None:
            spd = f"  🚌 현재:{cur:.1f}km/h 기준:{norm:.1f}km/h"
        elif cur is not None:
            spd = f"  🚌 현재:{cur:.1f}km/h"
        print(f"{base}{spd}")
    else:
        print(base)


# ── 메인 루프 ─────────────────────────────────
def main():
    norm, stop_pat, stations, turn_seq = load_data()

    last_processed_key: dict[str, str] = {}
    announced: set[str] = set()

    print(f"[{datetime.now():%H:%M:%S}] 추정 루프 시작 (Ctrl+C로 종료)")
    print(f"[{datetime.now():%H:%M:%S}] 통합 로그: {DELAY_LOG_PATH}")
    print(f"[{datetime.now():%H:%M:%S}] 정차 로그: {DWELL_LOG_PATH}")
    print(f"[{datetime.now():%H:%M:%S}] 이상치 판정: raw_pct < {MICRO_ANOMALY_PCT_LO:.0f} 또는 > {MICRO_ANOMALY_PCT_HI:.0f}  "
          f"(안전지대 p{MICRO_ANOMALY_PCT_LO:.0f}~p{MICRO_ANOMALY_PCT_HI:.0f})  → 즉시 on/off")
    print(f"[{datetime.now():%H:%M:%S}] micro 상한: ±{MICRO_DELAY_CAP:.0f}초")
    print(f"[{datetime.now():%H:%M:%S}] [수정34] 패턴정지→0km/h, 권장속도 최소지연={TARGET_SPEED_MIN_DELAY:.0f}s 상한={TARGET_SPEED_MAX_KMH:.0f}km/h")

    import time

    while True:
        now = datetime.now()
        if now.hour >= 24:
            print("수집 종료 시각 도달")
            break

        try:
            items = fetch_bus_positions()
        except _QuotaExceededError:
            print(f"[{now:%H:%M:%S}] [CRITICAL] 호출횟수 초과 — 1시간 대기")
            last_processed_key.clear()
            time.sleep(3600)
            continue

        if items is None:
            time.sleep(10)
            continue

        if len(items) == 0:
            time.sleep(4)
            continue

        for item in items:
            veh_id    = item.get("vehId", "unknown")
            data_tm_s = item.get("dataTm", "")
            seq       = int(item.get("sectOrd") or 0)
            sect_dist = float(item.get("sectDist") or 0)

            cur_key = f"{data_tm_s}_{seq}_{sect_dist}"
            if last_processed_key.get(veh_id) == cur_key:
                continue
            last_processed_key[veh_id] = cur_key

            if veh_id not in announced:
                print_new_vehicle(veh_id, seq, sect_dist, turn_seq)
                announced.add(veh_id)

            r = estimate(item, norm, stop_pat, stations, turn_seq)
            if r:
                gated_print_estimate(r)

        time.sleep(4)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[종료] 수집 중단")