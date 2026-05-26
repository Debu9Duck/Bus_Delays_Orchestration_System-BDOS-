import json
import math
import statistics
import heapq
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import halfnorm


# --------------------------------------------------
# Constants
# --------------------------------------------------

MIN_SAMPLES = 3

LOWER_MIN  = 5
LOWER_MAX  = 60
SAFE_MIN   = 45
SAFE_MAX   = 65

SCORE_ALPHA = 0.5
SCORE_BETA  = 0.5

TOP_K_SCAN  = 200


# --------------------------------------------------
# 1. Load records
# --------------------------------------------------

def load_records(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# --------------------------------------------------
# 2. Elapsed time
# --------------------------------------------------

def parse_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d%H%M%S")


def get_elapsed_sec(record: dict) -> float | None:
    dist_list  = record["10secDist"]
    times      = record["10secTm"]
    total_dist = dist_list[-1] - dist_list[0]
    if total_dist <= 0:
        return None
    total_sec = (parse_dt(times[-1]) - parse_dt(times[0])).total_seconds()
    if total_sec <= 0:
        return None
    return total_sec


# --------------------------------------------------
# 3. log_ratio
# --------------------------------------------------

def compute_log_ratios(records: list[dict]) -> list[dict]:
    seq_groups: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        elapsed = get_elapsed_sec(rec)
        if elapsed is None:
            continue
        seq_groups[rec["seq"]].append({
            "vehId"      : rec["vehId"],
            "seq"        : rec["seq"],
            "toSect"     : rec["toSect"],
            "elapsed_sec": elapsed,
        })

    results = []
    for seq, group in seq_groups.items():
        if len(group) < MIN_SAMPLES:
            continue
        median_sec = statistics.median([g["elapsed_sec"] for g in group])
        if median_sec <= 0:
            continue
        for g in group:
            r = g["elapsed_sec"] / median_sec
            results.append({
                **g,
                "median_sec": round(median_sec, 2),
                "ratio"     : round(r, 4),
                "log_ratio" : round(math.log(r), 6),
                "n_in_seq"  : len(group),
            })
    return results


# --------------------------------------------------
# 4. Percentile utils
# --------------------------------------------------

def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s   = sorted(data)
    n   = len(s)
    pos = p / 100 * (n - 1)
    lo, hi = int(pos), min(int(pos) + 1, n - 1)
    return s[lo] + (pos - lo) * (s[hi] - s[lo])


def value_to_percentile(sorted_data: list[float], value: float) -> float:
    n = len(sorted_data)
    if n == 0: return 0.0
    if value <= sorted_data[0]: return 0.0
    if value >= sorted_data[-1]: return 100.0
    for i in range(n - 1):
        if sorted_data[i] <= value <= sorted_data[i + 1]:
            span = sorted_data[i + 1] - sorted_data[i]
            frac = (value - sorted_data[i]) / span if span > 0 else 0
            return round((i + frac) / (n - 1) * 100, 1)
    return 100.0


# --------------------------------------------------
# 4-a. Q-Q common util - residual/slope time series
# --------------------------------------------------

def _qq_series(
    log_ratios : list[float],
    mu         : float,
    sigma      : float,
    smooth_window: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    from statistics import NormalDist

    sorted_data = np.sort(np.array(log_ratios))
    n           = len(sorted_data)
    nd          = NormalDist(mu=mu, sigma=sigma)

    theor_q = np.array([
        nd.inv_cdf((i - 0.375) / (n + 0.25)) for i in range(1, n + 1)
    ])
    actual_q = sorted_data
    residual = actual_q - theor_q

    d_theor  = np.diff(theor_q)
    d_actual = np.diff(actual_q)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(d_theor != 0, d_actual / d_theor, np.nan)

    if smooth_window is not None and smooth_window >= 5:
        w = min(smooth_window, len(slope) - (1 if len(slope) % 2 == 0 else 0))
        if w >= 5:
            slope = savgol_filter(slope, window_length=w | 1, polyorder=2)

    return theor_q, actual_q, residual, slope


# --------------------------------------------------
# [v2] Half-Normal threshold helper
# --------------------------------------------------

def _halfnorm_threshold(
    abs_curv   : np.ndarray,
    theor_mid  : np.ndarray,
    center_pct : tuple[float, float] = (20.0, 80.0),
    tail_alpha : float = 0.95,
) -> dict:

    lo_pct, hi_pct = center_pct
    center_mask = (
        (theor_mid >= np.percentile(theor_mid, lo_pct)) &
        (theor_mid <= np.percentile(theor_mid, hi_pct))
    )
    center_curv = abs_curv[center_mask]
    n_center    = int(center_mask.sum())

    fallback_used = False

    if n_center >= 5:
        sigma_hn = float(np.sqrt(np.mean(center_curv ** 2)))
    else:
        sigma_hn      = float(np.sqrt(np.mean(abs_curv ** 2)))
        fallback_used = True

    if sigma_hn == 0.0:
        sigma_hn      = float(np.std(abs_curv)) or 1e-10
        fallback_used = True

    threshold = float(halfnorm.ppf(tail_alpha, scale=sigma_hn))

    return {
        "threshold"    : threshold,
        "sigma_hn"     : sigma_hn,
        "n_center"     : n_center,
        "center_pct"   : center_pct,
        "tail_alpha"   : tail_alpha,
        "fallback_used": fallback_used,
    }


# --------------------------------------------------
# 4-b. S-curve curvature-based safe zone detection  [MAIN]
# --------------------------------------------------

def safe_zone_from_qq_curvature(
    log_ratios           : list[float],
    smooth_frac          : float = 0.05,
    # [v2] Half-Normal threshold parameters
    curv_tail_alpha      : float = 0.90,
    curv_center_lo_pct   : float = 10.0,
    curv_center_hi_pct   : float = 90.0,
    curvature_mad_factor : float = 2.0,   # deprecated, kept for API compat
    min_search_pct       : float = 5.0,
    max_search_pct       : float = 95.0,
    # [v3] V-shape valley detection parameter
    vshape_sigma_factor  : float = 1.0,
    # [v4] Central spike detection: valley search starts from this percentile
    spike_center_pct     : float = 49.0,
) -> dict:

    n = len(log_ratios)
    if n < 20:
        raise ValueError(f"Curvature detection requires at least 20 samples (current n={n})")

    mu    = float(np.mean(log_ratios))
    sigma = float(np.std(log_ratios, ddof=1))

    theor_q, actual_q, residual, _ = _qq_series(log_ratios, mu, sigma)

    sw = max(5, int(n * smooth_frac) | 1)
    residual_smooth = savgol_filter(residual, window_length=sw, polyorder=3)

    curvature = np.diff(residual_smooth, 2)
    theor_mid = theor_q[1:-1]
    abs_curv  = np.abs(curvature)

    # ------------------------------------------------------------------
    # [v2] Half-Normal threshold — sigma_hn을 V자 이탈 판정에도 재사용
    # ------------------------------------------------------------------
    hn_result = _halfnorm_threshold(
        abs_curv,
        theor_mid,
        center_pct = (curv_center_lo_pct, curv_center_hi_pct),
        tail_alpha = curv_tail_alpha,
    )
    threshold = hn_result["threshold"]   # HalfNorm threshold (참고용)
    sigma_hn  = hn_result["sigma_hn"]

    # 하위 호환용 legacy MAD
    legacy_mad_curv  = float(np.median(abs_curv))
    legacy_threshold = legacy_mad_curv * curvature_mad_factor

    # ------------------------------------------------------------------
    # [v4] Central spike: valley search 구간 설정
    # spike_center_pct 이상의 theor_q 구간에서만 valley 탐색
    # → 중앙 spike의 바닥(right side of spike center)을 반드시 포함
    # ------------------------------------------------------------------
    sorted_log = sorted(log_ratios)
    lo_bound   = float(np.percentile(theor_q, min_search_pct))
    hi_bound   = float(np.percentile(theor_q, max_search_pct))

    # [v4] spike center threshold: theor_q의 spike_center_pct 분위수 값
    spike_center_theor = float(np.percentile(theor_q, spike_center_pct))

    # valley 탐색 구간: [spike_center_theor, hi_bound] ∩ search range
    # → spike의 우측(바닥) 방향에서 최솟값을 탐색
    valley_lo = max(lo_bound, spike_center_theor)
    valley_hi = hi_bound

    valley_search_mask = (
        (theor_mid >= valley_lo) & (theor_mid <= valley_hi)
    )

    # V자 탐색용 추가 smoothing (노이즈 제거)
    sw2 = max(5, int(len(abs_curv) * 0.05) | 1)
    abs_curv_s = savgol_filter(abs_curv, window_length=sw2, polyorder=2) \
                 if len(abs_curv) > sw2 else abs_curv.copy()

    # ------------------------------------------------------------------
    # [v5] valley = spike_center_pct 이후 구간의 첫 번째 극솟값(local minimum)
    #
    # smoothed curvature의 1차 차분 부호가 - → + 로 바뀌는 첫 번째 지점
    # = 기울기가 0이 되는(감소 후 증가로 전환) 극솟값
    # 전역 최솟값 탐색 대신 사용하여 spike 바닥을 정확히 포착
    # ------------------------------------------------------------------
    valley_search_idx = np.where(valley_search_mask)[0]

    if len(valley_search_idx) < 3:
        # fallback: 전체 search 구간 전역 최솟값
        full_search_mask = (theor_mid >= lo_bound) & (theor_mid <= hi_bound)
        full_search_idx  = np.where(full_search_mask)[0]
        valley_pos = full_search_idx[int(np.argmin(abs_curv_s[full_search_idx]))] \
                     if len(full_search_idx) else len(abs_curv) // 2
        print(f"  [v5 Warning] valley_search too short → fallback to global min")
    else:
        # spike_center_pct 이후 구간에서 첫 번째 극솟값 탐색
        # diff 부호: -1(하강), +1(상승). -1→+1 전환점 = 극솟값
        seg        = abs_curv_s[valley_search_idx]
        seg_diff   = np.diff(seg)
        # 부호 전환: i번째 diff가 음수(하강)이고 i+1번째 diff가 양수(상승)
        local_min_rel = np.where((seg_diff[:-1] < 0) & (seg_diff[1:] >= 0))[0] + 1

        if len(local_min_rel) == 0:
            # 극솟값 없으면 구간 내 전역 최솟값 사용
            valley_pos = valley_search_idx[int(np.argmin(seg))]
            print(f"  [v5 Info] no local min found → using segment global min")
        else:
            # 첫 번째 극솟값 사용
            valley_pos = valley_search_idx[local_min_rel[0]]

    valley_val   = float(abs_curv_s[valley_pos])
    valley_theor = float(theor_mid[valley_pos])

    # V자 이탈 threshold: valley 값 + sigma_hn 기반 여유
    vshape_thr = valley_val + sigma_hn * vshape_sigma_factor

    # ------------------------------------------------------------------
    # [v4] 경계 탐색: 좌/우 독립 탐색
    # 전체 search 구간 (lo_bound ~ hi_bound) 에서
    # valley를 기준으로 좌/우로 vshape_thr 초과 지점을 찾음
    # ------------------------------------------------------------------
    full_search_mask = (theor_mid >= lo_bound) & (theor_mid <= hi_bound)

    # 왼쪽 경계: valley 왼쪽(전체 search 범위)에서 vshape_thr 초과하는 "마지막" 지점
    left_search  = full_search_mask & (theor_mid <= theor_mid[valley_pos])
    left_exceed  = np.where(left_search & (abs_curv_s > vshape_thr))[0]

    if len(left_exceed) == 0:
        left_candidates = np.where(left_search)[0]
        left_pos = int(left_candidates[0]) if len(left_candidates) else valley_pos
        left_fallback = True
    else:
        left_pos = int(left_exceed[-1])
        left_fallback = False

    # 오른쪽 경계: valley 오른쪽(전체 search 범위)에서 vshape_thr 초과하는 "첫" 지점
    right_search = full_search_mask & (theor_mid >= theor_mid[valley_pos])
    right_exceed = np.where(right_search & (abs_curv_s > vshape_thr))[0]

    if len(right_exceed) == 0:
        right_candidates = np.where(right_search)[0]
        right_pos = int(right_candidates[-1]) if len(right_candidates) else valley_pos
        right_fallback = True
    else:
        right_pos = int(right_exceed[0])
        right_fallback = False

    # left_pos == right_pos 방어 (V자가 너무 좁은 경우)
    if left_pos >= right_pos:
        left_pos  = max(0, valley_pos - 1)
        right_pos = min(len(abs_curv) - 1, valley_pos + 1)

    # ------------------------------------------------------------------
    # 경계 → percentile 변환
    # ------------------------------------------------------------------
    left_log  = float(actual_q[min(left_pos + 1, len(actual_q) - 1)])
    right_log = float(actual_q[min(right_pos + 1, len(actual_q) - 1)])

    lower_pct    = round(value_to_percentile(sorted_log, left_log))
    safe_end_pct = round(value_to_percentile(sorted_log, right_log))

    lower_pct    = max(int(min_search_pct), min(lower_pct, 49))
    safe_end_pct = max(lower_pct + 1, min(safe_end_pct, int(max_search_pct)))

    lower_log_v    = float(np.percentile(sorted_log, lower_pct))
    safe_end_log_v = float(np.percentile(sorted_log, safe_end_pct))

    # mid_mask (참고용)
    mid_mask = (theor_mid >= np.percentile(theor_q, 40)) & \
               (theor_mid <= np.percentile(theor_q, 60))
    mid_curv_median = float(np.median(abs_curv[mid_mask])) if mid_mask.sum() else legacy_mad_curv

    return {
        "lower_pct"              : lower_pct,
        "safe_end_pct"           : safe_end_pct,
        "safe_width"             : safe_end_pct - lower_pct,
        "lower_log"              : round(lower_log_v,    4),
        "safe_end_log"           : round(safe_end_log_v, 4),
        "lower_ratio"            : round(math.exp(lower_log_v),    4),
        "safe_end_ratio"         : round(math.exp(safe_end_log_v), 4),
        "left_curv_theor"        : round(float(theor_mid[left_pos]),  4),
        "right_curv_theor"       : round(float(theor_mid[right_pos]), 4),
        "left_curv_log"          : round(left_log,  4),
        "right_curv_log"         : round(right_log, 4),
        # [v2] Half-Normal threshold fields
        "sigma_hn"               : round(sigma_hn,   8),
        "curv_tail_alpha"        : curv_tail_alpha,
        "curv_threshold"         : round(threshold,  8),
        "curv_center_pct"        : (curv_center_lo_pct, curv_center_hi_pct),
        "hn_n_center"            : hn_result["n_center"],
        "hn_fallback_used"       : hn_result["fallback_used"],
        # [v3] V-shape valley fields
        "valley_pos"             : int(valley_pos),
        "valley_theor"           : round(valley_theor, 4),
        "valley_val"             : round(valley_val,   10),
        "vshape_thr"             : round(vshape_thr,   10),
        "vshape_sigma_factor"    : vshape_sigma_factor,
        "left_fallback"          : left_fallback,
        "right_fallback"         : right_fallback,
        # [v4] spike center fields
        "spike_center_pct"       : spike_center_pct,
        "spike_center_theor"     : round(spike_center_theor, 4),
        "valley_search_lo"       : round(valley_lo, 4),
        "valley_search_hi"       : round(valley_hi, 4),
        # 하위 호환용
        "mad_curvature"          : round(legacy_mad_curv,  8),
        "legacy_threshold"       : round(legacy_threshold, 8),
        "curvature_mad_factor"   : curvature_mad_factor,
        "mid_curvature_median"   : round(mid_curv_median, 8),
        "smooth_window"          : sw,
        "min_search_pct"         : min_search_pct,
        "max_search_pct"         : max_search_pct,
        "method"                 : (
            f"QQ-curvature [v5 LocalMin+SpikeCenter] "
            f"(spike_center=p{int(spike_center_pct)}, factor={vshape_sigma_factor}, "
            f"alpha={curv_tail_alpha}, "
            f"center=p{int(curv_center_lo_pct)}~p{int(curv_center_hi_pct)}, "
            f"smooth={sw}, search=p{int(min_search_pct)}~p{int(max_search_pct)})"
        ),
        "_theor_q"        : theor_q,
        "_actual_q"       : actual_q,
        "_residual"       : residual,
        "_residual_smooth": residual_smooth,
        "_theor_mid"      : theor_mid,
        "_curvature"      : curvature,
        "_abs_curv"       : abs_curv,
        "_abs_curv_s"     : abs_curv_s,
        "_threshold"      : threshold,
    }


# --------------------------------------------------
# 4-c. Q-Q slope-jump method (auxiliary)
# --------------------------------------------------

def safe_zone_from_qq_inflection(
    log_ratios       : list[float],
    smooth_frac      : float = 0.05,
    min_search_pct   : float = 5.0,
    max_search_pct   : float = 95.0,
    slope_jump_factor: float = 1.8,
) -> dict:

    n = len(log_ratios)
    if n < 20:
        raise ValueError(f"Q-Q inflection detection requires at least 20 samples (current n={n})")

    mu    = float(np.mean(log_ratios))
    sigma = float(np.std(log_ratios, ddof=1))

    smooth_window = max(5, int(n * smooth_frac) | 1)
    theor_q, actual_q, _, slope = _qq_series(
        log_ratios, mu, sigma, smooth_window=smooth_window
    )

    sorted_log = sorted(log_ratios)

    lo_bound = float(np.percentile(theor_q, min_search_pct))
    hi_bound = float(np.percentile(theor_q, max_search_pct))

    mid_mask     = (theor_q[:-1] >= np.percentile(theor_q, 10)) & \
                   (theor_q[:-1] <= np.percentile(theor_q, 90))
    mid_slopes   = slope[mid_mask]
    median_slope = float(np.nanmedian(mid_slopes)) if len(mid_slopes) else 1.0
    threshold    = median_slope * slope_jump_factor

    right_mask   = (theor_q[:-1] > np.percentile(theor_q, 50)) & \
                   (theor_q[:-1] >= lo_bound) & (theor_q[:-1] <= hi_bound)
    right_idx    = np.where(right_mask & (slope > threshold))[0]
    right_pos    = int(right_idx[0]) if len(right_idx) else \
                   int(np.where(right_mask)[0][-1]) if right_mask.sum() else len(slope) - 1

    left_mask    = (theor_q[:-1] < np.percentile(theor_q, 50)) & \
                   (theor_q[:-1] >= lo_bound) & (theor_q[:-1] <= hi_bound)
    left_idx     = np.where(left_mask & (slope > threshold))[0]
    left_pos     = int(left_idx[-1]) if len(left_idx) else \
                   int(np.where(left_mask)[0][0]) if left_mask.sum() else 0

    right_actual = float(actual_q[right_pos])
    left_actual  = float(actual_q[left_pos])

    safe_end_pct = round(value_to_percentile(sorted_log, right_actual))
    lower_pct    = round(value_to_percentile(sorted_log, left_actual))

    lower_pct    = max(int(min_search_pct), min(lower_pct, 49))
    safe_end_pct = max(lower_pct + 1, min(safe_end_pct, int(max_search_pct)))

    lower_log    = float(np.percentile(sorted_log, lower_pct))
    safe_end_log = float(np.percentile(sorted_log, safe_end_pct))

    return {
        "lower_pct"             : lower_pct,
        "safe_end_pct"          : safe_end_pct,
        "safe_width"            : safe_end_pct - lower_pct,
        "lower_log"             : round(lower_log,    4),
        "safe_end_log"          : round(safe_end_log, 4),
        "lower_ratio"           : round(math.exp(lower_log),    4),
        "safe_end_ratio"        : round(math.exp(safe_end_log), 4),
        "left_inflection_theor" : round(float(theor_q[left_pos]),  4),
        "right_inflection_theor": round(float(theor_q[right_pos]), 4),
        "left_inflection_log"   : round(left_actual,  4),
        "right_inflection_log"  : round(right_actual, 4),
        "median_slope"          : round(median_slope, 4),
        "slope_jump_factor"     : slope_jump_factor,
        "smooth_window"         : smooth_window,
        "min_search_pct"        : min_search_pct,
        "max_search_pct"        : max_search_pct,
        "method"                : (
            f"QQ-inflection (factor={slope_jump_factor}, smooth={smooth_window}, "
            f"search=p{int(min_search_pct)}~p{int(max_search_pct)})"
        ),
        "_theor_q"  : theor_q,
        "_actual_q" : actual_q,
        "_slope"    : slope,
    }


# --------------------------------------------------
# 5. Prefix-sum tables
# --------------------------------------------------

def _build_prefix_tables(arr_sorted: np.ndarray, pct_vals: np.ndarray):
    n_pct      = len(pct_vals)
    prefix_cnt = np.zeros(n_pct, dtype=np.int64)
    prefix_sum = np.zeros(n_pct, dtype=np.float64)
    prefix_sq  = np.zeros(n_pct, dtype=np.float64)
    for p in range(n_pct):
        idx           = int(np.searchsorted(arr_sorted, pct_vals[p], side="right"))
        prefix_cnt[p] = idx
        prefix_sum[p] = arr_sorted[:idx].sum()
        prefix_sq[p]  = (arr_sorted[:idx] ** 2).sum()
    return prefix_cnt, prefix_sum, prefix_sq


def _variance_from_prefix(
    prefix_cnt: np.ndarray,
    prefix_sum: np.ndarray,
    prefix_sq:  np.ndarray,
    lo_pct: int,
    hi_pct: int,
) -> tuple[int, float]:
    cnt_hi = int(prefix_cnt[hi_pct])
    sum_hi = prefix_sum[hi_pct]
    sq_hi  = prefix_sq[hi_pct]
    if lo_pct == 0:
        cnt, s, sq = cnt_hi, sum_hi, sq_hi
    else:
        cnt_lo = int(prefix_cnt[lo_pct - 1])
        cnt = cnt_hi - cnt_lo
        s   = sum_hi - prefix_sum[lo_pct - 1]
        sq  = sq_hi  - prefix_sq[lo_pct - 1]
    if cnt < 2:
        return cnt, float("inf")
    mean = s / cnt
    var  = sq / cnt - mean ** 2
    var  = var * cnt / (cnt - 1)
    return cnt, max(var, 0.0)


# --------------------------------------------------
# 6. Recommend safe zone (grid search)
# --------------------------------------------------

def recommend_safe_zone(
    log_ratios : list[float],
    lower_min  : int   = LOWER_MIN,
    lower_max  : int   = LOWER_MAX,
    safe_min   : int   = SAFE_MIN,
    safe_max   : int   = SAFE_MAX,
    alpha      : float = SCORE_ALPHA,
    k_sigma    : float | None = None,
) -> dict:
    n = len(log_ratios)
    if n < 2:
        raise ValueError(f"Too few samples (n={n})")

    mu    = float(np.mean(log_ratios))
    sigma = float(np.std(log_ratios, ddof=1))
    skewness = (
        float(np.mean([(x - mu) ** 3 for x in log_ratios]) / sigma ** 3)
        if sigma > 0 and n >= 3 else 0.0
    )

    arr      = np.array(sorted(log_ratios), dtype=np.float64)
    pct_vals = np.percentile(arr, np.arange(101), method="linear")
    prefix_cnt, prefix_sum, prefix_sq = _build_prefix_tables(arr, pct_vals)

    if k_sigma is not None:
        sorted_log  = sorted(log_ratios)
        span_lo_pct = value_to_percentile(sorted_log, mu - k_sigma * sigma)
        span_hi_pct = value_to_percentile(sorted_log, mu + k_sigma * sigma)
        safe_min    = max(1, round(span_hi_pct - span_lo_pct))
        print(f"  k_sigma={k_sigma}  ->  "
              f"2*k*sigma covers ~{safe_min}%p  (safe_min set to {safe_min}%p)")

    raw_list: list[tuple] = []
    top_heap: list[tuple] = []

    for lower_pct in range(lower_min, lower_max + 1):
        for safe_width in range(safe_min, safe_max + 1):
            safe_end_pct = lower_pct + safe_width
            if safe_end_pct >= 100:
                break
            cnt, var = _variance_from_prefix(
                prefix_cnt, prefix_sum, prefix_sq,
                lower_pct, safe_end_pct,
            )
            if cnt < 2:
                continue
            raw_list.append((lower_pct, safe_width, var))

    if not raw_list:
        raise ValueError("No valid (lower_pct, safe_width) combination found.")

    max_var = max(r[2] for r in raw_list) or 1.0

    best_score     = float("inf")
    best_lower_pct = raw_list[0][0]
    best_safe_w    = raw_list[0][1]

    for lower_pct, safe_width, var in raw_list:
        score = var / max_var
        if score < best_score:
            best_score     = score
            best_lower_pct = lower_pct
            best_safe_w    = safe_width
        entry = (-score, lower_pct, safe_width, round(var, 6))
        if len(top_heap) < TOP_K_SCAN:
            heapq.heappush(top_heap, entry)
        elif entry > top_heap[0]:
            heapq.heapreplace(top_heap, entry)

    top_scan = sorted(
        [{"lower_pct": t[1], "safe_width": t[2],
          "safe_var": t[3], "score": round(-t[0], 6)}
         for t in top_heap],
        key=lambda x: x["score"],
    )

    best_safe_end_pct = best_lower_pct + best_safe_w
    lower_log         = float(pct_vals[best_lower_pct])
    safe_end_log      = float(pct_vals[best_safe_end_pct])

    return {
        "lower_pct"      : best_lower_pct,
        "safe_width"     : best_safe_w,
        "safe_end_pct"   : best_safe_end_pct,
        "lower_log"      : round(lower_log,    4),
        "safe_end_log"   : round(safe_end_log, 4),
        "lower_ratio"    : round(math.exp(lower_log),    4),
        "safe_end_ratio" : round(math.exp(safe_end_log), 4),
        "best_score"     : round(best_score, 6),
        "alpha"          : alpha,
        "mean"           : round(mu,       4),
        "std"            : round(sigma,    4),
        "skewness"       : round(skewness, 4),
        "n"              : n,
        "scan"           : top_scan,
        "safe_min_used"  : safe_min,
        "_raw"           : [
            {"lower_pct": lp, "safe_width": sw, "score": round(v / max_var, 6)}
            for lp, sw, v in raw_list
        ],
        "method"         : f"grid (safe_min={safe_min}%p"
                           + (f" from k={k_sigma}*sigma)" if k_sigma else ")"),
    }


# --------------------------------------------------
# 6-b. Sigma-based safe zone
# --------------------------------------------------

def safe_zone_from_sigma(
    log_ratios: list[float],
    k: float = 2.0,
) -> dict:
    n = len(log_ratios)
    if n < 2:
        raise ValueError(f"Too few samples (n={n})")

    mu    = float(np.mean(log_ratios))
    sigma = float(np.std(log_ratios, ddof=1))
    skewness = (
        float(np.mean([(x - mu) ** 3 for x in log_ratios]) / sigma ** 3)
        if sigma > 0 and n >= 3 else 0.0
    )

    lower_log    = mu - k * sigma
    safe_end_log = mu + k * sigma

    sorted_log   = sorted(log_ratios)
    lower_pct    = round(value_to_percentile(sorted_log, lower_log))
    safe_end_pct = round(value_to_percentile(sorted_log, safe_end_log))

    inside   = [v for v in log_ratios if lower_log <= v <= safe_end_log]
    safe_var = float(np.var(inside, ddof=1)) if len(inside) >= 2 else float("inf")

    return {
        "lower_pct"      : lower_pct,
        "safe_width"     : safe_end_pct - lower_pct,
        "safe_end_pct"   : safe_end_pct,
        "lower_log"      : round(lower_log,    4),
        "safe_end_log"   : round(safe_end_log, 4),
        "lower_ratio"    : round(math.exp(lower_log),    4),
        "safe_end_ratio" : round(math.exp(safe_end_log), 4),
        "best_score"     : round(safe_var, 6),
        "alpha"          : None,
        "mean"           : round(mu,       4),
        "std"            : round(sigma,    4),
        "skewness"       : round(skewness, 4),
        "n"              : n,
        "scan"           : [],
        "_raw"           : [],
        "k"              : k,
        "method"         : f"mu +/- {k}*sigma",
    }


# --------------------------------------------------
# 7. Classification  (Good / Safe / Alert)
# --------------------------------------------------

def classify(log_ratio: float, lower_log: float, safe_end_log: float) -> str:
    if log_ratio < lower_log:
        return "Good"
    elif log_ratio <= safe_end_log:
        return "Safe (normal)"
    else:
        return "Alert (delayed)"


# --------------------------------------------------
# 8. Visualisation  (3x3 grid)
# --------------------------------------------------

def plot_distribution(
    log_ratios : list[float],
    sz         : dict,
    qq_curv    : dict | None = None,
    qq_inf     : dict | None = None,
    save_path  : str = "estimator_plot.png",
) -> None:
    mu           = sz["mean"]
    sigma        = sz["std"]
    lower_log    = sz["lower_log"]
    safe_end_log = sz["safe_end_log"]
    lower_pct    = sz["lower_pct"]
    safe_end_pct = sz["safe_end_pct"]

    data = np.array(log_ratios)
    x    = np.linspace(data.min() - 0.2, data.max() + 0.2, 500)
    pdf  = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    fig = plt.figure(figsize=(24, 18))
    gs  = fig.add_gridspec(3, 3, hspace=0.50, wspace=0.40)
    ax1        = fig.add_subplot(gs[0, 0])
    ax2        = fig.add_subplot(gs[0, 1])
    ax_summary = fig.add_subplot(gs[0, 2])
    ax3        = fig.add_subplot(gs[1, 0])
    ax4        = fig.add_subplot(gs[1, 1])
    ax5        = fig.add_subplot(gs[1, 2])
    ax6        = fig.add_subplot(gs[2, 0])
    ax7        = fig.add_subplot(gs[2, 1])
    ax8        = fig.add_subplot(gs[2, 2])

    curv_tag = (f"  |  [QQ-curvature] p{qq_curv['lower_pct']}~p{qq_curv['safe_end_pct']}"
                if qq_curv else "")
    inf_tag  = (f"  |  [QQ-inflection] p{qq_inf['lower_pct']}~p{qq_inf['safe_end_pct']}"
                if qq_inf else "")
    fig.suptitle(
        (f"log_ratio Distribution Analysis  "
         f"(n={sz['n']}  mean={mu:.4f}  std={sigma:.4f}  skew={sz['skewness']:.4f})\n"
         f"[Primary: {sz['method']}]  Safe zone: p{lower_pct}~p{safe_end_pct}"
         + curv_tag + inf_tag),
        fontsize=11, fontweight="bold", y=1.01,
    )

    # -- (1) Histogram --------------------------------------------------
    n_bins = min(80, max(30, sz["n"] // 50))
    counts, bins, patches = ax1.hist(
        data, bins=n_bins, density=True,
        color="#90CAF9", edgecolor="white", linewidth=0.4, alpha=0.85,
    )

    inf_lo = qq_inf["lower_log"]    if qq_inf else None
    inf_hi = qq_inf["safe_end_log"] if qq_inf else None
    for patch, left in zip(patches, bins[:-1]):
        if inf_lo is not None and inf_hi is not None:
            if left < inf_lo:
                patch.set_facecolor("#EF9A9A")
            elif left <= inf_hi:
                patch.set_facecolor("#A5D6A7")
            else:
                patch.set_facecolor("#EF9A9A")
        else:
            patch.set_facecolor("#90CAF9")

    skew_val = sz.get("skewness", 0.0)
    ax1.plot(x, pdf, color="#D32F2F", linewidth=2.0,
             label=f"Normal N({mu:.3f}, {sigma:.3f}^2)  skew={skew_val:.3f}")
    ax1.axvline(mu, color="#37474F", lw=1.2, ls=":", label=f"mean = {mu:.4f}")

    if qq_inf:
        ax1.axvline(qq_inf["lower_log"],    color="#00695C", lw=1.5, ls=":",
                    label=f"Infl lower  p{qq_inf['lower_pct']} = {qq_inf['lower_log']:.4f}")
        ax1.axvline(qq_inf["safe_end_log"], color="#1A237E", lw=1.5, ls=":",
                    label=f"Infl upper  p{qq_inf['safe_end_pct']} = {qq_inf['safe_end_log']:.4f}")

    if qq_curv:
        ax1.axvline(qq_curv["lower_log"],    color="#E65100", lw=1.5, ls="-.",
                    label=f"Curv lower  p{qq_curv['lower_pct']} = {qq_curv['lower_log']:.4f}")
        ax1.axvline(qq_curv["safe_end_log"], color="#6A1B9A", lw=1.5, ls="-.",
                    label=f"Curv upper  p{qq_curv['safe_end_pct']} = {qq_curv['safe_end_log']:.4f}")

    inf_lo_pct = qq_inf["lower_pct"]    if qq_inf else sz["lower_pct"]
    inf_hi_pct = qq_inf["safe_end_pct"] if qq_inf else sz["safe_end_pct"]
    p_alert_l = mpatches.Patch(color="#EF9A9A", label=f"Alert  (p0~p{inf_lo_pct})")
    p_safe    = mpatches.Patch(color="#A5D6A7", label=f"Safe   (p{inf_lo_pct}~p{inf_hi_pct})")
    p_alert_r = mpatches.Patch(color="#EF9A9A", label=f"Alert  (p{inf_hi_pct}~p100)")
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles + [p_alert_l, p_safe, p_alert_r],
               fontsize=6.0, loc="upper right")

    ax1.set_xlabel("log_ratio  [ log(elapsed / median) ]", fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)
    ax1.set_title("(1) log_ratio Distribution + Safe Zone", fontsize=11)
    ax1.grid(axis="y", alpha=0.3)

    ax1_top = ax1.twiny()
    ratio_ticks = [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
    ratio_ticks = [t for t in ratio_ticks if data.min() - 0.2 <= t <= data.max() + 0.2]
    ax1_top.set_xlim(ax1.get_xlim())
    ax1_top.set_xticks(ratio_ticks)
    ax1_top.set_xticklabels([f"{math.exp(t):.2f}" for t in ratio_ticks], fontsize=7)
    ax1_top.set_xlabel("ratio (original scale)", fontsize=8)

    # -- (2) Q-Q Plot ---------------------------------------------------
    sorted_data   = np.sort(data)
    nn            = len(sorted_data)
    from statistics import NormalDist
    nd            = NormalDist(mu=mu, sigma=sigma)
    theoretical_q = np.array([
        nd.inv_cdf((i - 0.375) / (nn + 0.25)) for i in range(1, nn + 1)
    ])
    ax2.scatter(theoretical_q, sorted_data, s=4, alpha=0.4, color="#5C6BC0")
    ref_min = min(theoretical_q.min(), sorted_data.min())
    ref_max = max(theoretical_q.max(), sorted_data.max())
    ax2.plot([ref_min, ref_max], [ref_min, ref_max],
             color="#D32F2F", linewidth=1.5, linestyle="--", label="y = x (perfect normal)")

    if qq_curv:
        ax2.axvline(qq_curv["left_curv_theor"],  color="#E65100", lw=2.0, ls="-.",
                    label=f"Curv left  (theor={qq_curv['left_curv_theor']:.3f})")
        ax2.axvline(qq_curv["right_curv_theor"], color="#6A1B9A", lw=2.0, ls="-.",
                    label=f"Curv right (theor={qq_curv['right_curv_theor']:.3f})")
        ax2.axhline(qq_curv["lower_log"],    color="#E65100", lw=1.0, ls=":", alpha=0.7)
        ax2.axhline(qq_curv["safe_end_log"], color="#6A1B9A", lw=1.0, ls=":", alpha=0.7)
        # [v4] spike center line
        ax2.axvline(qq_curv["spike_center_theor"], color="#F9A825", lw=1.2, ls=":",
                    label=f"spike center (p{int(qq_curv['spike_center_pct'])} theor={qq_curv['spike_center_theor']:.3f})")
    if qq_inf:
        ax2.axvline(qq_inf["left_inflection_theor"],  color="#00695C", lw=1.3, ls=":",
                    label=f"Infl left  (theor={qq_inf['left_inflection_theor']:.3f})")
        ax2.axvline(qq_inf["right_inflection_theor"], color="#1A237E", lw=1.3, ls=":",
                    label=f"Infl right (theor={qq_inf['right_inflection_theor']:.3f})")

    ax2.set_xlabel("Theoretical Normal Quantile", fontsize=10)
    ax2.set_ylabel("Actual log_ratio Quantile", fontsize=10)
    ax2.set_title("(2) Q-Q Plot  (Normality Check + Boundaries)", fontsize=11)
    ax2.legend(fontsize=7.0)
    ax2.grid(alpha=0.3)
    ax2.text(0.03, 0.97,
             "Closer to line = more normal\nUpper-right deviation = heavy right tail (delay)",
             transform=ax2.transAxes, fontsize=7.5, va="top", color="#424242",
             bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

    # -- (3) Summary ----------------------------------------------------
    ax_summary.axis("off")
    curv_lines = []
    if qq_curv:
        fallback_note = "  [!] fallback" if qq_curv.get("hn_fallback_used") else ""
        lb_note = "  [FB]" if qq_curv.get("left_fallback") else ""
        rb_note = "  [FB]" if qq_curv.get("right_fallback") else ""
        curv_lines = [
            "",
            "[ QQ-Curvature Safe Zone  * MAIN  v5 LocalMin+SpikeCenter ]",
            f"  {qq_curv['method']}",
            f"  lower   : p{qq_curv['lower_pct']}  log={qq_curv['lower_log']:.4f}"
            f"  ratio={qq_curv['lower_ratio']:.4f}{lb_note}",
            f"  upper   : p{qq_curv['safe_end_pct']}  log={qq_curv['safe_end_log']:.4f}"
            f"  ratio={qq_curv['safe_end_ratio']:.4f}{rb_note}",
            f"  width   : {qq_curv['safe_width']}%p",
            f"  [v4] spike_center : p{int(qq_curv['spike_center_pct'])}"
            f"  theor={qq_curv['spike_center_theor']:.4f}",
            f"  [v4] valley_search: [{qq_curv['valley_search_lo']:.4f},"
            f" {qq_curv['valley_search_hi']:.4f}]",
            f"  [V-shape] valley_theor : {qq_curv['valley_theor']:.4f}",
            f"  [V-shape] valley_val   : {qq_curv['valley_val']:.4e}",
            f"  [V-shape] vshape_thr   : {qq_curv['vshape_thr']:.4e}"
            f"  (factor={qq_curv['vshape_sigma_factor']})",
            f"  [HalfNorm] sigma_hn    : {qq_curv['sigma_hn']:.8f}" + fallback_note,
            f"  [HalfNorm] threshold   : {qq_curv['curv_threshold']:.8f}"
            f"  (alpha={qq_curv['curv_tail_alpha']})",
            f"  [legacy]   MAD thr     : {qq_curv['legacy_threshold']:.8f}",
            f"  search range  : p{int(qq_curv['min_search_pct'])}~p{int(qq_curv['max_search_pct'])}",
        ]
    inf_lines = []
    if qq_inf:
        inf_lines = [
            "",
            "[ QQ-Inflection Safe Zone  (auxiliary) ]",
            f"  lower   : p{qq_inf['lower_pct']}  log={qq_inf['lower_log']:.4f}"
            f"  ratio={qq_inf['lower_ratio']:.4f}",
            f"  upper   : p{qq_inf['safe_end_pct']}  log={qq_inf['safe_end_log']:.4f}"
            f"  ratio={qq_inf['safe_end_ratio']:.4f}",
            f"  median_slope : {qq_inf['median_slope']:.4f}  factor={qq_inf['slope_jump_factor']}",
            f"  search range : p{int(qq_inf['min_search_pct'])}~p{int(qq_inf['max_search_pct'])}",
        ]
    summary_lines = [
        "[ Optimal Parameters  (primary method) ]",
        "",
        f"  lower_pct  = p{sz['lower_pct']}   (safe zone start)",
        f"  safe_width = {sz['safe_width']}%p  -> p{lower_pct} ~ p{safe_end_pct}",
        "",
        f"  best score = {sz['best_score']:.6f}  (lower is better)",
        "",
        "[ Classification Thresholds ]",
        f"  Good   : ratio < {sz['lower_ratio']:.4f}",
        f"           (bottom {sz['lower_pct']}%  --  faster than normal)",
        f"  Safe   : {sz['lower_ratio']:.4f} ~ {sz['safe_end_ratio']:.4f}",
        f"           (p{lower_pct} ~ p{safe_end_pct}  --  normal range)",
        f"  Alert  : ratio > {sz['safe_end_ratio']:.4f}",
        f"           (top {100 - safe_end_pct}%  --  delayed)",
    ] + curv_lines + inf_lines
    ax_summary.text(
        0.04, 0.97, "\n".join(summary_lines),
        transform=ax_summary.transAxes,
        fontsize=7.5, va="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", fc="#F8F9FA", ec="#BDBDBD", lw=1.2),
    )
    ax_summary.set_title("(3) Parameter Summary", fontsize=11)

    raw      = sz.get("_raw", [])
    best_lp  = sz["lower_pct"]
    best_sw  = sz["safe_width"]
    LINE_COLOR = "#1565C0"
    BEST_COLOR = "#D32F2F"

    def _line_data_1fix(fix_key, fix_val, vary_key):
        bucket: dict[int, list[float]] = {}
        for r in raw:
            if r[fix_key] == fix_val:
                k = r[vary_key]
                bucket.setdefault(k, []).append(r["score"])
        xs = sorted(bucket)
        ys = [min(bucket[x]) for x in xs]
        return xs, ys

    def _draw_line_chart(ax, xs, ys, best_x, xlabel, title, unit="%p"):
        ax.plot(xs, ys, color=LINE_COLOR, lw=2.0, zorder=3)
        ax.fill_between(xs, ys, min(ys), alpha=0.10, color=LINE_COLOR)
        ax.axvline(best_x, color=BEST_COLOR, lw=2.0, ls="--", zorder=4,
                   label=f"Optimal = {best_x}{unit}")
        if best_x in xs:
            best_y = ys[xs.index(best_x)]
            ax.scatter([best_x], [best_y], s=100, color=BEST_COLOR, zorder=5)
            ax.annotate(
                f"  score={best_y:.4f}",
                xy=(best_x, best_y),
                xytext=(8, 6), textcoords="offset points",
                fontsize=8, color=BEST_COLOR,
            )
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("score  (lower is better)", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ymin, ymax = min(ys), max(ys)
        margin = (ymax - ymin) * 0.15 if ymax > ymin else 0.01
        ax.set_ylim(ymin - margin, ymax + margin * 2)

    # -- (4) Score vs lower_pct -----------------------------------------
    xs3, ys3 = _line_data_1fix("safe_width", best_sw, "lower_pct")
    if xs3:
        _draw_line_chart(
            ax3, xs3, ys3, best_lp,
            xlabel=f"lower_pct  (safe zone start, {LOWER_MIN}~{LOWER_MAX}%)",
            title=f"(4) Score vs lower_pct\n(safe_width={best_sw}%p fixed)",
            unit="%",
        )
    else:
        ax3.axis("off")
        ax3.text(0.5, 0.5, "No grid data\n(sigma mode)", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=12, color="gray")

    # -- (5) Score vs safe_width ----------------------------------------
    xs4, ys4 = _line_data_1fix("lower_pct", best_lp, "safe_width")
    if xs4:
        _draw_line_chart(
            ax4, xs4, ys4, best_sw,
            xlabel=f"safe_width  ({SAFE_MIN}~{SAFE_MAX}%p)",
            title=f"(5) Score vs safe_width\n(lower_pct=p{best_lp} fixed)",
        )
    else:
        ax4.axis("off")
        ax4.text(0.5, 0.5, "No grid data\n(sigma mode)", ha="center", va="center",
                 transform=ax4.transAxes, fontsize=12, color="gray")

    # -- (6) Q-Q slope series (auxiliary) --------------------------------
    if qq_inf and "_slope" in qq_inf:
        theor_mid_inf = qq_inf["_theor_q"][:-1]
        slope_arr     = qq_inf["_slope"]
        ax5.plot(theor_mid_inf, slope_arr, color="#546E7A", lw=1.5, alpha=0.85,
                 label="Q-Q slope (smoothed)")
        ax5.axhline(qq_inf["median_slope"], color="#37474F", lw=1.2, ls=":",
                    label=f"median slope = {qq_inf['median_slope']:.3f}")
        thr_inf = qq_inf["median_slope"] * qq_inf["slope_jump_factor"]
        ax5.axhline(thr_inf, color="#BF360C", lw=1.2, ls="--",
                    label=f"threshold = {thr_inf:.3f}")
        ax5.axvline(qq_inf["left_inflection_theor"],  color="#00695C", lw=2.0, ls=":",
                    label=f"Left p{qq_inf['lower_pct']} (theor={qq_inf['left_inflection_theor']:.3f})")
        ax5.axvline(qq_inf["right_inflection_theor"], color="#1A237E", lw=2.0, ls=":",
                    label=f"Right p{qq_inf['safe_end_pct']} (theor={qq_inf['right_inflection_theor']:.3f})")
        lo_b = float(np.percentile(qq_inf["_theor_q"], qq_inf["min_search_pct"]))
        hi_b = float(np.percentile(qq_inf["_theor_q"], qq_inf["max_search_pct"]))
        ax5.axvspan(theor_mid_inf.min(), lo_b, alpha=0.08, color="gray", label="excluded range")
        ax5.axvspan(hi_b, theor_mid_inf.max(), alpha=0.08, color="gray")
        ax5.set_xlabel("Theoretical Normal Quantile", fontsize=10)
        ax5.set_ylabel("Q-Q Slope  (delta_actual / delta_theor)", fontsize=9)
        ax5.set_title("(6) Q-Q Slope Series", fontsize=10)
        ax5.legend(fontsize=7.0, loc="upper left")
        ax5.grid(alpha=0.3)
        ymin_s = np.nanpercentile(slope_arr, 2)
        ymax_s = np.nanpercentile(slope_arr, 98)
        ax5.set_ylim(ymin_s - 0.3, ymax_s + 1.0)
    else:
        ax5.axis("off")
        ax5.text(0.5, 0.5, "Q-Q slope unavailable", ha="center", va="center",
                 transform=ax5.transAxes, fontsize=12, color="gray")

    # -- (7) Q-Q Residual -----------------------------------------------
    if qq_curv and "_residual" in qq_curv:
        tq      = qq_curv["_theor_q"]
        res     = qq_curv["_residual"]
        rsmooth = qq_curv["_residual_smooth"]
        ax6.plot(tq, res, color="#B0BEC5", lw=0.8, alpha=0.6, label="raw residual")
        ax6.plot(tq, rsmooth, color="#1565C0", lw=2.0, label=f"smoothed (w={qq_curv['smooth_window']})")
        ax6.axhline(0, color="#D32F2F", lw=1.2, ls="--", label="residual = 0 (perfect normal)")
        ax6.axvline(qq_curv["left_curv_theor"],  color="#E65100", lw=2.0, ls="-.",
                    label=f"Curv left  p{qq_curv['lower_pct']}")
        ax6.axvline(qq_curv["right_curv_theor"], color="#6A1B9A", lw=2.0, ls="-.",
                    label=f"Curv right p{qq_curv['safe_end_pct']}")
        # [v4] spike center line
        ax6.axvline(qq_curv["spike_center_theor"], color="#F9A825", lw=1.5, ls=":",
                    label=f"spike center p{int(qq_curv['spike_center_pct'])}")
        lo_c = float(np.percentile(tq, qq_curv["min_search_pct"]))
        hi_c = float(np.percentile(tq, qq_curv["max_search_pct"]))
        ax6.axvspan(tq.min(), lo_c, alpha=0.10, color="gray", label="excluded range")
        ax6.axvspan(hi_c, tq.max(), alpha=0.10, color="gray")
        ax6.set_xlabel("Theoretical Normal Quantile", fontsize=10)
        ax6.set_ylabel("Residual  (actual_q - theor_q)", fontsize=9)
        ax6.set_title("(7) Q-Q Residual  [input for curvature method]\n"
                      "(S-curve shape = deviation from normal)", fontsize=10)
        ax6.legend(fontsize=7.5)
        ax6.grid(alpha=0.3)
        ax6.text(0.03, 0.97,
                 "S-curve inflection point = start of dist. deviation\n"
                 "V-shape valley of curvature = normal region center",
                 transform=ax6.transAxes, fontsize=7.5, va="top", color="#37474F",
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))
    else:
        ax6.axis("off")

    # -- (8) Q-Q Curvature  [v4: V-shape valley + spike center + inset zoom] --
    if qq_curv and "_curvature" in qq_curv:
        tq_mid     = qq_curv["_theor_mid"]
        abs_c      = qq_curv["_abs_curv"]
        abs_c_s    = qq_curv["_abs_curv_s"]
        thr_c      = qq_curv["_threshold"]
        thr_legacy = qq_curv["legacy_threshold"]
        sigma_hn   = qq_curv["sigma_hn"]
        tail_alpha = qq_curv["curv_tail_alpha"]
        c_lo, c_hi = qq_curv["curv_center_pct"]
        valley_t   = qq_curv["valley_theor"]
        vshape_thr = qq_curv["vshape_thr"]
        spike_ct   = qq_curv["spike_center_theor"]   # [v4]
        vsl_lo     = qq_curv["valley_search_lo"]      # [v4]
        vsl_hi     = qq_curv["valley_search_hi"]      # [v4]

        lo_c_val = float(np.percentile(tq_mid, c_lo))
        hi_c_val = float(np.percentile(tq_mid, c_hi))
        lo_excl  = float(np.percentile(qq_curv["_theor_q"], qq_curv["min_search_pct"]))
        hi_excl  = float(np.percentile(qq_curv["_theor_q"], qq_curv["max_search_pct"]))

        left_t  = qq_curv["left_curv_theor"]
        right_t = qq_curv["right_curv_theor"]

        ymax_c = float(np.percentile(abs_c, 98))

        # ── 메인 플롯 ─────────────────────────────────────────────────
        ax7.plot(tq_mid, abs_c,   color="#78909C", lw=0.8, alpha=0.6,
                 label="|curvature| raw")
        ax7.plot(tq_mid, abs_c_s, color="#37474F", lw=1.8,
                 label="|curvature| smoothed (V-shape)")

        ax7.axhline(vshape_thr, color="#2E7D32", lw=2.0, ls="-",
                    label=f"[V-thr] valley+σ×{qq_curv['vshape_sigma_factor']}={vshape_thr:.2e}")
        ax7.axvline(valley_t, color="#43A047", lw=1.5, ls=":",
                    label=f"V-valley theor={valley_t:.3f}")
        # [v4] spike center & valley search range
        ax7.axvline(spike_ct, color="#F9A825", lw=1.5, ls="--",
                    label=f"spike_center p{int(qq_curv['spike_center_pct'])} theor={spike_ct:.3f}")
        ax7.axvspan(vsl_lo, vsl_hi, alpha=0.06, color="#F9A825",
                    label=f"valley search [{vsl_lo:.2f}, {vsl_hi:.2f}]")

        ax7.axhline(thr_c,      color="#D32F2F", lw=1.5, ls="--",
                    label=f"[HalfNorm] thr={thr_c:.2e}  (alpha={tail_alpha})")
        ax7.axhline(sigma_hn,   color="#FF6F00", lw=1.0, ls=":",
                    label=f"sigma_hn={sigma_hn:.2e}")
        ax7.axhline(thr_legacy, color="#90A4AE", lw=1.0, ls=":",
                    label=f"[legacy MAD] thr={thr_legacy:.2e}")

        ax7.axvspan(lo_c_val, hi_c_val, alpha=0.07, color="#1565C0",
                    label=f"center p{int(c_lo)}~p{int(c_hi)}")
        ax7.axvline(left_t,  color="#E65100", lw=2.0, ls="-.",
                    label=f"Left  p{qq_curv['lower_pct']} (theor={left_t:.3f})")
        ax7.axvline(right_t, color="#6A1B9A", lw=2.0, ls="-.",
                    label=f"Right p{qq_curv['safe_end_pct']} (theor={right_t:.3f})")
        ax7.axvspan(tq_mid.min(), lo_excl, alpha=0.10, color="gray",
                    label="excluded range")
        ax7.axvspan(hi_excl, tq_mid.max(), alpha=0.10, color="gray")


        ax7.set_xlabel("Theoretical Normal Quantile", fontsize=10)
        ax7.set_ylabel("|Curvature|  |d²(residual)/dx²|", fontsize=9)
        ax7.set_title("(8) Q-Q Curvature", fontsize=10)
        ax7.legend(fontsize=6.0, loc="upper left")
        ax7.grid(alpha=0.3)
        ax7.set_ylim(0, ymax_c * 1.5)


    # -- (9) Method comparison bar chart --------------------------------
    methods  = []
    lowers   = []
    uppers   = []
    colors_l = []
    colors_u = []

    methods.append(f"Primary\n({sz['method'][:20]})")
    lowers.append(sz["lower_pct"])
    uppers.append(sz["safe_end_pct"])
    colors_l.append("#2E7D32")
    colors_u.append("#1565C0")

    if qq_curv:
        methods.append(
            f"QQ-Curvature v5\n"
            f"(factor={qq_curv['vshape_sigma_factor']}, "
            f"spike_c=p{int(qq_curv['spike_center_pct'])})"
        )
        lowers.append(qq_curv["lower_pct"])
        uppers.append(qq_curv["safe_end_pct"])
        colors_l.append("#E65100")
        colors_u.append("#6A1B9A")

    if qq_inf:
        methods.append(f"QQ-Inflection\n(factor={qq_inf['slope_jump_factor']})")
        lowers.append(qq_inf["lower_pct"])
        uppers.append(qq_inf["safe_end_pct"])
        colors_l.append("#00695C")
        colors_u.append("#1A237E")

    y_pos = np.arange(len(methods))
    bar_h = 0.35

    for i, (lo, hi, cl, cu) in enumerate(zip(lowers, uppers, colors_l, colors_u)):
        ax8.barh(i - bar_h/2, lo,      height=bar_h, color=cl, alpha=0.8, label="_nolegend_")
        ax8.barh(i + bar_h/2, hi,      height=bar_h, color=cu, alpha=0.8, label="_nolegend_")
        ax8.barh(i,           hi - lo, height=bar_h*1.6,
                 left=lo, color="none",
                 edgecolor="#37474F", linewidth=1.5, linestyle="--", label="_nolegend_")
        ax8.text(lo - 0.5, i - bar_h/2, f"p{lo}", va="center", ha="right", fontsize=8, color=cl)
        ax8.text(hi + 0.5, i + bar_h/2, f"p{hi}", va="center", ha="left",  fontsize=8, color=cu)
        ax8.text((lo + hi) / 2, i,
                 f"{hi - lo}%p", va="center", ha="center", fontsize=9,
                 fontweight="bold", color="#212121")

    ax8.set_yticks(y_pos)
    ax8.set_yticklabels(methods, fontsize=9)
    ax8.set_xlabel("Percentile (%)", fontsize=10)
    ax8.set_title("(9) Safe Zone Boundary Comparison\n"
                  "(lower=green, upper=blue, width=dashed border)", fontsize=10)
    ax8.set_xlim(0, 100)
    ax8.axvline(50, color="#D32F2F", lw=1.0, ls=":", alpha=0.5, label="p50 (median)")
    ax8.legend(fontsize=8)
    ax8.grid(axis="x", alpha=0.3)
    ax8.invert_yaxis()

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Plot] Saved -> {save_path}")


# --------------------------------------------------
# 9. Main
# --------------------------------------------------

def main(
    input_path  : str   = "4st_input.jsonl",
    plot_path   : str   = "estimator_plot.png",
    lower_min   : int   = LOWER_MIN,
    lower_max   : int   = LOWER_MAX,
    safe_min    : int   = SAFE_MIN,
    safe_max    : int   = SAFE_MAX,
    alpha       : float = SCORE_ALPHA,
    mode        : str   = "grid",
    k_or_alpha  : float = 2.0,
    # QQ-C: Half-Normal + V-shape + [v4] spike center parameters
    qq_curv_tail_alpha      : float = 0.95,
    qq_curv_center_lo_pct   : float = 20.0,
    qq_curv_center_hi_pct   : float = 80.0,
    qq_curv_smooth_frac     : float = 0.10,
    qq_curv_min_pct         : float = 5.0,
    qq_curv_max_pct         : float = 95.0,
    qq_curv_vshape_factor   : float = 1.0,
    qq_curv_spike_center_pct: float = 49.0,   # [v4] spike center
    # QQ-I: unchanged
    qq_inf_smooth_frac      : float = 0.05,
    qq_inf_slope_jump_factor: float = 1.8,
    qq_inf_min_pct          : float = 5.0,
    qq_inf_max_pct          : float = 95.0,
):
    import time

    print("=" * 70)
    print("  Delay Estimator  v5  (LocalMin SpikeCenter + V-shape valley)")
    print("  + Q-Q Curvature [v5 MAIN]  &  Inflection [auxiliary]")
    print("=" * 70)

    records = load_records(input_path)
    print(f"\n[Load] {len(records)} records\n")

    rows = compute_log_ratios(records)
    if not rows:
        print(f"No valid data (min {MIN_SAMPLES} vehicles per segment required)")
        return

    skipped = len(records) - sum(
        1 for r in records if get_elapsed_sec(r) is not None
    )
    excluded_seqs = (
        len(set(r["seq"] for r in records))
        - len(set(r["seq"] for r in rows))
    )
    print(f"[Filter] zero-distance excluded: {skipped}  |  "
          f"insufficient-vehicle seqs excluded: {excluded_seqs}\n")

    log_ratios = [r["log_ratio"] for r in rows]
    sorted_log = sorted(log_ratios)

    print("[ log_ratio distribution ]  (0.0 = same as segment median)")
    for p in [0, 10, 25, 50, 70, 75, 90, 95, 100]:
        lv    = percentile(log_ratios, p)
        rv    = math.exp(lv)
        label = "min" if p == 0 else ("max" if p == 100 else f"p{p}")
        print(f"  {label:>4}: log={lv:>8.4f}  ratio={rv:.4f}  ({(rv-1)*100:+.1f}%)")

    # -- Q-Q curvature method [MAIN v4] ---------------------------------
    print(f"\n[ Q-Q Curvature  (QQ-C)  * MAIN  v5 LocalMin+SpikeCenter ]")
    qq_curv = None
    try:
        t_curv = time.perf_counter()
        qq_curv = safe_zone_from_qq_curvature(
            log_ratios,
            smooth_frac          = qq_curv_smooth_frac,
            curv_tail_alpha      = qq_curv_tail_alpha,
            curv_center_lo_pct   = qq_curv_center_lo_pct,
            curv_center_hi_pct   = qq_curv_center_hi_pct,
            min_search_pct       = qq_curv_min_pct,
            max_search_pct       = qq_curv_max_pct,
            vshape_sigma_factor  = qq_curv_vshape_factor,
            spike_center_pct     = qq_curv_spike_center_pct,
        )
        lb = "  [L-fallback]" if qq_curv["left_fallback"]  else ""
        rb = "  [R-fallback]" if qq_curv["right_fallback"] else ""
        print(f"  [v4] spike_center=p{int(qq_curv['spike_center_pct'])}"
              f"  theor={qq_curv['spike_center_theor']:.4f}"
              f"  valley_search=[{qq_curv['valley_search_lo']:.4f}, {qq_curv['valley_search_hi']:.4f}]")
        print(f"  sigma_hn={qq_curv['sigma_hn']:.4e}  "
              f"valley_theor={qq_curv['valley_theor']:.4f}  "
              f"valley_val={qq_curv['valley_val']:.4e}  "
              f"vshape_thr={qq_curv['vshape_thr']:.4e}  "
              f"(factor={qq_curv['vshape_sigma_factor']})")
        print(f"  left  theor_q={qq_curv['left_curv_theor']:>7.4f}"
              f"  -> log={qq_curv['lower_log']:>7.4f}  ratio={qq_curv['lower_ratio']:.4f}"
              f"  -> p{qq_curv['lower_pct']}{lb}")
        print(f"  right theor_q={qq_curv['right_curv_theor']:>7.4f}"
              f"  -> log={qq_curv['safe_end_log']:>7.4f}  ratio={qq_curv['safe_end_ratio']:.4f}"
              f"  -> p{qq_curv['safe_end_pct']}{rb}")
        print(f"  => p{qq_curv['lower_pct']} ~ p{qq_curv['safe_end_pct']}"
              f"  (width {qq_curv['safe_width']}%p)"
              f"  [{time.perf_counter()-t_curv:.3f}s]")
    except Exception as e:
        print(f"  [Warning] Q-Q curvature detection failed: {e}")
        qq_curv = None

    # -- Q-Q inflection method (auxiliary) ------------------------------
    print(f"\n[ Q-Q Inflection  (QQ-I)  auxiliary ]")
    qq_inf = None
    try:
        t_qq = time.perf_counter()
        qq_inf = safe_zone_from_qq_inflection(
            log_ratios,
            smooth_frac       = qq_inf_smooth_frac,
            slope_jump_factor = qq_inf_slope_jump_factor,
            min_search_pct    = qq_inf_min_pct,
            max_search_pct    = qq_inf_max_pct,
        )
        print(f"  median_slope={qq_inf['median_slope']:.4f}  "
              f"threshold={qq_inf['median_slope']*qq_inf['slope_jump_factor']:.4f}  "
              f"(factor={qq_inf['slope_jump_factor']})")
        print(f"  left  theor_q={qq_inf['left_inflection_theor']:>7.4f}"
              f"  -> log={qq_inf['lower_log']:>7.4f}  ratio={qq_inf['lower_ratio']:.4f}"
              f"  -> p{qq_inf['lower_pct']}")
        print(f"  right theor_q={qq_inf['right_inflection_theor']:>7.4f}"
              f"  -> log={qq_inf['safe_end_log']:>7.4f}  ratio={qq_inf['safe_end_ratio']:.4f}"
              f"  -> p{qq_inf['safe_end_pct']}")
        print(f"  => p{qq_inf['lower_pct']} ~ p{qq_inf['safe_end_pct']}"
              f"  (width {qq_inf['safe_width']}%p)"
              f"  [{time.perf_counter()-t_qq:.3f}s]")
    except Exception as e:
        print(f"  [Warning] Q-Q inflection detection failed: {e}")
        qq_inf = None

    # -- Final safe zone: QQ-C / QQ-I midpoint average ------------------
    print(f"\n[ Final Safe Zone  (QQ-C + QQ-I midpoint average) ]")
    final_sz    = None
    final_alert = 0
    if qq_curv:
        final_sz = compute_final_safe_zone(qq_curv, qq_inf)
        print(f"  method : {final_sz['final_method']}")
        print(f"  => p{final_sz['lower_pct']} ~ p{final_sz['safe_end_pct']}"
              f"  (width {final_sz['safe_width']}%p)")
    else:
        print(f"  [Skip] QQ-C unavailable")

    # -- Primary grid search (reference only) ---------------------------
    t0 = time.perf_counter()
    if mode == "sigma":
        sz = safe_zone_from_sigma(log_ratios, k=k_or_alpha)
    elif mode == "grid-sigma":
        sz = recommend_safe_zone(
            log_ratios,
            lower_min=lower_min, lower_max=lower_max,
            safe_min=safe_min,   safe_max=safe_max,
            k_sigma=k_or_alpha,
        )
    else:
        sz = recommend_safe_zone(
            log_ratios,
            lower_min=lower_min, lower_max=lower_max,
            safe_min=safe_min,   safe_max=safe_max,
        )
    print(f"\n[ Grid search done  ({time.perf_counter()-t0:.2f}s) ]"
          f"  n={sz['n']}  mean={sz['mean']:.4f}  std={sz['std']:.4f}"
          f"  skewness={sz['skewness']:.4f}")

    # -- Per-vehicle classification (Final zone 기준) -------------------
    ref      = final_sz if final_sz else (qq_curv if qq_curv else sz)
    ref_log  = ref["lower_log"]
    ref_hi   = ref["safe_end_log"]
    ref_name = final_sz["final_method"] if final_sz else "QQ-C" if qq_curv else "grid"

    print(f"\n[ Per-vehicle classification  (ref: {ref_name}) ]")
    print(f"  {'vehId':<14} {'seq':>4} {'toSect':>7} {'n':>4} "
          f"{'elapsed':>8} {'median':>7} {'ratio':>7} {'log':>8}  verdict")
    print("  " + "-" * 88)

    for row in sorted(rows, key=lambda x: x["log_ratio"], reverse=True):
        verdict = classify(row["log_ratio"], ref_log, ref_hi)
        pct     = value_to_percentile(sorted_log, row["log_ratio"])
        print(f"  {row['vehId']:<14} {row['seq']:>4} {row['toSect']:>7} "
              f"{row['n_in_seq']:>4} {row['elapsed_sec']:>8.0f} "
              f"{row['median_sec']:>7.1f} {row['ratio']:>7.4f} {row['log_ratio']:>8.4f}"
              f"  {verdict}  (p{pct:.0f})")

    # -- Alert 요약 -----------------------------------------------------
    def _alert_count(res):
        return sum(
            1 for row in rows
            if classify(row["log_ratio"], res["lower_log"], res["safe_end_log"]) == "Alert (delayed)"
        )

    n_total     = sz["n"]
    curv_alert  = _alert_count(qq_curv)  if qq_curv  else "-"
    inf_alert   = _alert_count(qq_inf)   if qq_inf   else "-"
    final_alert = _alert_count(final_sz) if final_sz else "-"

    print(f"\n  Alert  QQ-C    : {curv_alert} / {n_total}")
    print(f"  Alert  QQ-I    : {inf_alert}  / {n_total}")
    print(f"  Alert  Final   : {final_alert} / {n_total}")

    # -- 결과 박스 -------------------------------------------------------
    def _print_box(title: str, res: dict, alert_n, total: int):
        lo_log = res["lower_log"]
        hi_log = res["safe_end_log"]
        print(f"\n{'='*70}")
        print(f"  [ {title} ]")
        print(f"{'='*70}")
        print(f"  {res['method']}")
        print()
        print(f"  +----------------------------------------------------------+")
        print(f"  |  [Safe zone]  p{res['lower_pct']:>2} ~ p{res['safe_end_pct']:<2}  (width {res['safe_width']:>2}%p)            |")
        print(f"  |    lower : log={lo_log:>7.4f}  ratio={res['lower_ratio']:.4f}  ({(res['lower_ratio']-1)*100:+.1f}%)  |")
        print(f"  |    upper : log={hi_log:>7.4f}  ratio={res['safe_end_ratio']:.4f}  ({(res['safe_end_ratio']-1)*100:+.1f}%)  |")
        print(f"  |                                                          |")
        print(f"  |  Classification                                          |")
        print(f"  |    ratio < {res['lower_ratio']:.4f}         -> Good   (bottom {res['lower_pct']}%)  |")
        print(f"  |    {res['lower_ratio']:.4f} <= ratio <= {res['safe_end_ratio']:.4f} -> Safe   (normal)         |")
        print(f"  |    ratio > {res['safe_end_ratio']:.4f}         -> Alert  (delayed)        |")
        print(f"  |                                                          |")
        print(f"  |  Alert vehicles : {str(alert_n):>6} / {total:>6}                       |")
        print(f"  +----------------------------------------------------------+")

    if qq_curv:
        _print_box("QQ-Curvature v5  [LocalMin + SpikeCenter]", qq_curv, curv_alert, n_total)
    if qq_inf:
        _print_box("QQ-Inflection  (auxiliary)", qq_inf, inf_alert, n_total)
    if final_sz:
        _print_box(f"FINAL  [{final_sz['final_method']}]", final_sz, final_alert, n_total)

    print()

    # -- 최종 경계 요약 (경계 + 교집합) ---------------------------------
    print("\n" + "=" * 50)
    print("  [ 최종 경계 요약 ]")
    print("=" * 50)

    if qq_curv:
        print(f"  QQ-C  : p{qq_curv['lower_pct']} ~ p{qq_curv['safe_end_pct']}"
              f"  (log {qq_curv['lower_log']:.4f} ~ {qq_curv['safe_end_log']:.4f},"
              f"  ratio {qq_curv['lower_ratio']:.4f} ~ {qq_curv['safe_end_ratio']:.4f})")
    else:
        print("  QQ-C  : 계산 실패")

    if qq_inf:
        print(f"  QQ-I  : p{qq_inf['lower_pct']} ~ p{qq_inf['safe_end_pct']}"
              f"  (log {qq_inf['lower_log']:.4f} ~ {qq_inf['safe_end_log']:.4f},"
              f"  ratio {qq_inf['lower_ratio']:.4f} ~ {qq_inf['safe_end_ratio']:.4f})")
    else:
        print("  QQ-I  : 계산 실패")

    if qq_curv and qq_inf:
        inter_lo = max(qq_curv["lower_pct"],    qq_inf["lower_pct"])
        inter_hi = min(qq_curv["safe_end_pct"], qq_inf["safe_end_pct"])
        inter_lo_log = float(np.percentile(sorted_log, inter_lo))
        inter_hi_log = float(np.percentile(sorted_log, inter_hi))
        if inter_lo < inter_hi:
            print(f"  교집합: p{inter_lo} ~ p{inter_hi}"
                  f"  (log {inter_lo_log:.4f} ~ {inter_hi_log:.4f},"
                  f"  ratio {math.exp(inter_lo_log):.4f} ~ {math.exp(inter_hi_log):.4f})")
        else:
            print(f"  교집합: 없음  "
                  f"(QQ-C p{qq_curv['lower_pct']}~p{qq_curv['safe_end_pct']}"
                  f" ∩ QQ-I p{qq_inf['lower_pct']}~p{qq_inf['safe_end_pct']} = ∅)")

    print("=" * 50)

    plot_distribution(
        log_ratios, sz,
        qq_curv  = qq_curv,
        qq_inf   = qq_inf,
        save_path= plot_path,
    )


def compute_final_safe_zone(qq_curv: dict, qq_inf: dict | None) -> dict:
    """
    QQ-C와 QQ-I의 경계값 평균 (sensitivity midpoint).
    QQ-I 없으면 QQ-C 단독 사용.
    """
    if qq_inf is None:
        return {**qq_curv, "final_method": "QQ-C only (QQ-I unavailable)"}

    lo = round((qq_curv["lower_pct"] + qq_inf["lower_pct"]) / 2)
    hi = round((qq_curv["safe_end_pct"] + qq_inf["safe_end_pct"]) / 2)

    if lo >= hi:
        print("  [Final] Average collapsed → QQ-C standalone")
        return {**qq_curv, "final_method": "QQ-C only (average collapsed)"}

    width      = hi - lo
    sorted_log = sorted(qq_curv["_actual_q"].tolist())
    lower_log    = float(np.percentile(sorted_log, lo))
    safe_end_log = float(np.percentile(sorted_log, hi))

    print(f"  [Final] Average  "
          f"lower=({qq_curv['lower_pct']}+{qq_inf['lower_pct']})/2=p{lo}  "
          f"upper=({qq_curv['safe_end_pct']}+{qq_inf['safe_end_pct']})/2=p{hi}  "
          f"width={width}%p")

    return {
        **qq_curv,
        "lower_pct"     : lo,
        "safe_end_pct"  : hi,
        "safe_width"    : width,
        "lower_log"     : round(lower_log,    4),
        "safe_end_log"  : round(safe_end_log, 4),
        "lower_ratio"   : round(math.exp(lower_log),    4),
        "safe_end_ratio": round(math.exp(safe_end_log), 4),
        "final_method"  : "QQ-C ∩ QQ-I midpoint average",
    }


if __name__ == "__main__":
    import sys
    input_file              = sys.argv[1]  if len(sys.argv) > 1  else "4st_input.jsonl"
    out_plot                = sys.argv[2]  if len(sys.argv) > 2  else "estimator_plot.png"
    mode                    = sys.argv[3]  if len(sys.argv) > 3  else "grid"
    param                   = float(sys.argv[4])  if len(sys.argv) > 4  else 2.0
    # QQ-C params
    qq_curv_tail_alpha      = float(sys.argv[5])  if len(sys.argv) > 5  else 0.95
    qq_curv_center_lo       = float(sys.argv[6])  if len(sys.argv) > 6  else 20.0
    qq_curv_center_hi       = float(sys.argv[7])  if len(sys.argv) > 7  else 80.0
    qq_curv_smooth          = float(sys.argv[8])  if len(sys.argv) > 8  else 0.10
    qq_curv_min             = float(sys.argv[9])  if len(sys.argv) > 9  else 5.0
    qq_curv_max             = float(sys.argv[10]) if len(sys.argv) > 10 else 95.0
    qq_curv_vshape          = float(sys.argv[11]) if len(sys.argv) > 11 else 1.0
    qq_curv_spike_center    = float(sys.argv[12]) if len(sys.argv) > 12 else 49.0  # [v4]
    # QQ-I params
    qq_inf_factor           = float(sys.argv[13]) if len(sys.argv) > 13 else 1.8
    qq_inf_smooth           = float(sys.argv[14]) if len(sys.argv) > 14 else 0.05
    qq_inf_min              = float(sys.argv[15]) if len(sys.argv) > 15 else 5.0
    qq_inf_max              = float(sys.argv[16]) if len(sys.argv) > 16 else 95.0

    main(
        input_path               = input_file,
        plot_path                = out_plot,
        mode                     = mode,
        k_or_alpha               = param,
        qq_curv_tail_alpha       = qq_curv_tail_alpha,
        qq_curv_center_lo_pct    = qq_curv_center_lo,
        qq_curv_center_hi_pct    = qq_curv_center_hi,
        qq_curv_smooth_frac      = qq_curv_smooth,
        qq_curv_min_pct          = qq_curv_min,
        qq_curv_max_pct          = qq_curv_max,
        qq_curv_vshape_factor    = qq_curv_vshape,
        qq_curv_spike_center_pct = qq_curv_spike_center,
        qq_inf_smooth_frac       = qq_inf_smooth,
        qq_inf_slope_jump_factor = qq_inf_factor,
        qq_inf_min_pct           = qq_inf_min,
        qq_inf_max_pct           = qq_inf_max,
    )