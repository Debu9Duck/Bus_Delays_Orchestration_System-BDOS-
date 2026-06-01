1. 데이터 수집 및 전처리 파이프라인

GPS/이동 데이터를 수집하고, 이상치 제거 → 보간 → 속도 계산 → 이상 감지 → 정지 패턴 분석 → 정규화까지 이어지는 전처리 파이프라인입니다.

---

2. 전체 파이프라인 흐름

```
[데이터 수집]
    data_capture.py
    data_capture_route.py
         │
         ▼
[Step 0] 기본 이상치 필터링
    0st_Basic_Outlier_Filtering.py
         │
         ▼
[Step 1] 시작/종료 시간 보간
    1st_starttime_interpolation.py (first)
    1st_endtime_interpolation.py
         │
         ▼
[Step 2] 5초 단위 선형 보간 + 속도 추가
    2st_5s_linear_interpolation.py (first)
    2st_add_speed.py
         │
         ▼
[Step 3] GRU 기반 이상 감지
    gru_anomaly_detector.py
         │
         ▼
[Step 4] 정지 패턴 분석 + 정규화
    4-1st_stop_fattern.py (first)
    4-2st_Normalization.py
```

---

3-1. 데이터 수집

| 파일 | 설명 |
|------|------|
| `data_capture.py` | 원시 이동 로그 데이터 수집 |
| `data_capture_route.py` | 노선 단위 데이터 수집 |

---

3-2. Step 0 — 기본 이상치 필터링

**`0st_Basic_Outlier_Filtering.py`**

수집된 원시 데이터에서 기초적인 이상치(오류 좌표, 비정상 값 등)를 제거합니다.

---

3-3. Step 1 — 시작/종료 시간 보간

**`1st_starttime_interpolation.py`**

각 세그먼트의 시작 시간 누락값을 보간합니다.

**`1st_endtime_interpolation.py`**

각 세그먼트의 종료 시간 누락값을 보간합니다.

---

3-4. Step 2 — 5초 단위 선형 보간 및 속도 계산

**`2st_5s_linear_interpolation.py`**

데이터 포인트를 5초 간격으로 선형 보간하여 균일한 시계열 데이터를 생성합니다.

**`2st_add_speed.py`**

보간된 위치 데이터를 기반으로 속도(speed) 피처를 계산하여 추가합니다.

---

3-5. Step 3 — GRU 기반 이상 감지 *(핵심 모델)*

**`gru_anomaly_detector.py`**

GRU모델을 사용하여 시계열 데이터에서 비정상적인 이동 패턴을 감지합니다. 이 단계의 결과가 이후 분석의 기준이 됩니다.

---

3-6. Step 4 — 정지 패턴 분석 및 정규화

**`4-1st_stop_fattern.py`**

이동 데이터 내 정지(stop) 구간을 감지하고 패턴을 분류합니다.

**`4-2st_Normalization.py`**

모델 학습 및 분석에 적합하도록 피처 값을 정규화합니다.
