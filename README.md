# Bus_Delays_Orchestration_System-BDOS-
BDOS is a real-time bus analytics system that predicts delays using live GPS data and historical travel-time distributions. It models each bus as a stateful agent tracking segment progress, stops, and cumulative delay, generating adaptive ETA updates and propagating delay across remaining routes for accurate prediction.

2026 Korea land-transport data contest - 2026 국토교통 데이터 활용 경진대회

# 연구 요약
- 기존 버스관리시스템(BMS/BIS)은 정류장 도착 시간과 구간 거리 중심의 ETA 산출 구조로 인해 신호 대기, 교통 혼잡, 정차 시간 등 실제 도로 환경의 복합 변수를 충분히 반영하지 못하는 한계가 존재하였다.
- 이에 따라 본 연구에서는 서울시 공공 API 기반 실시간 운행 데이터를 활용하여 버스의 실제 주행 흐름과 누적 지연을 분위수 분포 기반으로 분석하는 BDOS(Bus Delays Orchestration System)를 설계·구현하였다.
- BDOS는 서울 160번 노선을 실증 대상으로 하여, 발차·도착시간 역추정, 5초 간격 선형 보간, GRU Autoencoder 기반 이상치 제거, 정지 패턴 추출, 분위수 기반 이동 정규화 등 5단계 전처리 파이프라인을 구축하였다.
- 또한 Q-Q plot 기반 분석을 통해 분위수 안전지대(p12~p86)를 도출하고, 이를 바탕으로 이상치 지연(Anomaly Delay)과 미세 누적 지연(Micro Delay)을 분리 추적하는 Dual-Delay 구조를 제안하였다.
- 본 시스템은 단순 정보 제공 중심의 기존 BMS를 넘어, 실시간 운행 성향 분석과 지연 극복량 동적 배분을 수행하는 능동적 정시성 관리 구조로 확장하는 것을 목표로 하며, 향후 전국 노선 확대 및 초정밀 버스 위치 데이터 기반 고도화를 목표로 한다.

# 전처리 & 분석
데이터 수집 및 전처리
- 서울특별시 버스 위치정보조회 서비스와 노선정보조회 서비스를 활용하여 서울 160번 노선의 실시간 운행 데이터를 수집하였다.
- 불규칙한 수신 간격 문제를 해결하기 위해 발차·도착시간 역추정 및 5초 간격 선형 보간을 수행하여 균일 시계열 데이터를 구축하였다.
- GPS 튐 현상, 비정상 속도 패턴 등 이상 데이터를 제거하기 위해 GRU Autoencoder 기반 비지도 이상치 탐지 모델을 적용하였다.

Hysteresis 기반 정차 패턴 추출
- GPS 진동에 의한 허위 정차 판정을 줄이기 위해 Enter/Exit 이중 임계값 기반 Hysteresis 구조를 적용하였다.
- Grid Search를 통해 Enter 5.0km/h, Exit 10.0km/h 조합을 최적 임계값으로 선정하였다.

분위수 기반 이동 정규화
- 차량별·시간대별 이동 특성을 동일 기준으로 비교하기 위해 분위수 기반 정규화를 수행하였다.
- 구간별 p10, p50, p90 분위수 통계를 산출하여 이후 ETA 및 지연 판정 엔진의 기준 데이터로 활용하였다.

GRU Autoencoder 기반 이상치 기각
- GPS 오류, 순간적인 위치 튐, 통신 오차 등으로 발생하는 비정상 궤적 제거를 위해 GRU Autoencoder 기반 비지도 이상치 탐지 모델을 적용하였다.
- 정상 차량 이동 패턴을 자기입출력(Self Reconstruction) 방식으로 학습하고, 복원 오차(MSE)를 기반으로 이상 여부를 판정하였다.

Q-Q Plot 기반 안전지대 분석
- Q-Q plot 분석을 활용하여 정규성 이탈 시점을 탐지하였다.
- QQ-Curvature(2차 미분 기반)와 QQ-Inflection(1차 미분 기반)을 병렬 비교하여 민감도 스펙트럼을 분석하였다.
- 두 방법론의 교집합을 기반으로 p12~p86 분위수 안전지대를 최종 경계값으로 선정하였다.

Dual-Delay 이중 채널 구조
- 이상치 지연(Anomaly Delay)과 미세 누적 지연(Micro Delay)을 분리 추적하는 Dual-Delay 구조를 설계하였다.
- EMA 기반 분위수 평활화를 적용하여 급격한 노이즈 반응을 줄이고 실제 지연 추세를 안정적으로 추적하였다.

# 분석결과
분위수 기반 안전지대 도출
- QQ-Curvature 방식은 p12~p86 수준의 좁은 안전구간을 도출하며 조기 이상 감지에 강점을 보였다.
- QQ-Inflection 방식은 p5~p95 수준의 넓은 안전구간을 형성하여 극단적 지연 상황 중심으로 판정하는 특성을 보였다.
- 두 방식의 교집합을 활용하여 p12~p86 구간을 최종 Safe Zone으로 확정하였다.
 
실시간 지연 추적
- 시스템은 raw_pct 기반 현재 주행 분위수를 계산하고, 안전지대 이탈 여부에 따라 Anomaly Channel과 Micro Channel을 동적으로 활성화하였다.
- 실제 장시간 신호 대기 및 교통 혼잡 상황에서 이상치 지연 누적과 지연 극복량 분배가 정상적으로 동작함을 확인하였다.
 
ETA 및 지연 극복량 분배
- 잔여 구간별 정차 확률 및 평균 정차시간 기반 가중치를 적용하여 지연 극복량을 동적으로 배분하였다.
- 속도 조정 여지가 높은 구간에 더 큰 지연 회복량을 할당함으로써 급격한 속도 변화 없이 정시성 회복을 유도하였다.
 
# 참고문헌
- 김승일, 김영찬 and 이청원. (2006). 버스정보시스템(BIS) 정류장도착예정시간 시스템오차 연구. 대한교통학회지, 24(4), 117-127.
- 박철영, 김홍근, 신창선, 조용윤 and 박장우. (2017). 은닉 마르코프 모델을 이용한 버스 정보 시스템의 도착 시간 예측. 정보처리학회논문지. 컴퓨터 및 통신시스템, 6(4), 189-196.
- 박철영, 김홍근, 신창선, 조용윤 and 박장우. (2017). 버스의 정차시간을 고려한 장기 도착시간 예측 모델. 정보처리학회논문지. 컴퓨터 및 통신시스템, 6(7), 297-306.
  
