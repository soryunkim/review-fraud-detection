# Exploiting Burstiness in Reviews for Review Spammer Detection (Fei, Mukherjee, Liu, Hsu, Castellanos, Ghosh — ICWSM 2013)

- **분류**: hybrid (행동+텍스트)
- **인용 위치(예정)**: 서론 / 선행연구 (사기 "유형"을 명시적으로 다루는 근거 논문 — 버스트형)
- **코드 공개 여부**: N
- **원문**: https://www.cs.uic.edu/~liub/publications/ICWSM-2013-Geli-burst.pdf

## Main Topic
리뷰가 짧은 시간에 몰리는 "버스트(burst)" 현상을 신호로 활용해, 특정 시점에 집중적으로 작성되는 스팸 캠페인을 어떻게 탐지할 것인가.

## How
(1) Kernel Density Estimation으로 시간대별 리뷰 밀집 구간(버스트) 탐지 (2주 단위 bin), (2) 검증구매비율/평점편차/버스트 리뷰 비율/내용 유사도/리뷰어 버스트성 5가지 행동 지표 설계, (3) 리뷰어-공출현(co-occurrence)을 Markov Random Field로 모델링해 Loopy Belief Propagation으로 추론.

## Data
Amazon 소프트웨어 카테고리: 리뷰 210,761건, 리뷰어 50,704명, 상품 112,953개 → 버스트 10,251건 탐지. 공개 데이터셋 기반(Amazon 크롤링).

## Experimental Setups
LBP(전체 리뷰) vs 버스트 한정 분류 비교, K-means 클러스터링과 비교, 전문가 3인 평가(카파 0.84)로 정성 검증.

## Results
전체 리뷰 기준 정밀도 77.8%/정확도 77.6%, 버스트 한정 분류는 정밀도 83.7%/정확도 77.6% — 버스트 신호에 집중할수록 정밀도 상승.

## Significance
**우리 m×m 매트릭스에서 "버스트형" 사기 유형에 해당하는 사실상 유일한 후보** — 오동진 님이 언급한 사기 유형(신규계정형/버스트형) 중 버스트형을 명시적으로 다루는 몇 안 되는 논문. 구조 계열 논문들(CARE-GNN 등)은 버스트를 명시적 피처로 쓰지 않는다는 대조점도 확인됨.

## Discussion
- 한계: Amazon 소프트웨어 카테고리 한정이라 리뷰 도메인(레스토랑) 일반화는 불확실 — YelpChi에 버스트 라벨이 있는지 별도 확인 필요(오동진 님 액션아이템과 연결)
- **[보완 필요]** 재현 가능성: 코드 미공개, 오래된 논문(2013)이라 재현보다는 "버스트 피처 정의"를 우리 데이터에 적용하는 참고용으로 활용하는 편이 현실적
- 우리 연구 방향: 이 논문의 버스트 피처를 구조/텍스트 모델에 각각 추가했을 때 성능 변화를 보면 m×m 매트릭스의 "버스트형" 행을 채울 수 있음
