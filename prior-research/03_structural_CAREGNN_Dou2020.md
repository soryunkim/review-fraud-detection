# Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters (Dou, Liu, Sun, Deng, Peng, Yu — CIKM 2020)

- **분류**: structural
- **인용 위치(예정)**: 방법론 (구조 계열 대표 베이스라인)
- **코드 공개 여부**: Y (https://github.com/YingtongDou/CARE-GNN)
- **원문**: https://penghao-bdsc.github.io/papers/cikm20.pdf

## Main Topic
사기꾼들이 정상 유저처럼 보이도록 특징/관계를 위장(camouflage)할 때, GNN 기반 사기 탐지기의 성능이 저하되는 문제 — "feature camouflage"와 "relation camouflage"를 어떻게 극복할 것인가.

## How
(1) label-aware similarity measure(1-layer MLP)로 유사 이웃 판별, (2) 강화학습(Bernoulli Multi-armed Bandit)으로 이웃 선택 임계값을 적응적으로 최적화하는 top-p 샘플링, (3) relation-aware aggregator로 관계 유형별 정보를 결합.

## Data
- **YelpChi**: 리뷰 45,954건, 사기 비율 14.5%, **32차원 수작업(handcrafted) 피처**만 사용 — 텍스트 임베딩 미사용
- **Amazon**: 유저 11,944명, 사기 비율 9.5%, 25차원 행동 피처
- 공개 데이터셋, 직접 접근 가능

## Experimental Setups
5%/40% 등 다양한 학습 비율로 AUC/Recall/F1 비교, 기존 GNN 베이스라인(GCN, GraphSAGE 등) 및 비GNN 베이스라인과 비교.

## Results
Yelp 5% 학습 기준 AUC 71.26% (기존 최고 62.31% 대비 대폭 개선). RL 기반 적응형 필터링이 고정 임계값 대비 유의미하게 우수.

## Significance
**우리 연구의 핵심 근거 논문 중 하나** — spectral/구조 계열이 텍스트를 전혀 쓰지 않고도 성능을 낸다는 것을 명시적으로 보여주는 대표 사례. "32차원 피처만 사용"이라는 사실이 오동진 님 문제의식의 1차 근거.

## Discussion
- 한계(원문 기준): 카멜플라주가 더 정교해지면(임계값 자체를 학습해 회피) 여전히 취약할 가능성
- **[보완 필요] 재현 가능성**: GitHub 코드 공개되어 있으므로 2주 내 파일럿 테스트 대상 1순위로 추천 (오동진 님 액션아이템과 직결)
- 우리 연구 아이디어와의 연결: 이 모델에 텍스트 임베딩을 추가 피처로 넣었을 때 어떤 사기 유형에서 성능이 달라지는지가 우리 m×m 매트릭스의 한 축이 될 수 있음
