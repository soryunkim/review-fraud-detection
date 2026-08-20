# Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection (Liu, Ao, Qin, Chi, Feng, Yang, He — WWW 2021)

- **분류**: structural
- **인용 위치(예정)**: 방법론 (구조 계열 베이스라인, CARE-GNN과 비교군)
- **코드 공개 여부**: Y (https://github.com/PonderLY/PC-GNN)
- **원문**: https://ponderly.github.io/pub/PCGNN_WWW2021.pdf

## Main Topic
사기 탐지에서 클래스 불균형(사기 유저가 극소수)이 GNN 성능을 어떻게 저해하는가, 그리고 이를 어떻게 극복할 것인가.

## How
3단계 프레임워크: (1) **Pick** — 라벨 균형 샘플러가 소수 클래스 노드에 더 높은 샘플링 확률 부여, (2) **Choose** — 학습 가능한 거리 함수로 유사한 소수 클래스 이웃은 오버샘플링, 비유사 다수 클래스 이웃은 언더샘플링, (3) **Aggregate** — 관계·이웃 정보 결합.

## Data
- **YelpChi**: 리뷰 45,954건, 불균형비 5.9:1, 100차원 피처, 관계 3종
- **Amazon**: 유저 11,944명, 불균형비 13.5:1, 100차원 피처, 관계 3종
- **M7/M9**(알리바바 금융 데이터): 18.8만~25.3만 유저, 불균형비 118~141:1, 908차원 피처, 관계 4종
- 텍스트 임베딩 미사용 — 구조화된 그래프 피처만 사용 (CARE-GNN과 동일한 문제의식 재확인)

## Experimental Setups
CARE-GNN, GraphConsis 등과 AUC/GMean 기준 비교, 데이터셋별 불균형비 다르게 설정해 강건성 테스트.

## Results
CARE-GNN·GraphConsis 대비 AUC 3~5%p, GMean 0.7~28%p 개선.

## Significance
구조 계열이 "텍스트 없이도" 지속적으로 발전해왔다는 흐름을 CARE-GNN 이후 시점에서 재확인 — 두 논문을 나란히 인용하면 "구조 계열의 텍스트 배제가 일회성이 아니라 계열 전체의 경향"이라는 주장이 탄탄해짐.

## Discussion
- 한계(원문 기준): 극단적 불균형(100:1 이상)에서도 성능이 유지되지만, 리뷰 텍스트 자체의 의미 정보는 여전히 미활용 — 우리 연구 갭과 정확히 일치
- **[보완 필요]**: M7/M9는 금융 도메인이라 리뷰 도메인과 다름 — 우리 계획서에는 YelpChi/Amazon 결과만 인용하는 것이 적절
- 재현 가능성: 코드 공개, CARE-GNN과 같은 저자 그룹(Xiang Ao 랩) 코드 스타일이라 두 개 같이 재현 테스트하기 용이 — 파일럿 테스트 2순위 추천
