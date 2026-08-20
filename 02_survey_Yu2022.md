# Graph Learning for Fake Review Detection (Yu, Ren, Li, Naseriparsa, Xia, 2022)

- **분류**: survey
- **인용 위치(예정)**: 서론 / 방법론 (GNN 계열 베이스라인 선정 근거)
- **코드 공개 여부**: N (서베이 논문)
- **원문**: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2022.922589/full

## Main Topic
그래프 러닝 기법들이 리뷰의 속성(attribute)과 관계(relation) 정보를 함께 활용해 가짜 리뷰를 얼마나 효과적으로 탐지하는가 — GNN 기반 방법론 전반을 정리한 서베이.

## How
지도학습(라벨된 노드에 대한 이진분류) vs 비지도학습(클러스터링), generation-based / contrast-based 등 세부 카테고리로 GNN 기반 탐지 기법을 분류.

## Data
YelpChi, YelpNYC, YelpZip, Amazon 계열, BeerAdvocate, RateBeer, CellarTracker, SWMReview, Epinions — GNN 기반 리뷰 사기 탐지 연구에서 표준적으로 쓰이는 데이터셋 총망라.

## Experimental Setups
서베이이므로 자체 실험 없음. 대표 방법론(PC-GNN, AO-GNN 등)의 일반화 성능을 교차 비교.

## Results
PC-GNN 계열이 일반화 성능에서 우수하다고 평가. 구조 관계를 강조하는 taxonomy 특성상 **텍스트 임베딩 활용 사례가 상대적으로 적게 다뤄짐** — 이는 오동진 님이 지적한 "spectral 계열의 텍스트 배제" 현상이 이 분야 서베이 수준에서도 관찰된다는 방증으로 인용 가능.

## Significance
가짜 리뷰 탐지 GNN 서베이 중 비교적 최신(2022) — 우리 연구가 "이 서베이 이후에도 구조-텍스트 결합이 부족하다"는 갭을 짚는 근거로 활용 가능.

## Discussion
- 한계: 원문에서 "설명가능성 부족, 대규모 데이터 계산 효율, 경량 모델 필요성"을 challenge로 명시 — 우리 연구의 interpretability 포지셔닝과 직접 연결됨(계획서 배경 문단에 인용 추천)
- [보완 필요] AO-GNN이 정확히 어떤 방법론인지 원문에서 확인 후 구조/하이브리드 분류 재검토
- 재현 가능성: 서베이라 해당 없음
