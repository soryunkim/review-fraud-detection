# Collective Opinion Spam Detection: Bridging Review Networks and Metadata (Rayana & Akoglu — KDD 2015, SpEagle)

- **분류**: hybrid
- **인용 위치(예정)**: 서론 / 방법론 (구조+텍스트+행동 결합의 원조 격 논문)
- **코드 공개 여부**: 일부 공개 (저자 페이지 https://shebuti.com/collective-opinion-spam-detection/)
- **원문**: https://www.andrew.cmu.edu/user/lakoglu/pubs/KDD2015_collective_opinion_spam_detection.pdf

## Main Topic
Yelp 같은 플랫폼에서 금전적 대가를 받고 작성되는 가짜 리뷰를, 리뷰 네트워크(유저-리뷰-상품)와 메타데이터를 "함께" 활용해 어떻게 탐지할 것인가.

## How
User-Review-Product 삼자 네트워크 위에서 **텍스트(내용 유사도, 느낌표/대문자 비율, 주관성 등 언어적 특징) + 행동(평점 편차, 리뷰 버스트, 시간 패턴) + 관계(네트워크 구조)**를 모두 결합. Pairwise Markov Random Field + Loopy Belief Propagation으로 collective classification 수행.

## Data
- YelpChi(시카고), YelpNYC, YelpZip — **오동진 님이 쓰기로 한 데이터셋과 동일 계열** (YelpChi 원조 논문 중 하나)
- 공개 데이터셋

## Experimental Setups
Average Precision(AP) 기준 평가, 지도학습 없이(비지도) 또는 소량 라벨(1%)로 준지도(SpEagle+) 버전 비교.

## Results
SpEagle이 기존 방법 대비 AP 0.24~0.36 큰 폭 우수. 준지도 SpEagle+는 라벨 1%만으로 AP 0.40~0.42. 경량 버전 SpLite는 성능 유지하며 속도 대폭 개선.

## Significance
**구조+텍스트+행동을 처음부터 통합한 원조 논문 중 하나** — 이후 GNN 계열(CARE-GNN 등)이 왜 텍스트를 다시 배제하는 방향으로 갔는지 대비시키기 좋은 논문. "예전엔 결합했는데 왜 최신 GNN은 안 하는가?"라는 질문을 서론에서 던질 때 핵심 대조군.

## Discussion
- 한계: 2015년 논문이라 텍스트 처리 방식이 요즘 기준(BERT류 임베딩)이 아니라 얕은 언어 통계 수준 — 우리 연구가 "이걸 최신 임베딩으로 갱신하면 어떨까"라는 각도로 차별화 가능
- **[보완 필요]** 재현 가능성: 코드가 저자 개인 페이지에만 있어 GitHub 공식 레포는 아님 — 실제 다운로드/실행 가능 여부 확인 필요
- YelpChi 원조 데이터라 오동진 님 데이터 이해에도 직접 도움
