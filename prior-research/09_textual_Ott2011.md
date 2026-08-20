# Finding Deceptive Opinion Spam by Any Stretch of the Imagination (Ott, Choi, Cardie, Hancock — ACL 2011)

- **분류**: textual
- **인용 위치(예정)**: 서론 (텍스트 단독 접근의 원조 격 논문, 구조 계열과의 극단적 대조군)
- **코드 공개 여부**: 데이터셋 공개(op_spam), 모델 코드는 논문 발표 시점 특성상 별도 공식 레포 없음
- **원문**: https://aclanthology.org/P11-1032/

## Main Topic
그래프나 네트워크 정보 없이, **순수 텍스트만으로** 사람이 작성한 가짜(허위) 리뷰를 탐지할 수 있는가 — 그리고 인간 평가자보다 잘할 수 있는가.

## How
Mechanical Turk로 허위 리뷰를 직접 작성시켜 골드 라벨 데이터셋 구축 → n-gram 피처 + LIWC(언어심리 통계) 피처를 결합해 SVM으로 분류.

## Data
호텔 리뷰 800건(진짜 400 + Turk로 작성한 허위 400) — 학계에서 널리 재사용되는 "op_spam" 데이터셋, 공개.

## Experimental Setups
인간 평가자(3인) vs 여러 텍스트 분류기(n-gram만/LIWC만/결합) 정확도 비교.

## Results
n-gram+LIWC 결합 SVM이 정확도 약 89.8%로 인간 평가자(약 50~60% 수준, 거의 랜덤)를 크게 상회.

## Significance
**"텍스트만으로도 충분히 잘 탐지된다"는 것을 보여준 원조 논문** — 구조 계열(CARE-GNN 등)이 텍스트를 배제하는 것과 정반대 극단에 위치. 우리 연구의 "구조 vs 텍스트" 축 양 끝단을 이 논문과 CARE-GNN으로 나란히 놓고 설명하면 서론이 명확해짐.

## Discussion
- 한계: 데이터가 Turk로 "작성된" 허위 리뷰라 실제 상업적 가짜 리뷰(대가성)와는 성격이 다를 수 있음 — 우리가 쓸 YelpChi(실제 필터링된 리뷰)와의 차이를 계획서에서 짚어줄 필요
- 재현 가능성: 데이터 공개, 방법(SVM+LIWC)은 구현 난이도 낮아 재현 용이 — 다만 최신 임베딩(BERT 등) 대비 베이스라인 성격으로만 활용 권장
- **[보완 필요]**: LIWC 피처가 정확히 어떤 카테고리인지 원문 Table 확인해 우리 텍스트 피처 설계에 참고할 것
