# 선행연구 조사 — 리뷰 사기 탐지 (김소륜 파트)

> 독립심화학습 I / 리뷰 사기 유형별 구조·텍스트·하이브리드 탐지 모델 비교 분석
> 작성일: 2026-08-17 · 작성자: 김소륜
> ⚠️ 아래 정리는 **초록·서베이·GitHub README 기반 1차 초안**입니다. 킥오프 미팅에서 강조된 "씹어먹듯 정독"(핵심 논문 2~5편)은 원문을 직접 읽어야 완성됩니다. 특히 **Results 세부 수치, Discussion(한계·재현가능성)** 항목은 원문 확인 후 보완 표시(`[보완 필요]`)를 지워주세요.

## 왜 이 논문들을 골랐나

킥오프 미팅에서 오동진 님이 짚은 문제의식 — "spectral 계열은 텍스트 임베딩을 안 쓰고 32차원 구조/행동 피처만 쓴다" — 을 검증하기 위해, **구조(structural) / 텍스트(textual) / 하이브리드(hybrid)** 세 계열에서 대표성 있는 논문을 각각 골랐습니다. 서베이 논문 2편을 먼저 훑어 전체 지형을 잡고, 그 아래 핵심 논문 7편을 배치했습니다.

| # | 논문 | 연도 | 분류 | 데이터셋 | 코드 |
|---|---|---|---|---|---|
| 1 | Shehnepoor et al., *Fraud Review Detection: Methods, Challenges, and Analysis* | 2021 | survey | Yelp/Amazon/TripAdvisor 등 | - |
| 2 | Yu et al., *Graph Learning for Fake Review Detection* | 2022 | survey | YelpChi/NYC/Zip, Amazon 등 | - |
| 3 | Dou et al., *CARE-GNN* | 2020 | structural | YelpChi, Amazon | Y |
| 4 | Liu et al., *PC-GNN* | 2021 | structural | YelpChi, Amazon, M7/M9 | Y |
| 5 | Tang et al., *BWGNN* | 2022 | structural | YelpChi, Amazon 외 | Y |
| 6 | *SplitGNN* | 2023 | structural | YelpChi, Amazon 등 | Y |
| 7 | Rayana & Akoglu, *SpEagle* | 2015 | hybrid | YelpChi/NYC/Zip | 일부(저자 페이지) |
| 8 | Fei et al., *Exploiting Burstiness* | 2013 | hybrid(행동+텍스트) | Amazon Software | N |
| 9 | Ott et al., *Finding Deceptive Opinion Spam* | 2011 | textual | 자체 수집(호텔 리뷰 800건) | 데이터 공개 |

## m×m 매트릭스 초안 (20일 대면 회의용 재료)

오동진 님 문제정의(사기 유형 × 방법론 비교)에 맞춰 위 논문들을 아래처럼 매핑해볼 수 있을 것 같습니다 (회의 때 다같이 다듬기):

| 방법론 \ 신호 | 구조/그래프만 | 텍스트만 | 구조+텍스트 |
|---|---|---|---|
| **일반 리뷰 사기** | CARE-GNN, PC-GNN, BWGNN, SplitGNN | Ott et al. | SpEagle |
| **버스트형(집중공격)** | — (구조 계열은 burst를 명시적 피처화 안 함) | — | Fei et al. (burst+content+behavior) |
| **신규계정형** | — 데이터에 계정 나이 피처 존재 여부 확인 필요 | — | — |

→ **갭:** 구조 기반 최신 GNN(CARE-GNN 이후)이 텍스트를 아예 안 쓰는 것은 확인됨. 반대로 텍스트/하이브리드 계열(SpEagle, Fei et al.)은 오히려 **오래된 방법론**(2013~2015, MRF/LBP 기반)이라 최신 임베딩(BERT류)을 안 씀 → "최신 GNN + 최신 텍스트 임베딩을 결합한 사례가 별로 없다"는 것 자체가 우리 연구의 포지셔닝 근거가 될 수 있음.

## 다음 액션

- [ ] 8/19(수)까지: 아래 논문 중 최소 2~5편 원문 정독, `[보완 필요]` 표시 채우기
- [ ] YelpChi 데이터에 실제로 "사기 유형"(신규계정/버스트 등) 라벨이 있는지 확인 (오동진 님 액션과 중복 — 회의 때 교차 확인)
- [ ] 8/20(목) 회의: 오동진·김민섭 님 조사분과 매핑 → 계획서 outline 정리
- [ ] 8/22~23(토일): 이 정리를 7원칙 포맷 기준으로 디벨롭

---

각 논문 상세는 `01_` ~ `09_` 파일 참고.
