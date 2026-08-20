# Rethinking Graph Neural Networks for Anomaly Detection (Tang et al. — ICML 2022, BWGNN)

- **분류**: structural (spectral)
- **인용 위치(예정)**: 방법론 (spectral 계열 대표 논문 — 오동진 님 문제의식의 핵심 근거)
- **코드 공개 여부**: Y (https://github.com/squareRoot3/Rethinking-Anomaly-Detection)
- **원문**: https://arxiv.org/abs/2205.15508

## Main Topic
그래프의 스펙트럴(주파수) 관점에서, 이상치(anomaly)가 존재하면 스펙트럴 에너지 분포가 고주파 쪽으로 "우측 이동(right-shift)"하는 현상이 나타난다 — 이를 GNN 설계에 어떻게 반영할 것인가.

## How
Beta Wavelet Graph Neural Network(BWGNN) 제안 — 스펙트럴·공간적으로 지역화된 밴드패스 필터를 사용해 이상치로 인한 고주파 신호 변화를 효과적으로 포착.

## Data
대규모 이상 탐지 벤치마크 4종 (YelpChi·Amazon 포함으로 추정 — **[보완 필요]** 원문에서 정확한 데이터셋 리스트·피처 차원 확인 필요).

## Experimental Setups
[보완 필요] — 검색 결과에 상세 실험 세팅(비교 베이스라인, 평가지표별 수치)이 충분히 잡히지 않음. 원문 표 직접 확인 권장.

## Results
[보완 필요] — 벤치마크 전반에서 우수한 성능을 보였다는 정성적 서술만 확보. 정량 수치는 원문 Table 확인 필요.

## Significance
**"spectral"이라는 용어 자체가 이 논문 계열에서 유래** — 오동진 님이 언급한 "spectral 계열 논문 7편"이 이 논문의 스펙트럴 이론 위에서 후속 연구된 것일 가능성이 높음. 우리 연구의 문제의식(spectral=구조/행동 피처만 사용)을 이론적으로 뒷받침하는 핵심 논문.

## Discussion
- **[보완 필요, 우선순위 높음]**: 이 논문은 정독 대상 최우선 추천 — "왜 스펙트럴 GNN 연구들이 텍스트를 배제해왔는가"에 대한 이론적 실마리(고주파 이상 신호와 텍스트 임베딩의 관계?)가 여기 있을 가능성
- 재현 가능성: 코드 공개, 비교적 간결한 구조로 추정 — 파일럿 테스트 후보
