# review-fraud-detection

가짜 리뷰(사기 리뷰) 탐지에서 **사기 유형별로 구조 기반(structural) / 텍스트 기반(textual) / 하이브리드(hybrid) 방법론의 성능이 어떻게 달라지는가**를 비교분석하는 연구 프로젝트입니다.

- **과목:** 독립심화학습 I (경희대학교 빅데이터응용학과)
- **지도교수:** 김민경 교수님
- **팀원:** 오동진, 김민섭, 김소륜
- **목표 학회:** 한국소프트웨어종합학술대회(KSC) — 제출 마감 목표 10월 말

## 연구 배경 및 방향

기존 가짜 리뷰 탐지 연구는 크게 spectral 계열과 spatial 계열로 나뉘는데, spectral 계열 논문들은 텍스트 임베딩(의미 정보)을 거의 사용하지 않고 구조/행동 피처(예: 32차원 사전 가공 피처)만 사용하는 경향이 있습니다. 반면 spatial 계열은 텍스트 임베딩을 활용하는 경우가 많습니다.

이 프로젝트는 새로운 SOTA 모델을 만드는 것이 아니라, **리뷰 사기 유형(예: 신규 계정형, 버스트형 등)에 따라 구조 기반 / 텍스트(컨텍스트) 기반 / 하이브리드 방법론의 성능이 어떻게 달라지는지를 m × m 비교 매트릭스로 분석**하여, 어떤 유형에서 어떤 정보(구조 vs 의미)가 더 중요한지에 대한 인사이트를 도출하는 것을 목표로 합니다. 향후 이 결과는 해석 가능성(interpretability) 향상을 위한 기초 연구로 포지셔닝합니다.

- 데이터셋(초기): Yelp 시카고 레스토랑 리뷰 데이터 (가짜 리뷰 비율 약 13%)
- 비교 관점: 구조 중심 모델 / 텍스트 컨텍스트 중심 모델 / 하이브리드 모델
- 평가축: 사기 유형(예: 신규 계정 여부, 리뷰 버스트 여부 등) × 방법론

자세한 배경은 [`docs/meeting_notes/2026-08-11_kickoff.md`](docs/meeting_notes/2026-08-11_kickoff.md) 참고.

## 리포지터리 구조

```
review-fraud-detection/
├── docs/                     # 연구 계획, 문헌조사, 미팅 기록
│   ├── project_plan.md       # 8/31 제출용 프로젝트 계획안 템플릿
│   ├── literature_review.md  # 선행연구 조사 정리 (7요소 프레임워크)
│   └── meeting_notes/        # 팀/지도교수 미팅 기록
├── data/
│   ├── raw/                  # 원본 데이터 (git에 커밋하지 않음)
│   └── processed/            # 전처리된 데이터 (git에 커밋하지 않음)
├── src/
│   ├── data/                 # 데이터 로딩/전처리 코드
│   ├── features/
│   │   ├── structural/       # 구조 기반 피처/모델 (그래프 구조, 행동 피처 등)
│   │   ├── textual/          # 텍스트 임베딩 기반 피처/모델
│   │   └── hybrid/           # 구조+텍스트 결합 모델
│   ├── models/                # 베이스라인 및 비교 대상 모델
│   ├── evaluation/            # 평가 지표, 사기 유형별 비교 매트릭스 계산
│   └── utils/                 # 공통 유틸리티
├── notebooks/                 # 탐색적 분석(EDA), 실험용 노트북
├── experiments/                # 실험 설정 및 로그
├── results/                    # 실험 결과, 표/그림
├── requirements.txt
└── README.md
```

## 시작하기

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 데이터

원본/전처리 데이터는 용량 문제로 git에 커밋하지 않습니다(`.gitignore` 참고). 데이터 출처와 다운로드 방법은 `data/README.md`에 기록합니다.

## 참고

- 이 저장소는 현재 **비공개(private)**로 시작하며, 학회 발표 시즌에 맞춰 공개 전환을 고려합니다.
