# Review Fraud Detection

가짜 리뷰(사기 리뷰) 탐지 프로젝트 — 독립심화학습 I

사기 유형(신규 계정형, 버스트형 등)에 따라 **구조 기반 / 텍스트 기반 / 하이브리드** 방법론의 성능이 어떻게 달라지는지 m×m 매트릭스로 비교분석합니다. 기존 spectral 계열 GNN 모델들이 텍스트 임베딩을 전혀 사용하지 않는다는 문제의식에서 출발했습니다.

## 팀

오동진, 김민섭, 김소륜

## 문서

| 문서 | 내용 |
|---|---|
| [`docs/project_plan.md`](docs/project_plan.md) | 프로젝트 계획안 (배경, 문제정의, 목표, 추진전략) |
| [`docs/literature_review.md`](docs/literature_review.md) | 선행연구 정독 분석 템플릿 (7원칙 포맷) |
| [`docs/meeting_notes/`](docs/meeting_notes) | 미팅 정리본 |
| [`prior-research/`](prior-research) | 논문별 요약 및 선행연구 목록 |
| [`data/README.md`](data/README.md) | 데이터셋 정보 및 확보 계획 |

## 구조

```
docs/            프로젝트 계획, 선행연구 분석, 미팅 정리
prior-research/  논문 요약(01~09) 및 선행연구 목록
data/            데이터셋 정보 (원본/전처리 데이터는 미포함)
src/             모델 구현 (data, features, models, evaluation, utils)
```

## 일정

제출 마감: 프로젝트 계획안 8/31, 학회(KSC) 제출 10월 말. 자세한 일정은 [`docs/project_plan.md`](docs/project_plan.md) 참고.
