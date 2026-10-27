# Review Fraud Detection

가짜 리뷰(사기 리뷰) 탐지 프로젝트 — 독립심화학습 I (지도교수 김민경 교수님)

**그래프 신경망 기반 사기 탐지의 조건적 유효성: 시간적 데이터 누수 통제 환경에서의 실증 분석**
— KSC 2026 투고. 최종본: [`docs/paper/KSC26_1차최종제출_오동진팀.docx`](docs/paper/KSC26_1차최종제출_오동진팀.docx)

## 요약

GNN 기반 리뷰 사기 탐지 연구는 대부분 리뷰를 무작위로 분할해 평가합니다. 이렇게 하면 채점 대상보다
나중에 작성된 같은 작성자의 리뷰가 학습 그래프에 들어가는 시간적 누수가 생깁니다. 이 저장소는
YelpZip 2014년 리뷰 180,659건으로 다음을 실험합니다. 피처·이력 구분·그래프 연결은 모두 작성 시점까지의
정보로만 재구성했고, 평가는 시간 순서(rolling)로 했습니다. 관계 정보의 기여는 GraphSAGE와 같은 피처를
쓰는 MLP의 PR-AUC 차이로 측정했습니다.

- 무작위 분할에서 보이는 같은 작성자 관계(R-U-R)의 기여 중 **약 28%가 미래 정보 유입**에서 온 것입니다.
- **이력 보유** 리뷰에 같은 작성자 관계를 쓰면 기여가 크게 증폭됩니다(성공 조건).
- 같은 상점 기반 관계(R-S-R, R-T-R)는 이웃 대부분이 정상 리뷰여서 성능이 오히려 떨어집니다(실패 조건).
- **이력 미보유** 리뷰에서는 기여가 거의 없습니다(한계 조건). 필터링 리뷰의 약 82%가 이 구간에 속합니다.

## 팀

오동진, 김민섭, 김소륜

## 구조

```
docs/
  paper/            논문 최종본·그림·학회 양식 (중간본은 paper/archive/)
  meeting_notes/    교수님 미팅 정리 (시간순 목록은 README)
  progress_reports/ 진행 보고서
  paper_draft.md    마크다운 원고 (v3 시점)
prior-research/     선행연구 요약(01~09)과 목록
data/               데이터셋 정보 (원본·전처리 데이터는 커밋하지 않음)
src/                재사용 모듈 (data, features, models, evaluation)
scripts/            번호순 실행 스크립트 01~20
results/            실험 결과 JSON/CSV (폴더별 README 참고)
```

## 재현

```bash
pip install -r requirements.txt
# data/README.md 대로 YelpZip을 data/ 에 둔 뒤
python scripts/01_build_reviews.py
# ... 번호 순서대로. 스크립트별 입력·출력은 scripts/README.md
```

## 주요 문서

| 문서 | 내용 |
|---|---|
| [`docs/paper/README.md`](docs/paper/README.md) | 논문 파일 안내 (최종본·중간본) |
| [`docs/meeting_notes/`](docs/meeting_notes) | 미팅 정리. 최근 결정사항은 10/7 문서 |
| [`docs/progress_reports/2026-10-01_progress_report.md`](docs/progress_reports/2026-10-01_progress_report.md) | 최종 수치(rolling + 동률 재분류) 확정 |
| [`scripts/README.md`](scripts/README.md) | 파이프라인 실행 순서 |
| [`data/README.md`](data/README.md) | 데이터셋 정보 |
| [`docs/project_plan.md`](docs/project_plan.md) | 초기 프로젝트 계획안 (8월, 이후 방향이 바뀜) |
