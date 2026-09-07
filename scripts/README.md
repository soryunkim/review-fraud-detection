# scripts/ — 파이프라인 실행 스크립트

번호 순서대로 실행합니다. 재사용 가능한 모듈은 `src/` 아래에 두고,
여기에는 "한 번 돌려서 산출물을 만드는" 진입점만 둡니다.

## 실행 순서

| # | 스크립트 | 입력 | 출력 | 소요 |
|---|---|---|---|---|
| 01 | `01_build_reviews.py` | `data/yelpzip.csv` | `data/processed/reviews.parquet` | 약 3분 |
| 02 | `02_embed_text.py` | `reviews.parquet` | `data/processed/emb_minilm_384.npy` | 1.5~4.7시간 |
| 03 | `03_build_features.py` | `reviews.parquet` | `data/processed/features_handcrafted.parquet` (조건 A 입력) | 수 분 |
| 04 | `04_add_burst_kde.py` (오동진) | `reviews.parquet` | `reviews.parquet` (+ `type_burst_kde`, `type_class_kde` 컬럼) | 수 분 |
| 05 | `05_train_gnn_condition_a.py` | `reviews.parquet` + `features_handcrafted.parquet` | `results/condition_a/<type>_<backbone>.json` | 유형·표본 크기에 따라 초~분 단위 |

```bash
python scripts/01_build_reviews.py
python scripts/02_embed_text.py 2>&1 | tee logs/embed.log
python scripts/03_build_features.py
python scripts/04_add_burst_kde.py
python scripts/05_train_gnn_condition_a.py --type-class new
```

## 03 피처 생성에 대해

`nltk` SentiWordNet 사전이 필요합니다. 최초 1회만 다운로드:

```python
import nltk
nltk.download("sentiwordnet")
nltk.download("wordnet")
```

## 05 조건 A 학습에 대해

- R-U-R(같은 작성자) 그래프 + 수작업 피처(38차원) + 바닐라 GCN/GraphSAGE.
  `--type-class`로 사기 서브유형(신규계정형 우선)을 골라 그 부분집합 안에서만
  그래프를 구성한다 — 다른 유형 리뷰와는 연결하지 않는다.
- 빠른 반복 실험은 `--frac 0.075` 처럼 작성자 단위 표본을 쓴다(업체 단위로 자르면
  라벨·R-U-R 구조가 왜곡되므로 항상 작성자 단위, `src/data/loaders.sample_by_user`와
  같은 원칙).
- `--exclude-cols singleton`으로 라벨 정의(신규계정형=리뷰 수 기준)에 관여한 피처를
  뺀 버전을 병행 보고할 수 있다(라벨 누수 점검용).

## 02 임베딩에 대해

- **전부 로컬 연산입니다.** API 호출도 토큰 소비도 없습니다
  (최초 1회 모델 다운로드에만 인터넷 필요).
- 5,000건마다 중간 저장하므로 **중간에 끊겨도 다시 실행하면 이어집니다.**
  중간 파일은 `data/processed/_emb_parts/` 에 쌓입니다.
- 동작만 빠르게 확인하려면: `python scripts/02_embed_text.py --limit 2000`
- 기본은 MiniLM-L6(384차원). BERT-base(768차원)는 `--model bert`.
  차원이 크면 구조 셀(수작업 피처)과의 차원 불균형이 커지므로,
  실험 시 PCA 축소 버전을 함께 보고하는 것을 권장합니다.

## 환경 제약

- **DGL은 Python 3.13+ 를 지원하지 않습니다.** FraudSquad 등 공개 구현이 DGL 의존이라
  그대로 실행되지 않습니다. 그래프는 `scipy.sparse`로 만들고, 모델은 `torch` /
  `torch-geometric` 으로 구현하는 방향을 권장합니다.
- 임베딩·학습 모두 CPU 기준으로 시간을 측정했습니다. GPU가 있으면 임베딩은 1시간 내외.

## 규칙

`reviews.parquet` 의 행 순서(`review_id` 0..N-1)가 모든 산출물의 정합 기준입니다.
`.npy` 에는 ID가 없으므로 **재정렬 금지**. 자세한 내용은 [`../data/README.md`](../data/README.md).
