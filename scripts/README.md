# scripts/ — 파이프라인 실행 스크립트

번호 순서대로 실행합니다. 재사용 가능한 모듈은 `src/` 아래에 두고,
여기에는 "한 번 돌려서 산출물을 만드는" 진입점만 둡니다.

## 실행 순서

| # | 스크립트 | 입력 | 출력 | 소요 |
|---|---|---|---|---|
| 01 | `01_build_reviews.py` | `data/yelpzip.csv` | `data/processed/reviews.parquet` | 약 3분 |
| 02 | `02_embed_text.py` | `reviews.parquet` | `data/processed/emb_minilm_384.npy` | 1.5~4.7시간 |

```bash
python scripts/01_build_reviews.py
python scripts/02_embed_text.py 2>&1 | tee logs/embed.log
```

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
