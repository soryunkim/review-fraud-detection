"""
데이터 로딩 헬퍼.

임베딩 .npy 에는 ID가 없고 행 순서가 유일한 정합 수단이다.
직접 np.load / read_parquet 하지 말고 여기 함수를 쓰면 정합을 자동으로 검사한다.

    from src.data.loaders import load_reviews, load_embeddings, sample_by_user

    df  = load_reviews()
    emb = load_embeddings()            # (N, 384) — df 와 행 1:1 대응 보장
    idx = sample_by_user(df, frac=0.075)
    df_s, emb_s = df.loc[idx], emb[idx]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

REVIEWS = PROCESSED / "reviews.parquet"
DEFAULT_EMB = PROCESSED / "emb_minilm_384.npy"

# 모델에 피처로 넣으면 안 되는 컬럼.
# user_id 는 숫자 크기만으로 사기율이 5배 차이난다(익명화 순서의 부산물).
# 그래프 연결에만 쓰고 피처로는 쓰지 말 것.
ID_COLUMNS = ["review_id", "user_id", "prod_id"]


def load_reviews(columns: list[str] | None = None) -> pd.DataFrame:
    """reviews.parquet 로드. review_id 0..N-1 순서를 그대로 유지한다."""
    df = pd.read_parquet(REVIEWS, columns=columns)
    if "review_id" in df.columns:
        if not df["review_id"].is_monotonic_increasing:
            raise ValueError("review_id 순서가 깨졌습니다. 재정렬하지 마세요.")
    return df


def load_embeddings(path: Path | str = DEFAULT_EMB, mmap: bool = True,
                    n_expected: int | None = None) -> np.ndarray:
    """
    임베딩 로드.

    mmap=True 면 디스크에서 지연 로드한다(891MB를 통째로 RAM에 올리지 않음).
    일부 행만 쓸 때는 mmap 을 켜두고 인덱싱하면 필요한 부분만 읽는다.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없습니다. 드라이브에서 받거나 "
            f"`python scripts/02_embed_text.py` 로 생성하세요."
        )
    emb = np.load(path, mmap_mode="r" if mmap else None)
    if n_expected is not None and emb.shape[0] != n_expected:
        raise ValueError(f"행 수 불일치: 임베딩 {emb.shape[0]} != 기대 {n_expected}")
    return emb


def load_aligned(emb_path: Path | str = DEFAULT_EMB, mmap: bool = True):
    """reviews + embeddings 를 함께 로드하고 행 수 일치를 검사한다."""
    df = load_reviews()
    emb = load_embeddings(emb_path, mmap=mmap, n_expected=len(df))
    return df, emb


def sample_by_user(df: pd.DataFrame, frac: float = 0.075, seed: int = 42) -> np.ndarray:
    """
    작성자 단위 표본 추출. 뽑힌 작성자의 리뷰를 전부 가져온다.

    업체 단위로 자르면 안 된다 — 다작 작성자의 리뷰가 일부만 들어와
    신규계정형 라벨의 약 40%가 거짓이 되고, R-U-R 엣지가 제곱으로 무너진다.
    작성자 단위로 통째로 가져오면 두 문제가 모두 사라진다.

    반환: 표본에 해당하는 행 위치(정렬된 정수 인덱스)
    """
    rng = np.random.default_rng(seed)
    users = df["user_id"].unique()
    picked = set(rng.choice(users, size=int(len(users) * frac), replace=False).tolist())
    mask = df["user_id"].isin(picked).values
    return np.flatnonzero(mask)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """모델 입력으로 안전한 컬럼만 반환 (ID·라벨·원문 제외)."""
    drop = set(ID_COLUMNS) | {"fraud", "text", "date", "type_class"}
    return [c for c in df.columns if c not in drop]
