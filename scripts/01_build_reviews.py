"""
01_build_reviews.py — YelpZip 원본 → reviews.parquet

이 스크립트가 하는 일:
  1) yelpzip.csv 의 원래 행 순서를 review_id(0..N-1)로 고정한다.
     이후 만들어지는 모든 산출물(임베딩 .npy, 피처 테이블, 그래프)은
     이 순서를 기준으로 정렬된다. 절대 재정렬하지 말 것.
  2) label(-1/1) -> fraud(1/0) 로 변환. tag 컬럼은 label 과 100% 중복이라 제거.
  3) 사기 유형 라벨을 "전체 데이터 기준"으로 계산한다.
     표본 안에서 계산하면 리뷰어의 리뷰 수가 과소집계되어 라벨이 오염된다
     (업체 400개 표본 실험에서 신규계정형의 39.8%가 거짓으로 확인됨).

주의: type_burst 는 전역 고정 임계값 기반의 임시 라벨이다.
      지도교수 피드백에 따른 상점별 KDE 적응형 정의는 별도 스크립트에서
      type_burst_kde 컬럼으로 추가한다.

출력: data/processed/reviews.parquet
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "yelpzip.csv"
OUT = ROOT / "data" / "processed" / "reviews.parquet"

# 신규계정형: 리뷰어 총 리뷰 수 <= NEW_MAX
NEW_MAX = 3
# 버스트형(임시): 같은 업체에 BURST_WINDOW_DAYS 내 BURST_MIN_K 건 이상
BURST_WINDOW_DAYS = 7
BURST_MIN_K = 5


def load() -> pd.DataFrame:
    df = pd.read_csv(SRC, index_col=0)
    df = df.reset_index(drop=True)
    df.insert(0, "review_id", np.arange(len(df), dtype=np.int64))
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rating"] = df["rating"].astype("float32")
    df["user_id"] = df["user_id"].astype("int64")
    df["prod_id"] = df["prod_id"].astype("int64")
    df["text"] = df["text"].fillna("")
    # label: -1 = 사기(filtered), 1 = 정상(recommended) -> fraud 1/0
    df["fraud"] = (df["label"] == -1).astype("int8")
    df = df.drop(columns=["label", "tag"])
    return df


def add_new_account_type(df: pd.DataFrame) -> pd.DataFrame:
    """신규계정형: 작성자의 전체 리뷰 수 기준."""
    n = df.groupby("user_id")["review_id"].transform("size")
    df["n_reviews_user"] = n.astype("int32")
    df["type_new"] = (n <= NEW_MAX).astype("int8")
    return df


def add_burst_type(df: pd.DataFrame) -> pd.DataFrame:
    """버스트형(임시): 업체별 슬라이딩 윈도우 안에 K건 이상이면 그 구간 전체를 표시."""
    order = np.lexsort((df["date"].values.astype("datetime64[ns]"), df["prod_id"].values))
    pid = df["prod_id"].values[order]
    dts = df["date"].values[order].astype("datetime64[ns]")
    flag = np.zeros(len(df), dtype=bool)
    win = np.timedelta64(BURST_WINDOW_DAYS, "D")

    bounds = np.r_[0, np.flatnonzero(pid[1:] != pid[:-1]) + 1, len(df)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        d = dts[a:b]
        j = 0
        for i in range(b - a):
            while d[i] - d[j] > win:
                j += 1
            if i - j + 1 >= BURST_MIN_K:
                flag[a + j: a + i + 1] = True

    out = np.zeros(len(df), dtype=bool)
    out[order] = flag
    df["type_burst"] = out.astype("int8")
    return df


def add_type_class(df: pd.DataFrame) -> pd.DataFrame:
    """신규/버스트 조합을 4개 층으로 정리 (혼합 = 신규 AND 버스트)."""
    cond = [
        (df.type_new == 1) & (df.type_burst == 0),
        (df.type_new == 0) & (df.type_burst == 1),
        (df.type_new == 1) & (df.type_burst == 1),
    ]
    df["type_class"] = np.select(cond, ["new", "burst", "mixed"], default="general")
    df["type_mixed"] = ((df.type_new == 1) & (df.type_burst == 1)).astype("int8")
    return df


def report(df: pd.DataFrame) -> None:
    print(f"리뷰 {len(df):,} | 리뷰어 {df.user_id.nunique():,} | 업체 {df.prod_id.nunique():,}")
    print(f"사기율 {100 * df.fraud.mean():.2f}%  |  날짜 {df.date.min().date()} ~ {df.date.max().date()}")
    print()
    t = df.groupby("type_class").agg(
        표본수=("fraud", "size"), 사기수=("fraud", "sum"), 사기율=("fraud", "mean")
    )
    t["비중%"] = (100 * t["표본수"] / len(df)).round(1)
    t["사기율"] = (100 * t["사기율"]).round(1)
    print(t[["표본수", "비중%", "사기수", "사기율"]].to_string())


def main() -> int:
    if not SRC.exists():
        print(f"입력 파일 없음: {SRC}", file=sys.stderr)
        return 1

    df = load()
    df = normalize(df)
    df = add_new_account_type(df)
    df = add_burst_type(df)
    df = add_type_class(df)

    assert df.review_id.is_monotonic_increasing, "review_id 순서가 깨졌습니다"
    assert len(df) == df.review_id.nunique(), "review_id 중복"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    report(df)
    print(f"\n저장: {OUT}  ({OUT.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
