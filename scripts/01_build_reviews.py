"""
01_build_reviews.py — YelpZip 원본 → reviews.parquet

이 스크립트가 하는 일:
  1) yelpzip.csv 의 원래 행 순서를 review_id(0..N-1)로 고정한다.
     이후 만들어지는 모든 산출물(임베딩 .npy, 피처 테이블, 그래프)은
     이 순서를 기준으로 정렬된다. 절대 재정렬하지 말 것.
  2) label(-1/1) -> fraud(1/0) 로 변환. tag 컬럼은 label 과 100% 중복이라 제거.
  3) 저활동형(구 "신규계정형") 라벨을 "그 리뷰 작성 시점까지의 누적 리뷰 수(as-of)"
     기준으로 계산한다(9/9 지도교수 피드백). 표본 안에서 계산하면 리뷰어의
     리뷰 수가 과소집계되어 라벨이 오염되고(업체 400개 표본 실험에서
     신규계정형의 39.8%가 거짓으로 확인됨), 반대로 유저 전체(과거+미래)
     리뷰 수로 계산하면 미래 시점 정보가 새어 들어간다(as-of 누수).
     이 as-of 카운트는 전체 10년 이력을 기준으로 계산한다 — 분석 대상
     리뷰를 2014년으로만 필터링하더라도(Phase 1-C), 카운트 자체는 필터링
     이전의 전체 이력을 본다(2026-09-11 팀 결정: 옵션 B, 카운트는 전체
     이력·분석 대상만 2014년).

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

# 저활동형(구 "신규계정형"): 리뷰 작성 시점까지 누적 리뷰 수(as-of) <= NEW_MAX
# NEW_MAX = 1 은 "작성자의 첫 리뷰(이전 이력 0건)" = cold-start 표준 정의
# (Wang et al., ACL 2017; Xiang et al., 2022). 2026-09-11 팀 결정.
# 작성자 리뷰 수 분포가 극도로 치우쳐(작성자의 65.4%가 1건) 하위 10~50% 분위수가
# 모두 1건이므로, 분위수 기반 정의와도 일치한다.
NEW_MAX = 1
# 민감도 분석용: 이전 이력 1건 이하(as-of <= 2)
NEW_MAX_SENS = 2
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
    df["year"] = df["date"].dt.year.astype("int32")
    return df


def add_new_account_type(df: pd.DataFrame) -> pd.DataFrame:
    """저활동형(구 "신규계정형"): 리뷰 작성 시점까지의 누적 리뷰 수(as-of) 기준.

    n_reviews_user(전체 이력 총합, 기존 유지 — 03_build_features.py의
    singleton 피처가 이 컬럼을 그대로 사용하므로 하위호환 위해 보존)와
    n_reviews_user_asof(그 리뷰 작성 시점까지 누적, 미래 리뷰 제외)를 구분한다.
    type_new(저활동형 라벨)는 반드시 후자 기준이어야 미래 누수가 없다.
    """
    n_total = df.groupby("user_id")["review_id"].transform("size")
    df["n_reviews_user"] = n_total.astype("int32")

    order = df.sort_values(["user_id", "date", "review_id"]).index
    asof = df.loc[order].groupby("user_id").cumcount().to_numpy() + 1
    df["n_reviews_user_asof"] = pd.Series(asof, index=order).reindex(df.index).astype("int32")
    df["type_new"] = (df["n_reviews_user_asof"] <= NEW_MAX).astype("int8")
    # 민감도 분석용 (주 분석은 type_new)
    df["type_new_le2"] = (df["n_reviews_user_asof"] <= NEW_MAX_SENS).astype("int8")
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

    df14 = df[df.year == 2014].copy()
    print(f"\n[2014년 분석 범위] {len(df14):,}건 ({100 * len(df14) / len(df):.1f}%)")
    t14 = df14.groupby("type_class").agg(
        표본수=("fraud", "size"), 사기수=("fraud", "sum"), 사기율=("fraud", "mean")
    )
    t14["비중%"] = (100 * t14["표본수"] / len(df14)).round(1)
    t14["사기율"] = (100 * t14["사기율"]).round(1)
    print(t14[["표본수", "비중%", "사기수", "사기율"]].to_string())
    # R-U-R 그래프는 2014년 리뷰끼리만 엣지를 만드므로, 싱글턴 여부는
    # (as-of 누적이 아니라) 2014년 내 해당 유저의 리뷰 수로 판단해야 한다.
    n_user_2014 = df14.groupby("user_id")["review_id"].transform("size")
    singleton_2014 = n_user_2014 == 1
    is_low_activity = df14.type_new == 1
    print(
        f"저활동형 중 2014년-그래프 싱글턴(R-U-R 엣지 없음) 비율: "
        f"{100 * (singleton_2014 & is_low_activity).sum() / is_low_activity.sum():.1f}%"
    )


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
