"""
09_burst_trial_exploration.py — 버스트형 정의 실증 트라이얼 (탐색적, 파이프라인 미편입)

docs/버스트형_정의_검토_소륜.md §3의 근거 코드. Fei et al.(2013)의 "상점 자기평균
대비 초과" 아이디어를 관측/기대 비율로 근사하고, 오동진 피드백(2026-09-13)을 반영해
포아송 검정 기반으로 재정의했다.

[배경]
- 원래 근사(ratio = 최근 7일 리뷰수 / 기대치)는 평소 리뷰가 거의 없는 조용한 상점에서
  리뷰 1건만 와도 ratio가 크게 튀는 문제가 있었다(오동진 지적 — 시간만 버스트의 43.6%가
  local_count==1). Fei 원문은 "평균 이하 구간 버림 + 단일 리뷰 구간 버림" 안전장치가
  있는데 근사 과정에서 빠졌다.
- 9/9 요건("사기율 상승=근거 금지")을 지키려면 임계값을 사기율을 보고 고르면 안 된다.
  → 포아송 검정(p-value)은 라벨과 무관한 통계적 관행(p<0.01 등)으로 컷오프를 정할 수
    있고, 동시에 저건수 문제도 자연히 해결한다(기대치가 낮으면 1건으로는 유의하지 않음).
- 저활동형(첫 리뷰)과 48.1% 겹친다(오동진 지적). 이 스크립트는 그룹별(저활동/비저활동)
  자체 기준선 대비 lift를 따로 계산한다 — 섞어서 계산하면 심슨의 역설처럼 왜곡된다
  (저활동형 기준선 20.8%, 비저활동형 6.3%로 8배 차이나기 때문).

실행: python scripts/09_burst_trial_exploration.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[1]
REVIEWS = ROOT / "data" / "processed" / "reviews.parquet"

WINDOW_DAYS = 7
RDEV_THRESHOLD = 1.5


def compute_time_signal(df: pd.DataFrame) -> pd.DataFrame:
    """상점별 as-of 관측/기대 비율 + 포아송 단측 p-value."""
    work = df.sort_values(["prod_id", "date", "review_id"]).reset_index(drop=True)
    out = []
    for pid, sub in work.groupby("prod_id", sort=False):
        dates = sub["date"].values.astype("datetime64[D]")
        n = len(dates)
        if n < 2:
            continue
        first = dates[0]
        idx = np.arange(1, n + 1)
        elapsed = (dates - first).astype("timedelta64[D]").astype(int)
        elapsed_safe = np.maximum(elapsed, 1)
        asof_rate = idx / elapsed_safe  # 그 시점까지의 하루 평균 발생률(자기 자신 포함)

        j = 0
        local_count = np.empty(n)
        for i in range(n):
            while dates[i] - dates[j] > np.timedelta64(WINDOW_DAYS - 1, "D"):
                j += 1
            local_count[i] = i - j + 1  # 최근 WINDOW_DAYS일 트레일링 리뷰 수(자기 포함)

        expected = asof_rate * WINDOW_DAYS
        out.append(
            pd.DataFrame(
                {
                    "review_id": sub["review_id"].values,
                    "local_count": local_count,
                    "expected": expected,
                    "elapsed": elapsed,
                }
            )
        )
    res = pd.concat(out, ignore_index=True)
    res["ratio"] = res["local_count"] / np.maximum(res["expected"], 1e-6)
    # 단측 검정: 관측치가 기대치보다 우연히 이 정도로 클 확률
    res["pval"] = poisson.sf(res["local_count"] - 1, res["expected"])
    return res


def compute_rating_signal(df: pd.DataFrame) -> pd.DataFrame:
    """상점별 as-of 평균 대비 극단평점 + 이탈."""
    work = df.sort_values(["prod_id", "date", "review_id"]).reset_index(drop=True)
    work["cum_n"] = work.groupby("prod_id").cumcount() + 1
    work["cum_sum"] = work.groupby("prod_id")["rating"].cumsum()
    work["asof_avg"] = work["cum_sum"] / work["cum_n"]
    work["rating_dev"] = (work["rating"] - work["asof_avg"]).abs()
    work["is_rating_anom"] = work["rating"].isin([1.0, 5.0]) & (
        work["rating_dev"] > RDEV_THRESHOLD
    )
    return work[["review_id", "is_rating_anom"]]


def compute_is_new(df: pd.DataFrame) -> pd.Series:
    """저활동형(as-of 누적 리뷰수<=1, 즉 작성자의 첫 리뷰) — 겹침 확인용."""
    work = df.sort_values(["user_id", "date", "review_id"])
    work["user_asof_count"] = work.groupby("user_id").cumcount() + 1
    is_new = work["user_asof_count"] <= 1
    return is_new.set_axis(work["review_id"]).reindex(df["review_id"]).values


def ingroup_report(d14: pd.DataFrame, name: str, mask: pd.Series) -> None:
    """그룹(저활동/비저활동)별 자체 기준선 대비 lift — 섞어서 계산하면 왜곡됨."""
    for grp_name, grp_mask in [("저활동", d14.is_new), ("비저활동", ~d14.is_new)]:
        base = d14.loc[grp_mask, "fraud"].mean()
        sel = mask & grp_mask
        n = int(sel.sum())
        if n == 0:
            print(f"  {name} / {grp_name}: n=0")
            continue
        fr = d14.loc[sel, "fraud"].mean()
        print(
            f"  {name} / {grp_name}: n={n:>6,}  사기율={fr*100:5.2f}%  "
            f"(그룹 기준선 {base*100:.2f}%)  lift={fr/base:.2f}x"
        )


def main() -> None:
    df = pd.read_parquet(
        REVIEWS,
        columns=["review_id", "prod_id", "user_id", "date", "rating", "fraud", "year"],
    )
    df["date"] = pd.to_datetime(df["date"])

    time_sig = compute_time_signal(df)
    rating_sig = compute_rating_signal(df)
    is_new = compute_is_new(df)

    res = time_sig.merge(df[["review_id", "fraud", "year"]], on="review_id")
    res = res.merge(rating_sig, on="review_id")

    is_new_map = pd.Series(is_new, index=df["review_id"].values)
    res["is_new"] = res["review_id"].map(is_new_map)

    d14 = res[(res["year"] == 2014) & (res["elapsed"] >= 1)].copy()

    print("=== 안전장치 비교 (2026-09-13 오동진 피드백 검증) ===")
    overall = d14["fraud"].mean()

    def flat_report(name, mask):
        n = int(mask.sum())
        if n == 0:
            print(f"{name}: n=0")
            return
        fr = d14.loc[mask, "fraud"].mean()
        singleton = (d14.loc[mask, "local_count"] == 1).mean()
        lt3 = (d14.loc[mask, "local_count"] < 3).mean()
        overlap = d14.loc[mask, "is_new"].mean()
        print(
            f"{name:32s} n={n:>7,} 사기율={fr*100:5.2f}% lift={fr/overall:.2f}x "
            f"1건비율={singleton*100:5.1f}% <3건비율={lt3*100:5.1f}% 저활동겹침={overlap*100:5.1f}%"
        )

    flat_report("기존: ratio>=5x (안전장치 없음)", d14["ratio"] >= 5)
    flat_report("수정: ratio>=5x AND count>=3", (d14["ratio"] >= 5) & (d14["local_count"] >= 3))
    flat_report("수정: 포아송 p<0.05", d14["pval"] < 0.05)
    flat_report("수정: 포아송 p<0.01", d14["pval"] < 0.01)
    flat_report("수정: 포아송 p<0.001", d14["pval"] < 0.001)

    print()
    print("=== 그룹별 in-group lift (섞으면 심슨의 역설로 왜곡됨) ===")
    for label, alpha in [("시간형 p<0.05", 0.05), ("시간형 p<0.01", 0.01), ("시간형 p<0.001", 0.001)]:
        print(f"-- {label} --")
        ingroup_report(d14, label, d14["pval"] < alpha)

    print()
    ingroup_report(d14, "평점형", d14["is_rating_anom"])
    print()
    ingroup_report(d14, "AND(시간 p<0.01 + 평점형)", (d14["pval"] < 0.01) & d14["is_rating_anom"])


if __name__ == "__main__":
    main()
