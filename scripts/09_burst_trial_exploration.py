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
- [2차 피드백, 2026-09-13] 평생평균 λ는 플랫폼 성장(2008년 1.7만건 → 2014년 18.1만건,
  약 10배) 때문에 오래된 상점일수록 초창기 한산했던 시절에 끌려 내려간다 → 2014년의
  평범한 주도 "몰림"으로 오판된다(오동진 지적). 반대로 "직전 90일"만 기준선으로 쓰면
  90일간 조용했던 상점은 기준선이 0에 가까워 리뷰 1건도 다시 유의해진다(1건비율 23%).
  → **최종 정의: 두 기준선(평생평균, 직전 90일) 모두에서 p<0.01인 경우만 채택**
    (한쪽의 약점을 다른 쪽이 보완하는 구조).
- [프로세스 제안, 오동진] 같은 정의를 각자 다시 계산하면 작은 불일치가 생긴다(예: 평점
  이탈형 표본 9,813 vs 10,175 — as-of 상점평균 계산 차이로 추정). 이 스크립트를
  **단일 라벨 소스**로 삼기 위해 `--write-labels`로 review_id별 최종 라벨 3컬럼
  (`is_burst`, `is_rating_deviation`, `is_burst_and_rating`)을 저장하는 기능을 추가했다.
- [기록, 민섭] "1건짜리"(90일간 조용하다 리뷰 1건, 평생평균 기준으로만 유의) 케이스는
  버스트형에서 제외하지만, "휴면 상점이 리뷰를 사서 활동을 재개하는" 패턴일 수 있어
  별도로 규모·사기율을 기록해둔다(향후 피처·유형 후보).

실행:
  python scripts/09_burst_trial_exploration.py              # 탐색 리포트만 출력
  python scripts/09_burst_trial_exploration.py --write-labels  # + 라벨 컬럼 저장
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[1]
REVIEWS = ROOT / "data" / "processed" / "reviews.parquet"
LABELS_OUT = ROOT / "data" / "processed" / "burst_labels.parquet"

WINDOW_DAYS = 7
BASELINE_DAYS = 90  # 직전 90일 기준선(최근 7일 제외)
RDEV_THRESHOLD = 1.5


def compute_time_signal(df: pd.DataFrame) -> pd.DataFrame:
    """상점별 관측/기대 비율 + 포아송 단측 p-value. 기준선 두 가지를 함께 계산한다:
    (1) 평생평균(as-of, 그 시점까지 전체 이력) (2) 직전 90일(최근 7일 제외 트레일링).
    """
    work = df.sort_values(["prod_id", "date", "review_id"]).reset_index(drop=True)
    out = []
    for pid, sub in work.groupby("prod_id", sort=False):
        dates = sub["date"].values.astype("datetime64[D]")
        n = len(dates)
        if n < 2:
            # 그 상점 리뷰가 1건뿐 — 평생평균/직전90일 모두 정의 불가.
            # 행 자체를 빠뜨리면(구 동작) burst_labels.parquet에서 조인 시
            # 결측이 되는 문제가 있었다(오동진 지적) — is_burst=False로
            # 명시적으로 평가 가능하도록 자리표시 행을 남긴다.
            out.append(
                pd.DataFrame(
                    {
                        "review_id": sub["review_id"].values,
                        "local_count": np.ones(n),
                        "expected_life": np.full(n, np.nan),
                        "expected_90": np.full(n, np.nan),
                        "base90_days": np.zeros(n),
                        "elapsed": np.zeros(n, dtype=int),
                    }
                )
            )
            continue
        first = dates[0]
        idx = np.arange(1, n + 1)
        elapsed = (dates - first).astype("timedelta64[D]").astype(int)
        elapsed_safe = np.maximum(elapsed, 1)
        asof_rate = idx / elapsed_safe  # 평생평균: 그 시점까지의 하루 평균 발생률

        j = 0
        local_count = np.empty(n)
        for i in range(n):
            while dates[i] - dates[j] > np.timedelta64(WINDOW_DAYS - 1, "D"):
                j += 1
            local_count[i] = i - j + 1  # 최근 WINDOW_DAYS일 트레일링 리뷰 수(자기 포함)

        # 직전 90일 기준선: [date-97, date-8] 구간 카운트 (최근 7일은 제외)
        k_lo = 0
        k_hi = 0
        base90_count = np.empty(n)
        for i in range(n):
            while k_hi < n and dates[k_hi] <= dates[i] - np.timedelta64(WINDOW_DAYS + 1, "D"):
                k_hi += 1
            while k_lo < k_hi and dates[i] - dates[k_lo] > np.timedelta64(
                BASELINE_DAYS + WINDOW_DAYS, "D"
            ):
                k_lo += 1
            base90_count[i] = k_hi - k_lo
        base90_days = np.clip(elapsed - WINDOW_DAYS, 0, BASELINE_DAYS)
        base90_rate = np.where(
            base90_days > 0, base90_count / np.maximum(base90_days, 1), np.nan
        )

        expected_life = asof_rate * WINDOW_DAYS
        expected_90 = base90_rate * WINDOW_DAYS
        out.append(
            pd.DataFrame(
                {
                    "review_id": sub["review_id"].values,
                    "local_count": local_count,
                    "expected_life": expected_life,
                    "expected_90": expected_90,
                    "base90_days": base90_days,
                    "elapsed": elapsed,
                }
            )
        )
    res = pd.concat(out, ignore_index=True)
    res["ratio"] = res["local_count"] / np.maximum(res["expected_life"], 1e-6)
    # 단측 검정: 관측치가 기대치보다 우연히 이 정도로 클 확률
    res["pval_life"] = poisson.sf(res["local_count"] - 1, res["expected_life"])
    res["pval_90"] = poisson.sf(res["local_count"] - 1, np.maximum(res["expected_90"], 1e-9))
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


MIN_PAST_REVIEWS = 20  # 오동진 제안 — 민감도 확인 결과(10/20/30건) 거의 무차이


def compute_rating_signal_v2(df: pd.DataFrame) -> pd.DataFrame:
    """방향별 평점 이탈(2026-09-13, 오동진 재정의).

    기존 is_rating_anom(RDEV_THRESHOLD=1.5, 절대·자기포함 평균)은 (a) 임계값 출처가
    불분명하고(버스트형 때와 같은 "인용 오기" 패턴) (b) 자기 자신이 평균에 포함돼
    이탈을 과소평가하고 (c) 방향(깎기/띄우기)을 구분하지 않아 서로 다른 두 현상을
    섞었다. 재정의:
      - 기준선 = 그 리뷰 이전(자기 제외) 가게 평균, 과거 리뷰 20건 미만이면 판정 불가(False)
      - dev = rating − 기준선 (부호 유지)
      - 임계값 = 2014년 이전 & eligible 리뷰의 dev 분포에서 하위/상위 5%(및 10%) —
        분석 대상 연도(2014)와 겹치지 않는 기간으로 미리 정해, 사기율을 보고 고르지 않음
      - 검증값(오동진·소륜 일치): 5% 임계 -2.24/+1.23, 2014년 10,402/9,230건
    """
    work = df.sort_values(["prod_id", "date", "review_id"]).reset_index(drop=True)
    work["cum_n"] = work.groupby("prod_id").cumcount()  # 자기 제외 과거 리뷰 수
    work["cum_sum"] = work.groupby("prod_id")["rating"].cumsum() - work["rating"]
    work["past_avg"] = work["cum_sum"] / work["cum_n"].replace(0, np.nan)
    work["eligible"] = work["cum_n"] >= MIN_PAST_REVIEWS
    work["dev"] = work["rating"] - work["past_avg"]

    calib = work[(work["year"] < 2014) & work["eligible"]]
    q05, q95 = calib["dev"].quantile([0.05, 0.95])
    q10, q90 = calib["dev"].quantile([0.10, 0.90])

    down5 = work["eligible"] & (work["dev"] <= q05)
    up5 = work["eligible"] & (work["dev"] >= q95)
    down10 = work["eligible"] & (work["dev"] <= q10)
    up10 = work["eligible"] & (work["dev"] >= q90)

    return pd.DataFrame(
        {
            "review_id": work["review_id"],
            "is_rating_down": down5.fillna(False).astype(bool),
            "is_rating_up": up5.fillna(False).astype(bool),
            "is_rating_deviation_v2": (down5 | up5).fillna(False).astype(bool),
            "is_rating_down_10": down10.fillna(False).astype(bool),
            "is_rating_up_10": up10.fillna(False).astype(bool),
            "is_rating_deviation_v2_10": (down10 | up10).fillna(False).astype(bool),
        }
    )


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


def build_labels(res: pd.DataFrame, rating_v2: pd.DataFrame) -> pd.DataFrame:
    """review_id별 최종 라벨 컬럼. 전체 이력(연도 무관) 기준 — 라벨은 전체 데이터에서
    계산한다는 프로젝트 원칙(experiment_matrix.md)을 따른다. elapsed==0(그 상점의
    첫 리뷰)은 평생평균 기준선이 정의상 불안정해 버스트 판정 대상에서 제외한다.
    """
    valid90 = res["base90_days"] >= 30
    evaluable = res["elapsed"] >= 1
    is_burst = evaluable & valid90 & (res["pval_life"] < 0.01) & (res["pval_90"] < 0.01)
    is_rating = res["is_rating_anom"]
    out = pd.DataFrame(
        {
            "review_id": res["review_id"],
            "is_burst": is_burst.fillna(False).astype(bool),
            "is_rating_deviation": is_rating.astype(bool),  # 구버전, 보존(오동진 지시)
            "is_burst_and_rating": (is_burst.fillna(False) & is_rating).astype(bool),
        }
    )
    return out.merge(rating_v2, on="review_id", how="left")


def report_dormant_case(res: pd.DataFrame) -> None:
    """'1건짜리'(직전90일 기준으로만 유의 — 평생평균으로는 그 상점이 원래 활발해서
    유의하지 않음) — "평소엔 정상 운영되던 상점이 최근 90일 조용하다가 리뷰 1건으로
    재개"되는 휴면 상점 리뷰 재개 후보. 버스트형(교집합)에서는 제외하지만 규모·사기율을
    별도로 기록한다(민섭 제안, §3.4).
    """
    d14 = res[(res["year"] == 2014) & (res["elapsed"] >= 1)]
    ninety_only = ((d14["pval_90"] < 0.01) & (d14["base90_days"] >= 30)) & ~(
        d14["pval_life"] < 0.01
    )
    dormant = ninety_only & (d14["local_count"] == 1)
    n = int(dormant.sum())
    if n == 0:
        print("휴면 상점 재개 후보: n=0")
        return
    fr = d14.loc[dormant, "fraud"].mean()
    print(f"휴면 상점 재개 후보(1건, 직전90일만 유의): n={n:,}  사기율={fr*100:.2f}%")
    for grp_name, grp_mask in [("저활동", d14["is_new"]), ("비저활동", ~d14["is_new"])]:
        sel = dormant & grp_mask
        ns = int(sel.sum())
        if ns == 0:
            continue
        print(f"  {grp_name}: n={ns:,}  사기율={d14.loc[sel, 'fraud'].mean()*100:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-labels",
        action="store_true",
        help=f"최종 라벨 3컬럼을 {LABELS_OUT} 에 저장(단일 소스, 오동진 제안)",
    )
    args = parser.parse_args()

    df = pd.read_parquet(
        REVIEWS,
        columns=["review_id", "prod_id", "user_id", "date", "rating", "fraud", "year"],
    )
    df["date"] = pd.to_datetime(df["date"])

    time_sig = compute_time_signal(df)
    rating_sig = compute_rating_signal(df)
    rating_v2 = compute_rating_signal_v2(df)
    is_new = compute_is_new(df)

    res = time_sig.merge(df[["review_id", "fraud", "year", "date"]], on="review_id")
    res = res.merge(rating_sig, on="review_id")
    res = res.merge(rating_v2, on="review_id")

    is_new_map = pd.Series(is_new, index=df["review_id"].values)
    res["is_new"] = res["review_id"].map(is_new_map)

    if args.write_labels:
        labels = build_labels(res, rating_v2)
        labels.to_parquet(LABELS_OUT, index=False)
        print(f"라벨 저장 완료: {LABELS_OUT} ({len(labels):,}행)")
        print(
            f"  is_burst={labels['is_burst'].sum():,}  "
            f"is_rating_deviation={labels['is_rating_deviation'].sum():,}  "
            f"is_burst_and_rating={labels['is_burst_and_rating'].sum():,}"
        )
        print(
            f"  is_rating_down={labels['is_rating_down'].sum():,}  "
            f"is_rating_up={labels['is_rating_up'].sum():,}  "
            f"is_rating_deviation_v2={labels['is_rating_deviation_v2'].sum():,}"
        )
        print(
            f"  (10%) is_rating_down_10={labels['is_rating_down_10'].sum():,}  "
            f"is_rating_up_10={labels['is_rating_up_10'].sum():,}  "
            f"is_rating_deviation_v2_10={labels['is_rating_deviation_v2_10'].sum():,}"
        )
        print()

    print("=== 평점 이탈형 v2 검증 (오동진 확인값과 대조, 2014년) ===")
    d14_check = res[res["year"] == 2014]
    print(
        f"is_rating_down n={d14_check['is_rating_down'].sum():,}  "
        f"is_rating_up n={d14_check['is_rating_up'].sum():,}  "
        f"(확인값: 10,402 / 9,230)"
    )
    print()

    d14 = res[(res["year"] == 2014) & (res["elapsed"] >= 1)].copy()
    valid90 = d14["base90_days"] >= 30

    print("=== 플랫폼 성장 확인 (연도별 리뷰 수) ===")
    print(df.groupby("year").size().to_string())

    print()
    print("=== 안전장치 비교: 평생평균 기준선 (1차 피드백) ===")
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
    flat_report("수정: 포아송 p<0.05", d14["pval_life"] < 0.05)
    flat_report("수정: 포아송 p<0.01", d14["pval_life"] < 0.01)
    flat_report("수정: 포아송 p<0.001", d14["pval_life"] < 0.001)

    print()
    print("=== 그룹별 in-group lift, 평생평균 기준 (섞으면 심슨의 역설로 왜곡됨) ===")
    for label, alpha in [("시간형 p<0.05", 0.05), ("시간형 p<0.01", 0.01), ("시간형 p<0.001", 0.001)]:
        print(f"-- {label} --")
        ingroup_report(d14, label, d14["pval_life"] < alpha)

    print()
    ingroup_report(d14, "평점형", d14["is_rating_anom"])
    print()
    ingroup_report(d14, "AND(시간 p<0.01 + 평점형)", (d14["pval_life"] < 0.01) & d14["is_rating_anom"])

    # 평점 이탈형(v2)은 시간 신호와 무관하므로, elapsed>=1 제약이 없는 2014년
    # 전체를 그룹 기준선으로 쓴다(오동진 매트릭스 관례와 일치, 2026-09-13 확인).
    d14_all = res[res["year"] == 2014].copy()
    print()
    print("=== 평점 이탈형 v2, 그룹별 in-group lift (기준선: 2014년 그룹 전체) ===")
    ingroup_report(d14_all, "깎기(is_rating_down, 5%)", d14_all["is_rating_down"])
    ingroup_report(d14_all, "띄우기(is_rating_up, 5%)", d14_all["is_rating_up"])
    ingroup_report(d14_all, "합집합(is_rating_deviation_v2, 5%)", d14_all["is_rating_deviation_v2"])

    print()
    print("=== 2차 피드백: 플랫폼 성장 편향 확인 (평생평균 버스트 vs 직전90일 실측) ===")
    life_burst = d14["pval_life"] < 0.01
    ratio_vs_90 = d14.loc[life_burst, "expected_90"] / d14.loc[life_burst, "expected_life"]
    print(f"직전90일/평생평균 비율 중앙값: {ratio_vs_90.median():.2f}")
    print(f"그 중 1.5배 이상 비율: {(ratio_vs_90 >= 1.5).mean()*100:.1f}%")
    new_def = (d14["pval_90"] < 0.01) & valid90
    overlap_rate = (life_burst & new_def).sum() / life_burst.sum()
    print(f"직전90일 기준으로 바꿨을 때도 남는 비율: {overlap_rate*100:.1f}%")
    print(f"직전90일 기준 1건비율: {(d14.loc[new_def, 'local_count']==1).mean()*100:.1f}%")

    print()
    print("=== 최종 정의: 두 기준선 모두 p<0.01 (교집합) ===")
    both = (d14["pval_life"] < 0.01) & (d14["pval_90"] < 0.01) & valid90
    print(f"n={both.sum():,}  1건비율={(d14.loc[both,'local_count']==1).mean()*100:.1f}%")
    ingroup_report(d14, "교집합(최종)", both)

    dec = d14["date"].dt.month == 12
    burst_nonnew = both & (~d14["is_new"])
    print(
        f"비저활동 교집합 12월 test: n={(burst_nonnew & dec).sum():,} "
        f"사기={d14.loc[burst_nonnew & dec, 'fraud'].sum():.0f}"
    )

    print()
    print("=== 기록: 휴면 상점 재개 후보 (버스트형에서는 제외, 민섭 제안) ===")
    report_dormant_case(res)


if __name__ == "__main__":
    main()
