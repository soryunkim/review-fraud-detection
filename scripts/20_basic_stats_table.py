"""
20_basic_stats_table.py — 표 1 기초 데이터 통계 (김민섭, 2026-10-15)

10/7 미팅 지시: "데이터 수집 부분이 너무 빈약함 → 표 1(기초 데이터 통계)을 만들 것.
본문에 '몇 건, 몇 건'을 늘어놓지 말고 표 하나로. Yelp가 주는 정보 중 중요한 것만
추려서 넣을 것."

→ 따라서 이 표에는 **Yelp 원본이 제공하는 속성**(리뷰·작성자·상점 식별자, 평점,
작성일, 본문, 필터 라벨)에서 바로 나오는 값만 넣는다. 우리가 만든 피처나 그래프
엣지 수는 넣지 않는다(각각 3.2절 본문 소관).

실행:
  python scripts/20_basic_stats_table.py            # 표 + 근거 수치 출력
  python scripts/20_basic_stats_table.py --docx     # 18 에 붙여 넣을 코드 조각 출력
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REVIEWS = ROOT / "data" / "processed" / "reviews.parquet"
NODES = ROOT / "data" / "processed" / "graph_2014" / "nodes.parquet"


def fmt_int(x) -> str:
    return f"{int(x):,}"


def summarize(x: pd.DataFrame) -> dict:
    """Yelp 원본 속성에서 바로 나오는 값만."""
    return {
        "리뷰 수": fmt_int(len(x)),
        "작성자 수": fmt_int(x["user_id"].nunique()),
        "상점 수": fmt_int(x["prod_id"].nunique()),
        "기간": f"{x.date.min():%Y.%m}~{x.date.max():%Y.%m}",
        "걸러진 리뷰": f"{fmt_int(x.fraud.sum())} ({x.fraud.mean()*100:.1f}%)",
        "평균 평점": f"{x.rating.mean():.2f}",
        "4~5점 비율": f"{(x.rating >= 4).mean()*100:.1f}%",
        "리뷰 길이(중앙값)": f"{int(x['len'].median()):,}자",
    }


def build() -> tuple[list[list[str]], dict]:
    rev = pd.read_parquet(REVIEWS)
    rev["len"] = rev["text"].str.len()
    nodes = pd.read_parquet(NODES)
    cur = rev[rev.review_id.isin(set(nodes.review_id))].copy()

    # 이력 미보유/보유 — 최초 관측일(동률 재분류, 3.1절) 기준
    sdf = nodes.set_index("review_id")["type_new_samedayfix"]
    cur["sdf"] = cur.review_id.map(sdf)
    assert cur["sdf"].notna().all(), "nodes.parquet 에 type_new_samedayfix 가 없습니다 (01 재실행)"

    a, b = summarize(rev), summarize(cur)
    rows = [["항목", "YelpZip 전체", "분석 대상(2014년)"]]
    rows += [[k, a[k], b[k]] for k in a]

    grp = {}
    for v, name in ((1, "이력 미보유형"), (0, "이력 보유형")):
        g = cur[cur.sdf == v]
        grp[name] = (len(g), g.fraud.mean())
        rows.append([f"  {name}", "–", f"{fmt_int(len(g))} ({g.fraud.mean()*100:.1f}%)"])

    extra = {
        "작성자당 리뷰(전체/2014, 중앙값)": (
            f"{rev.groupby('user_id').size().median():.0f} / {cur.groupby('user_id').size().median():.0f}"),
        "상점당 리뷰(전체/2014, 중앙값)": (
            f"{rev.groupby('prod_id').size().median():.0f} / {cur.groupby('prod_id').size().median():.0f}"),
        "2014 월별 건수": (
            f"최소 {cur.groupby(cur.date.dt.month).size().min():,} / "
            f"최대 {cur.groupby(cur.date.dt.month).size().max():,}"),
        "미보유형 작성자 수": fmt_int(cur[cur.sdf == 1].user_id.nunique()),
        "보유형 작성자 수": fmt_int(cur[cur.sdf == 0].user_id.nunique()),
    }
    return rows, extra


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx", action="store_true", help="18 에 붙여 넣을 table(...) 코드 조각 출력")
    args = ap.parse_args()

    rows, extra = build()
    w = [max(len(r[i]) for r in rows) for i in range(3)]
    print("표 1  기초 데이터 통계\n")
    for i, r in enumerate(rows):
        print("  " + " | ".join(c.ljust(w[j]) for j, c in enumerate(r)))
        if i == 0:
            print("  " + "-+-".join("-" * x for x in w))

    print("\n[표에 넣지 않은 참고 수치 — 본문·질의응답용]")
    for k, v in extra.items():
        print(f"  {k}: {v}")
    print("\n  ※ 미보유형·보유형의 작성자 수를 더하면 2014년 작성자 수보다 많다. 한 작성자의")
    print("     최초 관측일 리뷰는 미보유형, 이후 리뷰는 보유형으로 들어가 작성자가 두 집단에")
    print("     모두 나타나기 때문이다. 표에는 리뷰 수만 싣는다(리뷰 단위로는 배타적).")

    if args.docx:
        print("\n# --- scripts/18_build_paper_docx.py 에 붙여 넣기 ---")
        print('parts.append(caption("표 1  기초 데이터 통계"))')
        print("parts.append(table([")
        for r in rows:
            print("    [" + ", ".join(f'"{c}"' for c in r) + "],")
        print("], [1560, 1560, 1780]))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
