"""
06_build_graphs_2014.py — 2014년 리뷰 노드 + 과거 방향 관계 그래프 3종

9/9 미팅 + 2026-09-11 팀 결정:
  노드   2014년 리뷰 (180,659건)
  엣지   2014년 리뷰끼리만, 과거 방향만 (먼저 쓴 리뷰 → 나중 리뷰)
  관계   R-U-R(같은 작성자) / R-S-R(같은 업체+평점) / R-T-R(같은 업체+달)

출력 (data/processed/graph_2014/):
  nodes.parquet     로컬 노드 번호 → review_id, user_id, prod_id, date, fraud, 유형 라벨
  rur.npz / rsr.npz / rtr.npz   과거 방향 인접행렬 (A[i,j]=1 ⇔ j 가 i 보다 먼저)

로컬 노드 i 의 전역 review_id 는 nodes.parquet 의 i 번째 행. 임베딩·피처는
`emb[nodes.review_id]` 처럼 전역 review_id 로 슬라이스해 쓰면 된다.

7가지 조합은 학습 시 `src.features.structural.temporal_graph.combine` 으로 만든다.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loaders import load_reviews  # noqa: E402
from src.features.structural.temporal_graph import RELATIONS, past_only_adjacency  # noqa: E402

YEAR = 2014
OUT_DIR = ROOT / "data" / "processed" / f"graph_{YEAR}"


def main() -> int:
    t0 = time.time()
    df = load_reviews(columns=["review_id", "user_id", "prod_id", "rating", "date", "fraud",
                               "year", "n_reviews_user_asof", "type_new", "type_new_le2"])
    nodes = df[df["year"] == YEAR].reset_index(drop=True)   # 로컬 0..n-1, review_id 오름차순 유지
    nodes["month"] = nodes["date"].dt.to_period("M").astype(str)
    n = len(nodes)
    print(f"{YEAR}년 노드 {n:,}건 | 사기율 {100 * nodes.fraud.mean():.1f}%")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    adjs = {}
    for name, keys in RELATIONS.items():
        A = past_only_adjacency(nodes, keys)
        adjs[name] = A
        sp.save_npz(OUT_DIR / f"{name}.npz", A)
        indeg = np.asarray(A.sum(axis=1)).ravel()
        print(f"  {name.upper()}  엣지 {A.nnz:>10,} | 과거 이웃 0개 노드 {100 * (indeg == 0).mean():5.1f}% "
              f"| {time.time() - t0:.0f}초")

    # ── 정합성 검사 ─────────────────────────────────────────────────────
    # (1) 과거 방향: 모든 엣지의 보내는 쪽이 받는 쪽보다 시간상 앞서야 한다
    order_key = nodes["date"].to_numpy("datetime64[D]").astype(np.int64) * 10**7 + nodes["review_id"].to_numpy()
    for name, A in adjs.items():
        coo = A.tocoo()
        assert (order_key[coo.col] < order_key[coo.row]).all(), f"{name}: 미래 방향 엣지 존재"
    # (2) 첫 리뷰(type_new=1)는 같은 작성자의 과거 리뷰가 없으므로 R-U-R 이웃이 0이어야 한다
    rur_in = np.asarray(adjs["rur"].sum(axis=1)).ravel()
    assert (rur_in[nodes["type_new"].to_numpy() == 1] == 0).all(), "첫 리뷰인데 R-U-R 과거 이웃이 있음"
    print("정합성 검사 통과: 모든 엣지가 과거 방향 / 첫 리뷰의 R-U-R 이웃 0개")

    keep = ["review_id", "user_id", "prod_id", "rating", "date", "fraud",
            "n_reviews_user_asof", "type_new", "type_new_le2"]
    nodes[keep].to_parquet(OUT_DIR / "nodes.parquet", index=False)

    # ── 유형별 요약 ────────────────────────────────────────────────────
    print()
    print(f"{'층':<16} {'표본':>9} {'사기율':>7} | " + " | ".join(f"{r.upper()} 고립" for r in RELATIONS))
    for label, mask in [("저활동 (<=1)", nodes.type_new == 1), ("비저활동", nodes.type_new == 0),
                        ("저활동 (<=2)", nodes.type_new_le2 == 1), ("비저활동 (>2)", nodes.type_new_le2 == 0)]:
        m = mask.to_numpy()
        iso = [100 * (np.asarray(adjs[r].sum(axis=1)).ravel()[m] == 0).mean() for r in RELATIONS]
        print(f"{label:<16} {m.sum():>9,} {100 * nodes.fraud[m].mean():>6.1f}% | "
              + " | ".join(f"{v:7.1f}%" for v in iso))

    print(f"\n저장: {OUT_DIR}  |  총 {time.time() - t0:.0f}초")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
