"""
11_compare_cell_deltas.py — 두 칸의 그래프 기여(Δ = GNN − MLP) 차이에 대한 신뢰구간 (오동진, 2026-09-13)

"칸 X 의 그래프 기여가 칸 Y 보다 크다"를 주장하려면 두 Δ 각각의 신뢰구간이 아니라
Δ_X − Δ_Y 의 신뢰구간이 필요하다. 두 칸은 test 리뷰가 겹치지 않으므로(서로 다른 2×2 칸),
칸마다 따로 test 리뷰와 seed 를 복원추출해 Δ 를 만들고 그 차이를 모은다.
점수 파일 규칙과 계산 방식은 10_paired_bootstrap.py 와 같다.

사용 예
  python scripts/11_compare_cell_deltas.py --name normal_vs_ratingdev_rolling \
      --x-a "data/processed/scores/gnn/high_rur_sage_full_rolling_is_rating_deviation_v2-not*.parquet" \
      --x-b "data/processed/scores/mlp/high_all_seed*_rolling_is_rating_deviation_v2-not.parquet" \
      --y-a "data/processed/scores/gnn/high_rur_sage_full_rolling_is_rating_deviation_v2.parquet" \
            "data/processed/scores/gnn/high_rur_sage_full_rolling_is_rating_deviation_v2_seed?.parquet" \
      --y-b "data/processed/scores/mlp/high_all_seed*_rolling_is_rating_deviation_v2.parquet"
출력: results/bootstrap/<name>.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("pb10", ROOT / "scripts" / "10_paired_bootstrap.py")
pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pb)


def load_cell(a_pat: list[str], b_pat: list[str], y_map: pd.Series):
    fa, fb = pb.expand(a_pat), pb.expand(b_pat)
    ids_a, SA = pb.load_test_scores(fa)
    ids_b, SB = pb.load_test_scores(fb)
    if not np.array_equal(ids_a, ids_b):
        raise ValueError("같은 칸의 GNN·MLP test 리뷰 집합이 다릅니다")
    return fa, fb, ids_a, y_map.loc[ids_a].to_numpy().astype(int), SA, SB


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--x-a", nargs="+", required=True, help="칸 X 의 GNN 점수 파일들")
    ap.add_argument("--x-b", nargs="+", required=True, help="칸 X 의 MLP 점수 파일들")
    ap.add_argument("--y-a", nargs="+", required=True, help="칸 Y 의 GNN 점수 파일들")
    ap.add_argument("--y-b", nargs="+", required=True, help="칸 Y 의 MLP 점수 파일들")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    y_map = pd.read_parquet(pb.NODES, columns=["review_id", "fraud"]).set_index("review_id")["fraud"]
    X = load_cell(args.x_a, args.x_b, y_map)
    Y = load_cell(args.y_a, args.y_b, y_map)
    if np.intersect1d(X[2], Y[2]).size:
        raise ValueError("두 칸의 test 리뷰가 겹칩니다 — 독립 복원추출 가정이 깨집니다")

    def delta(cell, idx, sa, sb):
        _, _, _, y, SA, SB = cell
        return pb.ap_rows(y, SA[sa], idx).mean() - pb.ap_rows(y, SB[sb], idx).mean()

    full = lambda c: np.arange(len(c[3]))  # noqa: E731
    all_a = lambda c: np.arange(len(c[4]))  # noqa: E731
    all_b = lambda c: np.arange(len(c[5]))  # noqa: E731
    dx, dy = delta(X, full(X), all_a(X), all_b(X)), delta(Y, full(Y), all_a(Y), all_b(Y))
    print(f"칸 X Δ {dx:+.4f} (test {len(X[3]):,}, 사기 {X[3].sum():,}) | 칸 Y Δ {dy:+.4f} "
          f"(test {len(Y[3]):,}, 사기 {Y[3].sum():,}) | 차이 {dx - dy:+.4f}")

    rng = np.random.default_rng(args.seed)
    diff, bx, by = (np.empty(args.n_boot) for _ in range(3))
    for b in range(args.n_boot):
        vals = []
        for c in (X, Y):
            idx = rng.integers(0, len(c[3]), len(c[3]))
            sa = rng.integers(0, len(c[4]), len(c[4]))
            sb = rng.integers(0, len(c[5]), len(c[5]))
            vals.append(delta(c, idx, sa, sb))
        bx[b], by[b] = vals
        diff[b] = vals[0] - vals[1]

    res = {
        "name": args.name,
        "cell_x": {"files_a": [f.name for f in X[0]], "files_b": [f.name for f in X[1]],
                   "n_test": int(len(X[3])), "n_fraud_test": int(X[3].sum()), "delta": float(dx),
                   "ci95_delta": pb.ci(bx)},
        "cell_y": {"files_a": [f.name for f in Y[0]], "files_b": [f.name for f in Y[1]],
                   "n_test": int(len(Y[3])), "n_fraud_test": int(Y[3].sum()), "delta": float(dy),
                   "ci95_delta": pb.ci(by)},
        "delta_diff_x_minus_y": float(dx - dy),
        "ci95_delta_diff": pb.ci(diff),
        "share_diff_le_0": float((diff <= 0).mean()),
        "n_boot": args.n_boot, "boot_seed": args.seed,
        "note": "칸마다 test 리뷰와 seed 를 독립 복원추출. Δ = GNN 평균 PR-AUC − MLP 평균 PR-AUC",
    }
    pb.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = pb.OUT_DIR / f"{args.name}.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    lo, hi = res["ci95_delta_diff"]
    print(f"차이 {dx - dy:+.4f} | 95% CI [{lo:+.4f}, {hi:+.4f}] | 차이 ≤ 0 비율 {res['share_diff_le_0']:.3f}")
    print(f"저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
