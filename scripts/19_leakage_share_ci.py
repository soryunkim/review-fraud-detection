"""
19_leakage_share_ci.py — 누수 비중(28%)의 신뢰구간 (오동진, 2026-10-03)

누수 분해(9/12 §10.1)는 채점 대상을 고정한 채 학습 데이터만 바꾸는 test-fixed 설계다.
  A      원본(무작위 분할)                  기여 Δ_A
  B*     채점 대상 작성자의 미래 리뷰 제거    기여 Δ_B
  누수   Δ_A − Δ_B,  비중 = (Δ_A − Δ_B) / Δ_A
지금까지는 Δ_A, Δ_B 각각의 신뢰구간만 있어서 "28%가 유의한가"에 답할 수 없었다
(두 구간이 겹친다). 네 모델(A·B의 GNN/MLP)이 모두 같은 test 리뷰를 채점하므로,
같은 복원추출 표본에서 네 값을 함께 계산하면 짝지은 차이의 구간을 구할 수 있다.

  반복마다: test 리뷰를 복원추출(+ seed도 복원추출) → Δ_A, Δ_B, 누수, 비중을 함께 계산

사용 예
  python scripts/19_leakage_share_ci.py
출력: results/bootstrap/leakage_share.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
NODES = ROOT / "data" / "processed" / "graph_2014" / "nodes.parquet"
SCORES = ROOT / "data" / "processed" / "scores"
OUT = ROOT / "results" / "bootstrap" / "leakage_share.json"

SETS = {  # 조건 → (GNN 점수 glob, MLP 점수 glob)
    "A": ("gnn/high_rur_sage_full_abla.parquet", "gnn/high_rur_sage_full_abla_seed?.parquet",
          "mlp/high_all_seed*_abla.parquet"),
    "B": ("gnn/high_rur_sage_full_ablbstar.parquet", "gnn/high_rur_sage_full_ablbstar_seed?.parquet",
          "mlp/high_all_seed*_ablbstar.parquet"),
}


def load(patterns: list[str], ids: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    files: list[str] = []
    for p in patterns:
        hits = sorted(glob.glob(str(SCORES / p)))
        if not hits:
            raise FileNotFoundError(f"점수 파일 없음: {p}")
        files += hits
    mats = []
    for f in files:
        s = pd.read_parquet(f)
        s = s[s["split"] == "test"].sort_values("review_id")
        cur = s["review_id"].to_numpy()
        if ids is None:
            ids = cur
        elif not np.array_equal(ids, cur):
            raise ValueError(f"test 리뷰 집합이 다릅니다: {Path(f).name}")
        mats.append(s["score"].to_numpy(dtype=np.float64))
    return ids, np.column_stack(mats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    ids = None
    data = {}
    for k, (g1, g2, m) in SETS.items():
        ids, gnn = load([g1, g2], ids)
        ids, mlp = load([m], ids)
        if gnn.shape[1] != mlp.shape[1]:
            raise ValueError(f"{k}: seed 수가 다릅니다 (GNN {gnn.shape[1]}, MLP {mlp.shape[1]})")
        data[k] = (gnn, mlp)
    y = (pd.read_parquet(NODES, columns=["review_id", "fraud"])
         .set_index("review_id")["fraud"].reindex(ids).to_numpy(dtype=np.float64))
    n, S = len(y), data["A"][0].shape[1]

    def deltas(idx, seeds):
        out = {}
        for k, (g, m) in data.items():
            out[k] = float(np.mean([average_precision_score(y[idx], g[idx, s])
                                    - average_precision_score(y[idx], m[idx, s]) for s in seeds]))
        return out

    base = deltas(np.arange(n), range(S))
    leak = base["A"] - base["B"]
    share = leak / base["A"]

    boot = np.empty((args.n_boot, 4))
    for r in range(args.n_boot):
        idx = rng.integers(0, n, size=n)
        if y[idx].sum() < 2:
            boot[r] = np.nan
            continue
        sd = rng.integers(0, S, size=S)
        d = deltas(idx, sd)
        boot[r] = [d["A"], d["B"], d["A"] - d["B"], (d["A"] - d["B"]) / d["A"] if d["A"] else np.nan]

    def ci(col):
        v = boot[:, col]
        return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]

    res = {
        "n_test": int(n), "n_seeds": int(S), "n_boot": args.n_boot,
        "delta_A": base["A"], "ci95_delta_A": ci(0),
        "delta_B": base["B"], "ci95_delta_B": ci(1),
        "leak_abs": leak, "ci95_leak_abs": ci(2),
        "leak_share": share, "ci95_leak_share": ci(3),
        "share_leak_le_0": float(np.nanmean(boot[:, 2] <= 0)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Δ_A {res['delta_A']:+.4f} {np.round(res['ci95_delta_A'], 4)}")
    print(f"Δ_B {res['delta_B']:+.4f} {np.round(res['ci95_delta_B'], 4)}")
    print(f"누수(Δ_A−Δ_B) {leak:+.4f} {np.round(res['ci95_leak_abs'], 4)}  "
          f"(≤0 비율 {res['share_leak_le_0']:.3f})")
    print(f"누수 비중 {share * 100:.1f}% [{res['ci95_leak_share'][0] * 100:.1f}%, "
          f"{res['ci95_leak_share'][1] * 100:.1f}%]")
    print("저장:", OUT)


if __name__ == "__main__":
    main()
