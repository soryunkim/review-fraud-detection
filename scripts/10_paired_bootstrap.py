"""
10_paired_bootstrap.py — 두 모델(GNN vs MLP) PR-AUC 차이의 짝지은 부트스트랩 신뢰구간 (오동진, 2026-09-13)

같은 test 리뷰로 채점한 두 모델의 차이(그래프 기여 = GNN − MLP)는, 모델 하나의 PR-AUC보다
오차가 훨씬 작다. 두 모델의 점수가 같은 리뷰에서 같이 움직이기 때문이다. 이 스크립트는
05·08 `--save-scores` 로 저장한 리뷰별 점수로 그 오차를 실측한다.

  (1) test 표본 오차: test 리뷰를 복원추출해 두 모델을 같은 표본으로 함께 채점
  (2) 전체 오차: (1)에 더해 seed 도 복원추출 (초기값 흔들림 + test 표본 흔들림)
  (3) 참고: 모델 하나의 PR-AUC 오차 (짝짓지 않았을 때 얼마나 과장되는지 비교용)
  두 모델 점수의 실제 상관(ρ)도 함께 보고한다.

사용 예
  python scripts/10_paired_bootstrap.py --name high_rur_sage_timesplit \
      --a data/processed/scores/gnn/high_rur_sage_full_timesplit.parquet \
          "data/processed/scores/gnn/high_rur_sage_full_timesplit_seed?.parquet" \
      --b "data/processed/scores/mlp/high_all_seed*_timesplit.parquet"
출력: results/bootstrap/<name>.json (요약 통계만 — 리뷰별 값은 저장하지 않음)
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
NODES = ROOT / "data" / "processed" / "graph_2014" / "nodes.parquet"
OUT_DIR = ROOT / "results" / "bootstrap"


def expand(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in patterns:
        hits = sorted(glob.glob(str(ROOT / p) if not Path(p).is_absolute() else p))
        if not hits:
            raise FileNotFoundError(f"점수 파일 없음: {p}")
        files += [Path(h) for h in hits]
    return files


def load_test_scores(files: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """test 리뷰만. 모든 파일의 test 리뷰 집합이 같아야 한다(같은 채점 문제지)."""
    ids, mats = None, []
    for f in files:
        s = pd.read_parquet(f)
        s = s[s["split"] == "test"].sort_values("review_id")
        if ids is None:
            ids = s["review_id"].to_numpy()
        elif not np.array_equal(ids, s["review_id"].to_numpy()):
            raise ValueError(f"test 리뷰 집합이 다릅니다: {f.name}")
        mats.append(s["score"].to_numpy(dtype=np.float64))
    return ids, np.vstack(mats)


def ap_rows(y: np.ndarray, S: np.ndarray, idx: np.ndarray) -> np.ndarray:
    yy = y[idx]
    return np.array([average_precision_score(yy, s[idx]) for s in S])


def ci(x: np.ndarray) -> list[float]:
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="결과 파일 이름 (results/bootstrap/<name>.json)")
    ap.add_argument("--a", nargs="+", required=True, help="모델 A(보통 GNN) 점수 파일들 — seed 별, 글롭 가능")
    ap.add_argument("--b", nargs="+", required=True, help="모델 B(보통 MLP) 점수 파일들 — seed 별, 글롭 가능")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    fa, fb = expand(args.a), expand(args.b)
    print(f"A {len(fa)}개: {[f.name for f in fa]}")
    print(f"B {len(fb)}개: {[f.name for f in fb]}")
    ids_a, SA = load_test_scores(fa)
    ids_b, SB = load_test_scores(fb)
    if not np.array_equal(ids_a, ids_b):
        raise ValueError("A 와 B 의 test 리뷰 집합이 다릅니다 — 같은 분할·같은 칸인지 확인하세요")
    nodes = pd.read_parquet(NODES, columns=["review_id", "fraud"]).set_index("review_id")
    y = nodes.loc[ids_a, "fraud"].to_numpy().astype(int)
    n = len(y)

    full = np.arange(n)
    ap_a, ap_b = ap_rows(y, SA, full), ap_rows(y, SB, full)
    delta = ap_a.mean() - ap_b.mean()
    k = min(len(SA), len(SB))
    rho_p = [pearsonr(SA[i], SB[i])[0] for i in range(k)]
    rho_s = [spearmanr(SA[i], SB[i])[0] for i in range(k)]
    print(f"test {n:,}건 (사기 {y.sum():,}) | A {ap_a.mean():.4f} B {ap_b.mean():.4f} Δ {delta:+.4f} "
          f"| 점수 상관 ρ(피어슨) {np.mean(rho_p):.3f}")

    rng = np.random.default_rng(args.seed)
    d_test, d_total, a_only, b_only = (np.empty(args.n_boot) for _ in range(4))
    for b in range(args.n_boot):
        idx = rng.integers(0, n, n)
        pa, pb = ap_rows(y, SA, idx), ap_rows(y, SB, idx)
        d_test[b] = pa.mean() - pb.mean()
        a_only[b], b_only[b] = pa.mean(), pb.mean()
        sa = rng.integers(0, len(SA), len(SA))          # seed 도 복원추출
        sb = rng.integers(0, len(SB), len(SB))
        d_total[b] = pa[sa].mean() - pb[sb].mean()
        if (b + 1) % 500 == 0:
            print(f"  부트스트랩 {b + 1}/{args.n_boot} | {time.time() - t0:.0f}초", flush=True)

    res = {
        "name": args.name,
        "files_a": [f.name for f in fa], "files_b": [f.name for f in fb],
        "n_test": int(n), "n_fraud_test": int(y.sum()), "base_rate_test": float(y.mean()),
        "auprc_a_by_seed": ap_a.round(4).tolist(), "auprc_b_by_seed": ap_b.round(4).tolist(),
        "auprc_a_mean": float(ap_a.mean()), "auprc_b_mean": float(ap_b.mean()),
        "delta_mean": float(delta),
        "delta_by_seed_pair": (ap_a[:k] - ap_b[:k]).round(4).tolist(),
        "score_corr_pearson": float(np.mean(rho_p)), "score_corr_spearman": float(np.mean(rho_s)),
        "ci95_delta_test_only": ci(d_test),
        "ci95_delta_seed_and_test": ci(d_total),
        "halfwidth_delta_seed_and_test": float((np.percentile(d_total, 97.5) - np.percentile(d_total, 2.5)) / 2),
        "share_delta_le_0_seed_and_test": float((d_total <= 0).mean()),
        "ci95_auprc_a_alone": ci(a_only), "ci95_auprc_b_alone": ci(b_only),
        "halfwidth_auprc_a_alone": float((np.percentile(a_only, 97.5) - np.percentile(a_only, 2.5)) / 2),
        "n_boot": args.n_boot, "boot_seed": args.seed,
        "note": "Δ = A 평균 − B 평균 (seed 평균 PR-AUC). test_only 는 test 표본만, seed_and_test 는 seed 까지 복원추출",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.name}.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    lo, hi = res["ci95_delta_seed_and_test"]
    print(f"\nΔ {delta:+.4f} | 95% CI (seed+test) [{lo:+.4f}, {hi:+.4f}] | test만 {res['ci95_delta_test_only']} "
          f"| 참고: A 단독 PR-AUC 반폭 ±{res['halfwidth_auprc_a_alone']:.3f}")
    print(f"저장: {out} | {time.time() - t0:.0f}초")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
