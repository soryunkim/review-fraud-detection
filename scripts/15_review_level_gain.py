"""
15_review_level_gain.py — 리뷰 단위 그래프 기여 지표 (9/21 제출요약 §8 질문2 선택지 ②)

칸별 PR-AUC는 그 칸의 **걸러진(filtered) 리뷰 수**로 검정력이 정해진다 — 평점 이탈형
칸은 채점 대상 671건 중 걸러진 리뷰가 73건뿐이라 판정이 안 됐다(`14_score_by_type.py`).
이 스크립트는 PR-AUC 대신, 리뷰 한 건마다 "그래프를 쓰면 점수가 얼마나 바뀌었는가"
(Δscore = GNN 점수 − MLP 점수, seed 5개 평균)를 재고, Δscore 가 사기 여부를 얼마나
가르는지(AUC)를 본다. 표본이 **그 칸의 전체 리뷰 수**(671건)가 되어 filtered 수(73건)
보다 훨씬 커진다 — §8 질문2가 기대한 효과("표본이 2,030건") 그대로다.

지표: AUC(Δscore, fraud) — Δscore 가 사기 리뷰에서 정상 리뷰보다 체계적으로 큰지
(그래프가 사기 리뷰를 정상보다 더 밀어올렸는지)를 0.5 기준으로 판정한다. PR-AUC 차이와
달리 케이스별 쌍대(pairwise) 비교라 걸러진 리뷰의 절대 개수가 아니라 (사기, 정상) 쌍의
개수(대략 filtered × normal)로 검정력이 정해진다.

사용 예
    python scripts/15_review_level_gain.py                     # 기본 high × rur/rsr/rtr
출력: results/review_level_gain/<group>_<rel>.json, results/review_level_gain/summary.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NODES = ROOT / "data" / "processed" / "graph_2014" / "nodes.parquet"
LABELS = ROOT / "data" / "processed" / "burst_labels.parquet"
OUT_DIR = ROOT / "results" / "review_level_gain"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m14 = load_module("score_by_type14", "14_score_by_type.py")   # build_cells
m12 = load_module("type_intensity12", "12_type_intensity_gain.py")   # score_files, load_test, w_auc


def cell_stat(delta: np.ndarray, y: np.ndarray, prod: np.ndarray,
              n_boot: int, rng: np.random.Generator) -> dict:
    order = np.argsort(-delta, kind="stable")
    w1 = np.ones(len(y))
    auc = m12.w_auc(order, y, w1)
    mean_delta_fraud = float(delta[y == 1].mean()) if (y == 1).any() else float("nan")
    mean_delta_normal = float(delta[y == 0].mean()) if (y == 0).any() else float("nan")

    uniq, pinv = np.unique(prod, return_inverse=True)
    boot = np.empty(n_boot)
    for r in range(n_boot):
        cnt = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)))
        wall = cnt[pinv].astype(np.float64)
        boot[r] = m12.w_auc(order, y, wall)
    return {
        "n": int(len(y)), "n_fraud": int(y.sum()), "fraud_rate": float(y.mean()),
        "auc_delta_vs_fraud": float(auc),
        "auc_ci": [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))],
        "mean_delta_fraud": mean_delta_fraud, "mean_delta_normal": mean_delta_normal,
        "mean_delta_gap": mean_delta_fraud - mean_delta_normal,
    }


def analyze(group: str, rel: str, cells: dict[str, pd.Series], prod_of: pd.Series,
            n_boot: int, rng: np.random.Generator) -> dict:
    gf, mf = m12.score_files(group, rel)
    missing = [f.name for f in gf + mf if not f.exists()]
    if missing:
        return {"group": group, "relation": rel, "skipped": f"점수 파일 없음: {missing}"}
    GA, MB = m12.load_test(gf), m12.load_test(mf)
    if not GA.index.equals(MB.index):
        raise ValueError("GNN/MLP test 리뷰 집합이 다릅니다")
    y_all = pd.read_parquet(NODES, columns=["review_id", "fraud"]).set_index("review_id")["fraud"] \
        .reindex(GA.index).to_numpy(dtype=np.float64)
    prod_all = prod_of.reindex(GA.index).to_numpy()
    delta_all = GA.to_numpy().mean(axis=1) - MB.to_numpy().mean(axis=1)

    out = {"group": group, "relation": rel, "n_test_total": int(len(GA)), "n_boot": n_boot, "cells": {}}
    for name, mask in cells.items():
        sel = mask.reindex(GA.index).fillna(False).to_numpy()
        if sel.sum() < 20 or y_all[sel].sum() < 2 or (sel.sum() - y_all[sel].sum()) < 2:
            out["cells"][name] = {"skipped": f"n={int(sel.sum())}, n_fraud={int(y_all[sel].sum())}"}
            continue
        out["cells"][name] = cell_stat(delta_all[sel], y_all[sel], prod_all[sel], n_boot, rng)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="+", default=["high"])
    ap.add_argument("--relations", nargs="+", default=["rur", "rsr", "rtr"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    labels = pd.read_parquet(LABELS, columns=["review_id", "is_burst", "is_rating_deviation_v2"])
    cells = m14.build_cells(labels)
    prod_of = pd.read_parquet(NODES, columns=["review_id", "prod_id"]).set_index("review_id")["prod_id"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for g in args.groups:
        for rel in args.relations:
            res = analyze(g, rel, cells, prod_of, args.n_boot, np.random.default_rng(args.seed))
            (OUT_DIR / f"{g}_{rel}.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            if "skipped" in res:
                print(f"[skip] {g} {rel}: {res['skipped']}")
                continue
            print(f"\n{g} {rel} (n_test={res['n_test_total']:,})")
            for cname, c in res["cells"].items():
                if "skipped" in c:
                    print(f"  {cname:30s} [skip] {c['skipped']}")
                    continue
                print(f"  {cname:30s} n={c['n']:6,d}(fraud {c['n_fraud']:4d}) "
                      f"AUC(Δscore,fraud) {c['auc_delta_vs_fraud']:.4f} "
                      f"[{c['auc_ci'][0]:.4f}, {c['auc_ci'][1]:.4f}]")
                rows.append({"group": g, "relation": rel, "cell": cname,
                             "n": c["n"], "n_fraud": c["n_fraud"],
                             "auc": c["auc_delta_vs_fraud"], "auc_lo": c["auc_ci"][0], "auc_hi": c["auc_ci"][1],
                             "mean_delta_gap": c["mean_delta_gap"]})
    if rows:
        pd.DataFrame(rows).to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
