"""
14_score_by_type.py — 학습은 그룹 전체로, 채점만 유형별로 (9/21 제출요약 §8 질문2 선택지 ①)

§6-3의 문제: 평점 이탈형·버스트형처럼 드문 유형을 "그 칸으로만 학습"하면(예: 평점
이탈형 7,908건) 채점 대상의 걸러진 리뷰가 73건뿐이라 판정이 안 됐다. 이 스크립트는
재학습 없이 — `05`/`08` `--save-scores`로 이미 저장해 둔 **비저활동형(high) 전체
101,926건 학습** 모델의 점수(`12_type_intensity_gain.py`가 쓰는 것과 같은 파일)를
불러와, 채점 단계에서만 유형 라벨(`burst_labels.parquet`)로 잘라 ΔPR-AUC를 잰다.

"학습 대상을 줄이면 손해인가" 자체는 이 스크립트로 답할 수 없다(그 비교를 하려면
실제로 칸만으로 재학습해야 한다 — 그 결과는 이미 `results/condition_a/phase1b/
{group}_{rel}_sage_full_{split}_{behavior_col}[...].json`에 있다). 이 스크립트는
"표본 문제가 재학습 없이 완화되는가"만 확인한다: 채점 대상 자체가 그 유형에 속한
리뷰 전부(전체 test 안에서의 유형 비율)이므로, 걸러진 리뷰 수는 학습 방식과 무관하게
동일하다 — 즉 표본 부족은 **채점 대상의 정의**(유형 안의 filtered 리뷰 수) 문제이지
학습 대상 문제가 아니라는 것을 이 결과 자체가 보여준다.

셀 정의(`burst_labels.parquet`):
  is_burst / is_rating_deviation_v2  각각 yes/no
  normal = 둘 다 no ("일반" 칸, §3의 기준 칸과 동일 정의)

사용 예
    python scripts/14_score_by_type.py                       # 기본 high × rur/rsr/rtr
출력: results/score_by_type/<group>_<rel>.json, results/score_by_type/summary.csv
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
OUT_DIR = ROOT / "results" / "score_by_type"
SEEDS = [42, 0, 1, 2, 3]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m12 = load_module("type_intensity12", "12_type_intensity_gain.py")   # score_files, load_test, w_ap, w_auc


def build_cells(labels: pd.DataFrame) -> dict[str, pd.Series]:
    """review_id -> bool 마스크. normal = 두 유형 모두 아님(§3의 '일반' 칸과 동일 정의)."""
    burst = labels.set_index("review_id")["is_burst"]
    dev = labels.set_index("review_id")["is_rating_deviation_v2"]
    return {
        "all": pd.Series(True, index=burst.index),
        "is_burst=yes": burst,
        "is_burst=no": ~burst,
        "is_rating_deviation_v2=yes": dev,
        "is_rating_deviation_v2=no": ~dev,
        "normal(burst=no & dev=no)": (~burst) & (~dev),
    }


def cell_delta(a: np.ndarray, b: np.ndarray, y: np.ndarray, prod: np.ndarray,
               n_boot: int, rng: np.random.Generator) -> dict:
    """a, b: (n, n_seeds) 점수 행렬. 상점 단위 cluster bootstrap (12 와 동일 관례)."""
    S = a.shape[1]
    A = [np.argsort(-a[:, s], kind="stable") for s in range(S)]
    B = [np.argsort(-b[:, s], kind="stable") for s in range(S)]
    w1 = np.ones(len(y))
    dap = np.mean([m12.w_ap(A[s], y, w1) - m12.w_ap(B[s], y, w1) for s in range(S)])
    dauc = np.mean([m12.w_auc(A[s], y, w1) - m12.w_auc(B[s], y, w1) for s in range(S)])
    ap_a = np.mean([m12.w_ap(A[s], y, w1) for s in range(S)])
    ap_b = np.mean([m12.w_ap(B[s], y, w1) for s in range(S)])

    uniq, pinv = np.unique(prod, return_inverse=True)
    boot_dap = np.empty(n_boot)
    boot_dauc = np.empty(n_boot)
    for r in range(n_boot):
        cnt = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)))
        wall = cnt[pinv].astype(np.float64)
        sd = rng.integers(0, S, size=S)
        boot_dap[r] = np.mean([m12.w_ap(A[s], y, wall) - m12.w_ap(B[s], y, wall) for s in sd])
        boot_dauc[r] = np.mean([m12.w_auc(A[s], y, wall) - m12.w_auc(B[s], y, wall) for s in sd])
    return {
        "n": int(len(y)), "n_filtered": int(y.sum()), "filtered_rate": float(y.mean()),
        "auprc_full_group_trained_gnn": float(ap_a), "auprc_full_group_trained_mlp": float(ap_b),
        "dAP": float(dap), "dAP_ci": [float(np.nanpercentile(boot_dap, 2.5)), float(np.nanpercentile(boot_dap, 97.5))],
        "dAUC": float(dauc), "dAUC_ci": [float(np.nanpercentile(boot_dauc, 2.5)), float(np.nanpercentile(boot_dauc, 97.5))],
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
    a_all, b_all = GA.to_numpy(), MB.to_numpy()

    out = {"group": group, "relation": rel, "n_test_total": int(len(GA)), "n_boot": n_boot, "cells": {}}
    for name, mask in cells.items():
        sel = mask.reindex(GA.index).fillna(False).to_numpy()
        if sel.sum() < 20:
            out["cells"][name] = {"skipped": f"n={int(sel.sum())} < 20"}
            continue
        out["cells"][name] = cell_delta(a_all[sel], b_all[sel], y_all[sel], prod_all[sel], n_boot, rng)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="+", default=["high"],
                    help="기본 high(비저활동)만 — §6-3의 표본 문제가 이 그룹에서만 발생")
    ap.add_argument("--relations", nargs="+", default=["rur", "rsr", "rtr"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    labels = pd.read_parquet(LABELS, columns=["review_id", "is_burst", "is_rating_deviation_v2"])
    cells = build_cells(labels)
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
                print(f"  {cname:30s} n={c['n']:6,d} filtered={c['n_filtered']:4d} "
                      f"dAP {c['dAP']:+.4f} [{c['dAP_ci'][0]:+.4f}, {c['dAP_ci'][1]:+.4f}]")
                rows.append({"group": g, "relation": rel, "cell": cname, **{k: v for k, v in c.items()
                             if k not in ("dAP_ci", "dAUC_ci")},
                             "dAP_lo": c["dAP_ci"][0], "dAP_hi": c["dAP_ci"][1],
                             "dAUC_lo": c["dAUC_ci"][0], "dAUC_hi": c["dAUC_ci"][1]})
    if rows:
        pd.DataFrame(rows).to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
