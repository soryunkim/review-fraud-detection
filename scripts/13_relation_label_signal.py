"""
13_relation_label_signal.py — 관계 자체에 사기 신호가 있는가 (오동진, 2026-09-22)

버스트형은 성능 비교(GNN − MLP)로는 test filtered가 수십 건이라 판단이 안 됐다.
여기서는 모델을 학습하지 않고 그래프와 라벨만으로 두 단계를 본다.

  1단계 — 관계에 정보가 있나?
    각 리뷰의 과거 이웃 중 filtered 비율(nbr_rate)을 구하고, 그 값이 자기 라벨을 얼마나
    가르는지 AUC로 잰다. 버스트 리뷰 vs 나머지 리뷰를 비교한다. 2014년 전체(1~12월)
    리뷰를 쓰므로 버스트 filtered 표본이 test 한 달보다 훨씬 크다.
    주의: GNN은 이웃 '라벨'이 아니라 이웃 '피처'를 본다. 여기 값은 관계가 실어 나를 수
    있는 신호의 상한(oracle)에 가깝다.

  2단계 — 그 정보가 수작업 피처 37개와 겹치나?
    로지스틱 회귀로 (37개) vs (37개 + nbr_rate)를 1~9월 학습, 10~12월 채점하고 버스트/
    나머지 리뷰 안에서 PR-AUC 차이를 본다. nbr_rate가 37개 위에 더 보태는 게 없으면
    "정보는 있지만 피처와 겹친다", 보태면 "피처와 겹치지 않는 정보가 있다".
    같은 비교를 이웃 라벨 대신 이웃의 예측 점수 평균(현실판, 모델이 이웃 피처로 알 수 있는 만큼)
    으로도 한다. 라벨판은 오르는데 현실판은 안 오르면 "정보는 있지만 피처로는 꺼낼 수 없다".

신뢰구간: 상점 단위 cluster bootstrap.

사용 예
  python scripts/13_relation_label_signal.py
출력: results/relation_signal/summary.json, results/relation_signal/stage1.csv, stage2.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GRAPH_DIR = ROOT / "data" / "processed" / "graph_2014"
LABELS = ROOT / "data" / "processed" / "burst_labels.parquet"
FEATS = ROOT / "data" / "processed" / "features_rayana_asof_2014.parquet"
OUT_DIR = ROOT / "results" / "relation_signal"
RELATIONS = ["rur", "rsr", "rtr"]
TYPES = {"burst": "is_burst", "ratingdev": "is_rating_deviation_v2"}


def cluster_boot(fn, prod: np.ndarray, n_boot: int, rng) -> list[float]:
    """fn(weights) → 값. 상점 단위 복원추출 가중치로 반복."""
    uniq, inv = np.unique(prod, return_inverse=True)
    vals = []
    for _ in range(n_boot):
        cnt = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)))
        vals.append(fn(cnt[inv].astype(float)))
    v = np.array(vals, dtype=float)
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def w_auc(x, y, w):
    m = w > 0
    if y[m].min() == y[m].max():
        return np.nan
    return roc_auc_score(y[m], x[m], sample_weight=w[m])


def w_ap(x, y, w):
    m = w > 0
    if y[m].sum() == 0:
        return np.nan
    return average_precision_score(y[m], x[m], sample_weight=w[m])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    nodes = pd.read_parquet(GRAPH_DIR / "nodes.parquet")
    lab = pd.read_parquet(LABELS).set_index("review_id")
    y = nodes["fraud"].to_numpy().astype(float)
    prod = nodes["prod_id"].to_numpy()
    month = pd.to_datetime(nodes["date"]).dt.month.to_numpy()
    high = nodes["type_new"].to_numpy() == 0
    types = {k: lab[c].reindex(nodes["review_id"]).fillna(False).to_numpy().astype(bool)
             for k, c in TYPES.items()}

    nbr, adj = {}, {}
    for r in RELATIONS:
        A = sp.load_npz(GRAPH_DIR / f"{r}.npz").tocsr().astype(np.float64)
        adj[r] = A
        deg = np.asarray(A.sum(1)).ravel()
        fsum = A @ y
        rate = np.divide(fsum, deg, out=np.full_like(deg, np.nan), where=deg > 0)
        nbr[r] = (deg, rate)

    # ── 1단계 ────────────────────────────────────────────────────────────────
    rows1 = []
    for gname, gmask in [("high", high), ("low", ~high)]:
        for tname, tmask in types.items():
            for r in RELATIONS:
                deg, rate = nbr[r]
                for sub, smask in [("yes", tmask), ("no", ~tmask)]:
                    m = gmask & smask
                    has = m & (deg > 0)
                    x, yy, pp = rate[has], y[has], prod[has]
                    auc = w_auc(x, yy, np.ones(len(x))) if has.sum() > 1 else np.nan
                    ci = cluster_boot(lambda w: w_auc(x, yy, w), pp, args.n_boot, rng) if has.sum() > 50 else [np.nan, np.nan]
                    rows1.append({
                        "group": gname, "type": tname, "subset": sub, "relation": r,
                        "n": int(m.sum()), "n_filtered": int(y[m].sum()),
                        "filtered_rate": float(y[m].mean()) if m.sum() else np.nan,
                        "share_with_nbr": float(has.sum() / max(m.sum(), 1)),
                        "mean_deg": float(deg[m].mean()) if m.sum() else np.nan,
                        "nbr_rate_if_filtered": float(np.nanmean(rate[has & (y == 1)])) if (has & (y == 1)).any() else np.nan,
                        "nbr_rate_if_normal": float(np.nanmean(rate[has & (y == 0)])) if (has & (y == 0)).any() else np.nan,
                        "auc_nbr_rate": float(auc), "auc_lo": ci[0], "auc_hi": ci[1],
                    })
                print(f"[1] {gname} {tname} {r} done", flush=True)
    s1 = pd.DataFrame(rows1)

    # ── 2단계 ────────────────────────────────────────────────────────────────
    F = pd.read_parquet(FEATS).set_index("review_id").reindex(nodes["review_id"])
    X37 = F.to_numpy(dtype=np.float64)
    train = month <= 9
    test = month >= 10
    rows2 = []
    for gname, gmask in [("high", high), ("low", ~high)]:
        for r in RELATIONS:
            deg, rate = nbr[r]
            extra = np.column_stack([np.nan_to_num(rate, nan=0.0), (deg > 0).astype(float), np.log1p(deg)])
            preds = {}
            tr = gmask & train

            def fit_predict(X):
                sc = StandardScaler().fit(X[tr])
                clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
                return clf.predict_proba(sc.transform(X))[:, 1]

            preds["f37"] = fit_predict(X37)
            preds["f37+nbr"] = fit_predict(np.hstack([X37, extra]))
            # 현실판: 이웃 '라벨' 대신 이웃의 f37 예측 점수 평균(모델이 이웃 피처로 알 수 있는 만큼)
            A = adj[r]
            nbr_pred = np.divide(A @ preds["f37"], deg, out=np.zeros_like(deg), where=deg > 0)
            preds["f37+nbrpred"] = fit_predict(np.hstack([X37, np.column_stack(
                [nbr_pred, (deg > 0).astype(float), np.log1p(deg)])]))
            for tname, tmask in types.items():
                for sub, smask in [("yes", tmask), ("no", ~tmask), ("all", np.ones_like(tmask))]:
                    m = gmask & test & smask
                    b, yy, pp = preds["f37"][m], y[m], prod[m]
                    one = np.ones(len(yy))
                    row = {"group": gname, "relation": r, "type": tname, "subset": sub,
                           "n_test": int(m.sum()), "n_filtered": int(yy.sum()),
                           "ap_f37": float(w_ap(b, yy, one))}
                    for key, tag in [("f37+nbr", "label"), ("f37+nbrpred", "pred")]:
                        a = preds[key][m]
                        row[f"dAP_{tag}"] = float(w_ap(a, yy, one) - w_ap(b, yy, one))
                        ci = cluster_boot(lambda w: w_ap(a, yy, w) - w_ap(b, yy, w), pp, args.n_boot, rng)
                        row[f"{tag}_lo"], row[f"{tag}_hi"] = ci
                    rows2.append(row)
            print(f"[2] {gname} {r} done", flush=True)
    s2 = pd.DataFrame(rows2).drop_duplicates(subset=["group", "relation", "subset", "n_test"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s1.to_csv(OUT_DIR / "stage1.csv", index=False, encoding="utf-8-sig")
    s2.to_csv(OUT_DIR / "stage2.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "summary.json").write_text(json.dumps(
        {"stage1": rows1, "stage2": rows2, "n_boot": args.n_boot}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    pd.set_option("display.width", 200)
    print(s1.round(3).to_string())
    print(s2.round(3).to_string())


if __name__ == "__main__":
    main()
