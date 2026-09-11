"""
08_train_mlp_baseline.py — 그래프 없는 MLP 기준선 (실험 순서 ①, 조건 C 포함)

05_train_gnn_condition_a.py 와 **그래프만 빼고** 모든 조건을 똑같이 맞춘다.
→ 05 결과 − 이 결과 = 관계 그래프의 기여.

  모델   05 의 VanillaGNN(backbone="gcn") 에 인접행렬 = 단위행렬(자기 자신만).
         GCN 층이 A·XW → XW 가 되므로 Linear-ReLU-Dropout-Linear-ReLU-Dropout-Linear,
         즉 은닉 64 의 2층 MLP 와 수학적으로 같다(초기화·dropout 난수 순서까지 05 와 동일).
  나머지 그룹 마스크·stratified_split·학습 설정 기본값은 05 에서 그대로 가져온다.
         표준화는 05 의 수정본(float64 + sigma<1e-6 → 1, 커밋 ba45773)과 같은 식.

피처 구성 (--features)
  all      as-of 수작업 피처 37개 (07 산출물)
  review   리뷰 수준 16개
  user     작성자 수준 11개
  product  업체 수준 10개
  text     텍스트 임베딩 384차원만 (all-MiniLM-L6-v2) — 조건 C(텍스트 전용)

검증: 저활동 × all × seed 42 를 05 `--group low --relations rur --scope isolated`
      (저활동은 R-U-R 과거 이웃이 정의상 0개 → 그 실험도 이웃 없는 모델)와 비교하면
      초기 가중치·epoch 1~2 loss·val AUROC 가 소수점 끝까지 일치한다(같은 모델 확인).
      이후 부동소수점 누적 오차(1e-7 수준)로 조기 종료 시점이 갈려 test PR-AUC 가
      0.3665(05) vs 0.3701(08)로 달라진다 → 단일 실행 간 ±0.004 는 잡음 수준이라는 뜻.
      그래서 seed 5개 평균±표준편차를 함께 낸다.

사용 예
  python scripts/08_train_mlp_baseline.py                          # 2그룹 × 5구성 × seed 5개 전부
  python scripts/08_train_mlp_baseline.py --groups low --features all --seeds 42
출력: results/mlp_baseline/{group}_{features}_seed{seed}.json, summary.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loaders import load_embeddings, load_rayana_asof_features  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402
from src.features.structural.graph import scipy_to_torch_sparse  # noqa: E402
from src.models.gcn import VanillaGNN  # noqa: E402

_spec = importlib.util.spec_from_file_location("train05", ROOT / "scripts" / "05_train_gnn_condition_a.py")
b05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b05)

OUT_DIR = ROOT / "results" / "mlp_baseline"
FEATURE_SETS = ["all", "review", "user", "product", "text"]
GROUPS = ["low", "high"]
SEEDS = [42, 0, 1, 2, 3]
# 05 의 기본 하이퍼파라미터와 동일
HIDDEN, DROPOUT, LR, WD, EPOCHS, PATIENCE = 64, 0.3, 0.01, 5e-4, 200, 20


def load_features(nodes: pd.DataFrame, features: str) -> tuple[np.ndarray, list[str]]:
    rid = nodes["review_id"].to_numpy()
    if features == "text":
        emb = load_embeddings()
        return np.asarray(emb[rid], dtype=np.float32), [f"emb_{i}" for i in range(emb.shape[1])]
    f = load_rayana_asof_features(level=None if features == "all" else features)
    assert (f["review_id"].to_numpy() == rid).all(), "피처 행 순서가 graph_2014 노드와 다릅니다"
    cols = [c for c in f.columns if c != "review_id"]
    return f[cols].to_numpy(dtype=np.float32), cols


def run(nodes: pd.DataFrame, X_all: np.ndarray, group: str, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    target = b05.build_group_mask(nodes, group, le2=False)
    X = X_all[target]
    fraud = nodes.loc[target, "fraud"].to_numpy().astype("float32")
    n = len(X)

    train_idx, val_idx, test_idx = (np.flatnonzero(m) for m in b05.stratified_split(fraud, seed=seed))

    mu = X[train_idx].astype("float64").mean(axis=0, keepdims=True)
    sigma = X[train_idx].astype("float64").std(axis=0, keepdims=True)
    sigma[sigma < 1e-6] = 1.0
    X = ((X - mu) / sigma).astype("float32")

    x_t = torch.from_numpy(X)
    y_t = torch.from_numpy(fraud)
    train_idx_t = torch.from_numpy(train_idx)
    eye = scipy_to_torch_sparse(sp.eye(n, format="csr", dtype=np.float32))   # 이웃 없음

    pos = fraud[train_idx].sum()
    pos_weight = torch.tensor([(len(train_idx) - pos) / max(pos, 1.0)])

    model = VanillaGNN(in_dim=X.shape[1], hidden_dim=HIDDEN, dropout=DROPOUT, backbone="gcn")
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auroc, best_state, bad_epochs = -1.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optim.zero_grad()
        logits = model(x_t, eye)
        loss = loss_fn(logits[train_idx_t], y_t[train_idx_t])
        loss.backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(model(x_t, eye)[val_idx]).numpy()
        val_auroc = evaluate(fraud[val_idx], val_scores)["auroc"]
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= PATIENCE:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(x_t, eye)).numpy()
    rate = float(fraud[train_idx].mean())
    return {
        "n_target": n,
        "fraud_rate_target": float(fraud.mean()),
        "epochs_run": epoch,
        "val": evaluate(fraud[val_idx], scores[val_idx], pos_rate=rate),
        "test": evaluate(fraud[test_idx], scores[test_idx], pos_rate=rate),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="*", default=GROUPS, choices=GROUPS)
    ap.add_argument("--features", nargs="*", default=FEATURE_SETS, choices=FEATURE_SETS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    args = ap.parse_args()

    t0 = time.time()
    nodes = pd.read_parquet(b05.GRAPH_DIR / "nodes.parquet")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for features in args.features:
        X_all, cols = load_features(nodes, features)
        for group in args.groups:
            for seed in args.seeds:
                r = run(nodes, X_all, group, seed)
                r.update({"model": "MLP (VanillaGNN-gcn, 인접행렬=단위행렬)", "group": group,
                          "features": features, "n_features": len(cols), "seed": seed,
                          "feature_columns": cols if features != "text" else "emb_minilm_384 (384)",
                          "hyperparams": {"hidden": HIDDEN, "dropout": DROPOUT, "lr": LR,
                                          "weight_decay": WD, "epochs": EPOCHS, "patience": PATIENCE}})
                out = OUT_DIR / f"{group}_{features}_seed{seed}.json"
                out.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
                t = r["test"]
                print(f"{group:<5} {features:<8} seed {seed:<3} | AUROC {t['auroc']:.4f} "
                      f"PR-AUC {t['auprc']:.4f} (x{t['auprc'] / r['fraud_rate_target']:.2f}) "
                      f"MacroF1 {t['macro_f1']:.4f} 사기F1 {t['fraud_f1_fixed']:.4f} "
                      f"| ep {r['epochs_run']} | {time.time() - t0:.0f}초", flush=True)
                rows.append({"group": group, "features": features, "n_features": len(cols), "seed": seed,
                             "fraud_rate": r["fraud_rate_target"], **{k: t[k] for k in
                             ("auroc", "auprc", "macro_f1", "fraud_f1_fixed")}})

    df = pd.DataFrame(rows)
    agg = df.groupby(["group", "features"], sort=False).agg(
        n_features=("n_features", "first"), fraud_rate=("fraud_rate", "first"), n_seeds=("seed", "size"),
        auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
        auprc_mean=("auprc", "mean"), auprc_std=("auprc", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        fraud_f1_mean=("fraud_f1_fixed", "mean"), fraud_f1_std=("fraud_f1_fixed", "std")).reset_index()
    s42 = df[df.seed == 42].set_index(["group", "features"])
    summary = {"note": "test 지표. mean/std 는 seed 전체, seed42 는 05(GCN) 결과와 직접 비교용",
               "seeds": args.seeds, "rows": []}
    for _, a in agg.iterrows():
        row = a.to_dict()
        if (a.group, a.features) in s42.index:
            row["seed42"] = s42.loc[(a.group, a.features), ["auroc", "auprc", "macro_f1", "fraud_f1_fixed"]].to_dict()
        summary["rows"].append(row)
    if set(args.groups) == set(GROUPS) and set(args.features) == set(FEATURE_SETS):
        (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n요약 (test, seed 평균 ± 표준편차)")
    for _, a in agg.iterrows():
        print(f"  {a.group:<5} {a.features:<8} ({a.n_features:>3}개) PR-AUC {a.auprc_mean:.4f}±{a.auprc_std:.4f} "
              f"(x{a.auprc_mean / a.fraud_rate:.2f}) AUROC {a.auroc_mean:.4f}±{a.auroc_std:.4f} "
              f"MacroF1 {a.macro_f1_mean:.4f}±{a.macro_f1_std:.4f}")
    print(f"총 {time.time() - t0:.0f}초")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
