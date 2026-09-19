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
    if features == "all_text":
        # 로드맵 ③: 수작업 37개 + 텍스트 384 — "텍스트를 더했을 때" 효과(조건 B 기준선)
        Xa, ca = load_features(nodes, "all")
        Xt, ct = load_features(nodes, "text")
        return np.hstack([Xa, Xt]), ca + ct
    f = load_rayana_asof_features(level=None if features == "all" else features)
    assert (f["review_id"].to_numpy() == rid).all(), "피처 행 순서가 graph_2014 노드와 다릅니다"
    cols = [c for c in f.columns if c != "review_id"]
    return f[cols].to_numpy(dtype=np.float32), cols


def run(nodes: pd.DataFrame, X_all: np.ndarray, group: str, seed: int,
        split: str = "review", drop_ties: bool = False,
        behavior_col: str | None = None, behavior: str = "yes",
        ablation: str | None = None, ablation_seed: int = 42) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    target = b05.build_group_mask(nodes, group, le2=False)
    if drop_ties:
        target = target & ~b05.sameday_tie_mask(nodes)
    if behavior_col:
        target = target & b05.behavior_mask(nodes, behavior_col, behavior)
    X = X_all[target]
    fraud = nodes.loc[target, "fraud"].to_numpy().astype("float32")
    n = len(X)

    if ablation and split != "review":
        raise ValueError("--ablation 은 --split review 와 함께만 사용합니다 (test-fixed 설계, 05 와 동일)")
    if split == "user":
        # 작성자 단위 분할 — 같은 작성자의 리뷰가 train/test 에 나뉘지 않게 해
        # "작성자 암기" 효과를 제거한다(README 주의사항 3번 검증용).
        folds = [b05.user_stratified_split(nodes.loc[target, "user_id"].to_numpy(), fraud, seed=seed)]
    elif split == "time":
        # 시간 단위 분할 — 1~9월 train / 10~11월 val / 12월 test (주 결과용, 05 와 동일 함수)
        folds = [b05.temporal_split(nodes.loc[target, "date"])]
    elif split == "rolling":
        # 한 달씩 밀어가며 3회차 — test 10·11·12월 합산 (선택지 (c'), 05 와 동일 함수)
        folds = b05.rolling_splits(nodes.loc[target, "date"])
    else:
        folds = [b05.stratified_split(fraud, seed=ablation_seed if ablation else seed)]
    if ablation:
        # 누수분해 test-fixed ablation(9/12 §10.1) — test/val 은 고정, train 에서만 뺀다.
        # GNN(05)과 같은 ablation_seed·같은 함수를 써서 동일한 test 집합을 보장한다.
        train_rel0, val_rel, test_rel = folds[0]
        train_rel = b05.ablation_train_mask(nodes.loc[target, "date"], nodes.loc[target, "user_id"].to_numpy(),
                                            train_rel0, test_rel, ablation)
        folds = [(train_rel, val_rel, test_rel)]

    y_t = torch.from_numpy(fraud)
    eye = scipy_to_torch_sparse(sp.eye(n, format="csr", dtype=np.float32))   # 이웃 없음
    test_score = np.full(n, np.nan)
    test_fold = np.zeros(n, dtype=int)
    val_score = np.full(n, np.nan)
    rates, fold_results = [], []

    for k, masks in enumerate(folds, 1):
        train_idx, val_idx, test_idx = (np.flatnonzero(m) for m in masks)

        mu = X[train_idx].astype("float64").mean(axis=0, keepdims=True)
        sigma = X[train_idx].astype("float64").std(axis=0, keepdims=True)
        sigma[sigma < 1e-6] = 1.0
        x_t = torch.from_numpy(((X - mu) / sigma).astype("float32"))
        train_idx_t = torch.from_numpy(train_idx)

        pos = fraud[train_idx].sum()
        pos_weight = torch.tensor([(len(train_idx) - pos) / max(pos, 1.0)])

        torch.manual_seed(seed)   # 회차마다 같은 초기화. 회차가 1개인 분할은 기존과 동일
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
                best_state = {k_: v.clone() for k_, v in model.state_dict().items()}
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
        test_score[test_idx] = scores[test_idx]
        test_fold[test_idx] = k
        val_score[val_idx] = scores[val_idx]
        rates.append((rate, len(train_idx)))
        fold_results.append({"fold": k, "n_train": len(train_idx), "n_test": len(test_idx), "epochs_run": epoch,
                             "test": evaluate(fraud[test_idx], scores[test_idx], pos_rate=rate)})

    rid = nodes.loc[target, "review_id"].to_numpy()
    if len(folds) == 1:
        split_names = np.full(n, "train", dtype=object)
        split_names[val_idx] = "val"
        split_names[test_idx] = "test"
        return {
            "n_target": n,
            "fraud_rate_target": float(fraud.mean()),
            "fraud_rate_test": float(fraud[test_idx].mean()),
            "epochs_run": epoch,
            "val": evaluate(fraud[val_idx], scores[val_idx], pos_rate=rate),
            "test": evaluate(fraud[test_idx], scores[test_idx], pos_rate=rate),
            # 점수 저장용(결과 JSON 에는 넣지 않음)
            "_scores": (rid, split_names, scores),
        }
    # rolling: 세 회차 test 를 합쳐 채점, 고정 임계값은 회차 train 사기율의 가중 평균
    pooled = float(sum(r * m for r, m in rates) / sum(m for _, m in rates))
    te, va = ~np.isnan(test_score), ~np.isnan(val_score)
    return {
        "n_target": n,
        "fraud_rate_target": float(fraud.mean()),
        "fraud_rate_test": float(fraud[te].mean()),
        "epochs_run": [f["epochs_run"] for f in fold_results],
        "val": evaluate(fraud[va], val_score[va], pos_rate=pooled),
        "test": evaluate(fraud[te], test_score[te], pos_rate=pooled),
        "folds": fold_results,
        "_scores": (rid[te], np.full(int(te.sum()), "test", dtype=object), test_score[te], test_fold[te]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="*", default=GROUPS, choices=GROUPS)
    ap.add_argument("--features", nargs="*", default=FEATURE_SETS, choices=FEATURE_SETS + ["all_text"],
                    help="기본은 5구성. all_text(수작업 37 + 텍스트 384, 로드맵 ③)는 따로 지정해야 돈다")
    ap.add_argument("--exclude-cols", nargs="*", default=[],
                    help="피처에서 뺄 컬럼 (예: RD DEV EXT — 평점 이탈형 정의에 쓰인 피처, 05 와 동일). "
                         "파일명에 _excl-... 이 붙는다")
    ap.add_argument("--behavior-col", default=None,
                    help="2×2 칸 선택용 행동 패턴 라벨 컬럼(burst_labels.parquet, 05 와 동일)")
    ap.add_argument("--behavior", default="yes", choices=["yes", "no"])
    ap.add_argument("--save-scores", action="store_true",
                    help="대상 리뷰별 점수를 data/processed/scores/mlp/ 에 저장(짝지은 부트스트랩용, git 비대상)")
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--drop-sameday-ties", action="store_true",
                    help="작성자 첫날 동률 리뷰를 대상에서 제외(05 와 동일). 파일명에 _notie")
    ap.add_argument("--split", default="review", choices=["review", "user", "time", "rolling"],
                    help="review(기본, 기존 결과와 동일) / user(작성자 단위 분할 — "
                         "결과 파일명에 _usersplit 접미사가 붙는다) / time(1~9월 train, "
                         "10~11월 val, 12월 test — 접미사 _timesplit) / rolling(한 달씩 밀어가며 3회차, "
                         "test 10~12월 합산 — 접미사 _rolling)")
    ap.add_argument("--ablation", default=None, choices=["a", "bstar", "cstar"],
                    help="누수분해 test-fixed ablation(9/12 §10.1, 05 와 동일 정의). --split review 필수. "
                         "GNN(05)과 같은 --ablation-seed 로 돌려야 같은 test 로 그래프 기여를 뺄 수 있다.")
    ap.add_argument("--ablation-seed", type=int, default=42,
                    help="--ablation 사용 시 train/val/test 분할을 고정하는 seed(기본 42, 05 와 동일)")
    args = ap.parse_args()

    t0 = time.time()
    nodes = pd.read_parquet(b05.GRAPH_DIR / "nodes.parquet")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    # seed 는 08 파일명에 원래 들어가므로 접미사에서는 뺀다(seed=42 로 넘김)
    sfx = b05.run_suffix(args.split, args.drop_sameday_ties, args.behavior_col, args.behavior,
                         args.exclude_cols, seed=42, ablation=args.ablation)
    for features in args.features:
        X_all, cols = load_features(nodes, features)
        if args.exclude_cols:
            keep = [i for i, c in enumerate(cols) if c not in args.exclude_cols]
            X_all, cols = X_all[:, keep], [cols[i] for i in keep]
        for group in args.groups:
            for seed in args.seeds:
                r = run(nodes, X_all, group, seed, split=args.split,
                        drop_ties=args.drop_sameday_ties,
                        behavior_col=args.behavior_col, behavior=args.behavior,
                        ablation=args.ablation, ablation_seed=args.ablation_seed)
                sc = r.pop("_scores")   # (review_id, split, score[, fold])
                r.update({"model": "MLP (VanillaGNN-gcn, 인접행렬=단위행렬)", "group": group,
                          "features": features, "n_features": len(cols), "seed": seed,
                          "split": args.split, "excluded_cols": args.exclude_cols,
                          "behavior_col": args.behavior_col,
                          "behavior": args.behavior if args.behavior_col else None,
                          "ablation": args.ablation,
                          "ablation_seed": args.ablation_seed if args.ablation else None,
                          "feature_columns": (cols if "text" not in features else
                                              "emb_minilm_384 (384)" if features == "text" else
                                              cols[:len(cols) - 384] + ["emb_minilm_384 (384)"]),
                          "hyperparams": {"hidden": HIDDEN, "dropout": DROPOUT, "lr": LR,
                                          "weight_decay": WD, "epochs": EPOCHS, "patience": PATIENCE}})
                stem = f"{group}_{features}_seed{seed}{sfx}"
                if args.save_scores:
                    score_path = b05.SCORES_DIR / "mlp" / f"{stem}.parquet"
                    b05.save_scores(score_path, sc[0], sc[1], sc[2], fold=sc[3] if len(sc) == 4 else None)
                    r["scores_file"] = str(score_path.relative_to(ROOT)).replace("\\", "/")
                out = OUT_DIR / f"{stem}.json"
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
        name = f"summary{sfx}.json"   # review 분할·옵션 없음이면 summary.json (기존과 동일)
        (OUT_DIR / name).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n요약 (test, seed 평균 ± 표준편차)")
    for _, a in agg.iterrows():
        print(f"  {a.group:<5} {a.features:<8} ({a.n_features:>3}개) PR-AUC {a.auprc_mean:.4f}±{a.auprc_std:.4f} "
              f"(x{a.auprc_mean / a.fraud_rate:.2f}) AUROC {a.auroc_mean:.4f}±{a.auroc_std:.4f} "
              f"MacroF1 {a.macro_f1_mean:.4f}±{a.macro_f1_std:.4f}")
    print(f"총 {time.time() - t0:.0f}초")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
