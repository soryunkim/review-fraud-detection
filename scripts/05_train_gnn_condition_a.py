"""
05_train_gnn_condition_a.py — 조건 A(구조/메타데이터) 바닐라 GNN 학습·평가

조건 A = 관계 그래프 구조 + 수작업 피처(38차원) + 바닐라 GCN/GraphSAGE.

**2026-09-11 재작성**: 구 버전(무방향 R-U-R, 임계값 3, 전체 연도)은 폐기.
`scripts/06_build_graphs_2014.py`가 만든 2014년 과거 방향(temporal) 그래프
3종(R-U-R/R-S-R/R-T-R, `data/processed/graph_2014/`)을 사용한다. 각 리뷰는
자기보다 먼저 작성된 같은 관계의 리뷰에서만 메시지를 받으므로(미래 누수 차단),
대칭 정규화 대신 `temporal_graph.row_normalize`(행 정규화)를 쓴다.

Phase 1-B(7가지 관계 조합) 실험은 `--relations`로 조합을 지정해 반복 실행한다:
    rur / rsr / rtr / rur,rsr / rur,rtr / rsr,rtr / rur,rsr,rtr

유형은 저활동(`--group low`, as-of 누적 리뷰 수 ≤1 = 작성자의 첫 리뷰) vs
비저활동(`--group high`)으로 비교한다(오동진, 2026-09-11 확정). 그래프는
지정한 그룹 부분집합 안에서만 남기고(2014년 전체 그래프를 그 그룹의 노드로
재인덱싱), 그 부분그래프로 학습·평가한다 — 유형별로 독립된 미니 실험이라는
기존 매트릭스 설계(신규계정형/버스트형을 각각 별도로 도는 방식)와 일관되게
유지한 것. 그룹을 섞은 전체 그래프로 학습하고 그룹별로 평가만 나누는
대안(전체 그래프 학습)도 고려할 수 있으나, 그러면 비저활동 유저의 신호가
저활동 예측에 새어 들어와 "관계 자체의 정보 추가 효과"가 아니라 "이웃
데이터가 많아진 효과"와 섞일 위험이 있어 채택하지 않았다 — 필요하면
후속 논의.

⚠️ `--exclude-cols`: 라벨 정의에 쓰인 피처(`singleton` 등)를 빼는 병행
보고용(9/2 피드백). 현재 `features_handcrafted.parquet`는 오동진이 as-of
방식으로 재계산 중(2026-09-11 기준 진행 중) — 완료 전까지 이 스크립트의
결과는 잠정치다.

사용 예:
    python scripts/05_train_gnn_condition_a.py --group low --relations rur
    python scripts/05_train_gnn_condition_a.py --group low --relations rur,rsr,rtr
    python scripts/05_train_gnn_condition_a.py --group high --relations rsr --backbone sage
"""

from __future__ import annotations

import argparse
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

from src.data.loaders import load_handcrafted_features  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402
from src.features.structural.graph import scipy_to_torch_sparse  # noqa: E402
from src.features.structural.temporal_graph import RELATIONS, combine, row_normalize  # noqa: E402
from src.models.gcn import VanillaGNN  # noqa: E402

GRAPH_DIR = ROOT / "data" / "processed" / "graph_2014"
RESULTS_DIR = ROOT / "results" / "condition_a" / "phase1b"


def stratified_split(fraud: np.ndarray, seed: int, val_frac: float = 0.15,
                     test_frac: float = 0.15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """fraud 비율을 유지하는 train/val/test 마스크(불리언, 노드 전체 길이)."""
    rng = np.random.default_rng(seed)
    n = len(fraud)
    train = np.zeros(n, dtype=bool)
    val = np.zeros(n, dtype=bool)
    test = np.zeros(n, dtype=bool)
    for cls in (0, 1):
        idx = np.flatnonzero(fraud == cls)
        rng.shuffle(idx)
        n_val = int(len(idx) * val_frac)
        n_test = int(len(idx) * test_frac)
        val[idx[:n_val]] = True
        test[idx[n_val:n_val + n_test]] = True
        train[idx[n_val + n_test:]] = True
    return train, val, test


def subsample_users(user_ids: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """작성자 단위 표본(불리언 마스크). 다작 작성자의 리뷰가 반쪼가리로 잘려 R-U-R
    구조와 라벨이 왜곡되는 것을 막기 위해 항상 작성자 단위로 자른다."""
    if frac >= 1.0:
        return np.ones(len(user_ids), dtype=bool)
    rng = np.random.default_rng(seed)
    users = np.unique(user_ids)
    picked = rng.choice(users, size=int(len(users) * frac), replace=False)
    return np.isin(user_ids, picked)


def load_graph_2014() -> tuple[pd.DataFrame, dict[str, sp.csr_matrix]]:
    if not GRAPH_DIR.exists():
        raise FileNotFoundError(
            f"{GRAPH_DIR} 가 없습니다. `python scripts/06_build_graphs_2014.py` 로 먼저 생성하세요."
        )
    nodes = pd.read_parquet(GRAPH_DIR / "nodes.parquet")
    adjs = {name: sp.load_npz(GRAPH_DIR / f"{name}.npz") for name in RELATIONS}
    return nodes, adjs


def build_group_mask(nodes: pd.DataFrame, group: str, le2: bool) -> np.ndarray:
    col = "type_new_le2" if le2 else "type_new"
    if group == "low":
        return nodes[col].to_numpy() == 1
    elif group == "high":
        return nodes[col].to_numpy() == 0
    return np.ones(len(nodes), dtype=bool)


def subset_adjacency(adj: sp.csr_matrix, mask: np.ndarray) -> sp.csr_matrix:
    """mask 로 선택된 노드끼리만 남긴 부분그래프. 로컬 인덱스가 0..n-1 로 재부여된다."""
    idx = np.flatnonzero(mask)
    return adj[idx][:, idx].tocsr()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="low", choices=["low", "high", "all"],
                    help="저활동(low, as-of<=1, 1순위) / 비저활동(high) / all(그룹 구분 없이 2014 전체)")
    ap.add_argument("--le2", action="store_true",
                    help="저활동 임계값을 as-of<=1 대신 <=2 로(민감도 분석용, type_new_le2 컬럼)")
    ap.add_argument("--relations", default="rur",
                    help="쉼표구분 관계 조합 {rur,rsr,rtr} 중 선택 — Phase 1-B 7조합: "
                         "rur / rsr / rtr / rur,rsr / rur,rtr / rsr,rtr / rur,rsr,rtr")
    ap.add_argument("--frac", type=float, default=1.0,
                    help="작성자 단위 표본 비율 (기본 1.0 = 전체). 빠른 반복 실험용")
    ap.add_argument("--backbone", default="gcn", choices=["gcn", "sage"])
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20,
                    help="val AUROC 개선 없이 버틸 최대 epoch 수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-cols", nargs="*", default=[],
                    help="피처에서 뺄 컬럼 (예: singleton — 라벨 정의에 쓰인 피처)")
    ap.add_argument("--out", type=Path, default=None,
                    help="결과 JSON 저장 경로 (기본: results/condition_a/phase1b/<group>_<relations>_<backbone>.json)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    relations = [r.strip().lower() for r in args.relations.split(",") if r.strip()]
    for r in relations:
        if r not in RELATIONS:
            raise ValueError(f"알 수 없는 관계: {r} (가능: {sorted(RELATIONS)})")
    if not relations:
        raise ValueError("--relations 가 비어있습니다.")

    t0 = time.time()
    print("[0/4] 2014년 그래프·노드·피처 로드")
    nodes, adjs = load_graph_2014()
    feats = load_handcrafted_features()
    feats_idx = feats.set_index("review_id")

    group_mask = build_group_mask(nodes, args.group, args.le2)
    sample_mask = subsample_users(nodes["user_id"].to_numpy(), args.frac, args.seed)
    keep = group_mask & sample_mask
    n_kept = int(keep.sum())
    if n_kept < 100:
        raise ValueError(f"부분집합이 너무 작습니다 (n={n_kept}). --frac 을 키우세요.")

    sub_nodes = nodes.loc[keep].reset_index(drop=True)
    fraud = sub_nodes["fraud"].to_numpy().astype("float32")

    feat_cols = [c for c in feats.columns if c != "review_id" and c not in args.exclude_cols]
    X = feats_idx.loc[sub_nodes["review_id"].to_numpy(), feat_cols].to_numpy(dtype="float32")

    group_label = f"{args.group}{'(<=2)' if args.le2 else ''}"
    print(f"[설정] group={group_label} relations={'+'.join(relations)} frac={args.frac} "
          f"n={n_kept:,} 사기율={fraud.mean():.1%} 피처={len(feat_cols)}개 backbone={args.backbone}")

    print("[1/4] 관계 그래프 결합 및 부분그래프 추출")
    combined = combine(adjs, relations)
    adj = subset_adjacency(combined, keep)
    n_edges = adj.nnz
    indeg = np.asarray(adj.sum(axis=1)).ravel()
    isolated = int((indeg == 0).sum())
    print(f"      엣지 {n_edges:,}개 | 과거 이웃 0개(고립) {isolated:,}개 "
          f"({isolated/n_kept:.1%}) | {time.time()-t0:.0f}초")

    self_loop = args.backbone == "gcn"
    adj_norm = scipy_to_torch_sparse(row_normalize(adj, self_loop=self_loop))

    print("[2/4] train/val/test 분할 (사기율 유지, seed 고정)")
    train_mask, val_mask, test_mask = stratified_split(fraud, seed=args.seed)
    print(f"      train {train_mask.sum():,} / val {val_mask.sum():,} / "
          f"test {test_mask.sum():,}")

    mu = X[train_mask].mean(axis=0, keepdims=True)
    sigma = X[train_mask].std(axis=0, keepdims=True)
    sigma[sigma == 0] = 1.0
    X = (X - mu) / sigma

    x_t = torch.from_numpy(X)
    y_t = torch.from_numpy(fraud)
    train_idx = torch.from_numpy(np.flatnonzero(train_mask))
    val_idx = np.flatnonzero(val_mask)
    test_idx = np.flatnonzero(test_mask)

    pos = fraud[train_mask].sum()
    neg = train_mask.sum() - pos
    pos_weight = torch.tensor([neg / max(pos, 1.0)])
    print(f"      pos_weight(train)={pos_weight.item():.2f}")

    print("[3/4] 학습")
    model = VanillaGNN(in_dim=X.shape[1], hidden_dim=args.hidden,
                       dropout=args.dropout, backbone=args.backbone)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auroc = -1.0
    best_state = None
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optim.zero_grad()
        logits = model(x_t, adj_norm)
        loss = loss_fn(logits[train_idx], y_t[train_idx])
        loss.backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            logits = model(x_t, adj_norm)
            val_scores = torch.sigmoid(logits[val_idx]).numpy()
        val_metrics = evaluate(fraud[val_idx], val_scores)

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"      epoch {epoch:3d} | loss {loss.item():.4f} | "
                  f"val AUROC {val_metrics['auroc']:.4f} AUPRC {val_metrics['auprc']:.4f}")

        if bad_epochs >= args.patience:
            print(f"      epoch {epoch}: {args.patience}회 개선 없어 조기 종료")
            break

    print("[4/4] 최종 평가 (best-val 체크포인트)")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x_t, adj_norm)
        scores = torch.sigmoid(logits).numpy()

    val_final = evaluate(fraud[val_idx], scores[val_idx])
    test_final = evaluate(fraud[test_idx], scores[test_idx])
    print(f"      val  : AUROC {val_final['auroc']:.4f} | AUPRC {val_final['auprc']:.4f} "
          f"| best-F1 {val_final['best_f1']:.4f}")
    print(f"      test : AUROC {test_final['auroc']:.4f} | AUPRC {test_final['auprc']:.4f} "
          f"| best-F1 {test_final['best_f1']:.4f}")

    result = {
        "condition": "A",
        "group": args.group,
        "le2": args.le2,
        "relations": relations,
        "graph_source": "data/processed/graph_2014 (과거 방향 temporal graph, 2026-09-11)",
        "backbone": args.backbone,
        "frac": args.frac,
        "n_nodes": n_kept,
        "n_edges": n_edges,
        "isolated_nodes": isolated,
        "fraud_rate": float(fraud.mean()),
        "n_features": len(feat_cols),
        "excluded_cols": args.exclude_cols,
        "epochs_run": epoch,
        "val": val_final,
        "test": test_final,
        "seed": args.seed,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out = args.out or (RESULTS_DIR / f"{args.group}_{'-'.join(relations)}_{args.backbone}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장: {out} | 총 {time.time()-t0:.0f}초")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
