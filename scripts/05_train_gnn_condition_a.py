"""
05_train_gnn_condition_a.py — 조건 A(구조/메타데이터) 바닐라 GNN 학습·평가

조건 A = R-U-R(같은 작성자) 그래프 구조 + 수작업 피처(38차원) + 바닐라 GCN/GraphSAGE.
9/2 미팅 피드백에 따라 CARE-GNN(강화학습) 대신 가장 순수한 구조 신호 기준선부터
검증한다. 사기 서브유형별로 따로 돌려서(우선순위: 신규계정형 > 버스트형) 매트릭스의
"A" 열을 채운다.

그래프는 지정한 사기 유형(`--type-class`) 부분집합 안에서만 R-U-R 엣지를 만든다.
`docs/데이터_확인결과_오동진.md` §6: R-T-R·R-S-R은 라벨 동질성이 무작위 수준이라
쓰지 않음. `--exclude-cols singleton` 으로 라벨 정의에 쓰인 피처를 뺀 버전도 병행
보고할 수 있다(9/2 피드백: 라벨 누수 방지 병행 보고 요청).

사기 유형 부분집합은 겹침을 허용하는 불리언 플래그(`type_new`, `type_burst_kde`)로
독립적으로 뽑는다(`scripts/04_add_burst_kde.py` 주석 참고: 상호배타 `type_class`는
참고용일 뿐, 실제 실험은 각 플래그를 따로 씀). `--type-class burst`/`mixed`/`general`은
`type_burst_kde` 컬럼이 필요하므로, 먼저 `python scripts/04_add_burst_kde.py` 로
버스트형 재정의를 `reviews.parquet` 에 병합해야 한다.

사용 예:
    python scripts/05_train_gnn_condition_a.py --type-class new
    python scripts/05_train_gnn_condition_a.py --type-class new --exclude-cols singleton
    python scripts/05_train_gnn_condition_a.py --type-class burst --backbone sage
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loaders import load_handcrafted_features, load_reviews  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402
from src.features.structural.graph import (  # noqa: E402
    build_rur_adjacency,
    gcn_normalize,
    mean_normalize,
    scipy_to_torch_sparse,
)
from src.models.gcn import VanillaGNN  # noqa: E402

RESULTS_DIR = ROOT / "results" / "condition_a"


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type-class", default="new",
                    choices=["new", "burst", "mixed", "general", "all"],
                    help="사기 서브유형 (기본: new = 신규계정형, 1순위). "
                         "겹침 허용 플래그(type_new/type_burst_kde) 기준: "
                         "new=type_new / burst=type_burst_kde / "
                         "mixed=둘 다 / general=둘 다 아님")
    ap.add_argument("--frac", type=float, default=1.0,
                    help="작성자 단위 표본 비율 (기본 1.0 = 전체). 빠른 반복 실험용")
    ap.add_argument("--year", type=int, default=2014,
                    help="분석 대상 연도 (기본 2014 — 9/9 팀 결정: 임의 샘플링 대신 "
                         "연도 필터링. 0을 주면 연도 필터 없이 전체 기간 사용)")
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
                    help="결과 JSON 저장 경로 (기본: results/condition_a/<type>_<backbone>.json)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.time()
    try:
        reviews = load_reviews(
            columns=["review_id", "user_id", "fraud", "type_new", "type_burst_kde", "year"])
        has_burst_kde = True
    except Exception:
        reviews = load_reviews(columns=["review_id", "user_id", "fraud", "type_new", "year"])
        has_burst_kde = False

    feats = load_handcrafted_features()
    if len(reviews) != len(feats):
        raise ValueError(f"행 수 불일치: reviews {len(reviews)} != features {len(feats)}")

    if args.type_class == "all":
        type_mask = np.ones(len(reviews), dtype=bool)
    elif args.type_class == "new":
        type_mask = reviews["type_new"].to_numpy() == 1
    else:
        if not has_burst_kde:
            raise ValueError(
                "type_burst_kde 컬럼이 없습니다. 먼저 "
                "`python scripts/04_add_burst_kde.py` 로 버스트형 재정의를 "
                "reviews.parquet 에 병합하세요."
            )
        is_new = reviews["type_new"].to_numpy() == 1
        is_burst = reviews["type_burst_kde"].to_numpy() == 1
        if args.type_class == "burst":
            type_mask = is_burst
        elif args.type_class == "mixed":
            type_mask = is_new & is_burst
        else:  # general
            type_mask = ~is_new & ~is_burst

    sample_mask = subsample_users(reviews["user_id"].to_numpy(), args.frac, args.seed)
    year_mask = (reviews["year"].to_numpy() == args.year) if args.year else np.ones(len(reviews), dtype=bool)
    keep = type_mask & sample_mask & year_mask
    n_kept = int(keep.sum())
    if n_kept < 100:
        raise ValueError(f"부분집합이 너무 작습니다 (n={n_kept}). --frac 을 키우거나 --year 0(전체 기간)을 확인하세요.")

    user_ids = reviews.loc[keep, "user_id"].to_numpy()
    fraud = reviews.loc[keep, "fraud"].to_numpy().astype("float32")

    feat_cols = [c for c in feats.columns if c != "review_id" and c not in args.exclude_cols]
    X = feats.loc[keep, feat_cols].to_numpy(dtype="float32")
    print(f"[설정] type_class={args.type_class} year={args.year or '전체'} frac={args.frac} "
          f"n={n_kept:,} 사기율={fraud.mean():.1%} 피처={len(feat_cols)}개 backbone={args.backbone}")

    print("[1/4] R-U-R 그래프 구성")
    adj = build_rur_adjacency(user_ids)
    n_edges = adj.nnz
    isolated = int((np.asarray(adj.sum(axis=1)).flatten() == 0).sum())
    print(f"      엣지 {n_edges:,}개 | 고립 노드(엣지 0개) {isolated:,}개 "
          f"({isolated/n_kept:.1%}) | {time.time()-t0:.0f}초")

    norm_fn = gcn_normalize if args.backbone == "gcn" else mean_normalize
    adj_norm = scipy_to_torch_sparse(norm_fn(adj))

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
        "type_class": args.type_class,
        "type_definition": "type_new/type_burst_kde flags (겹침 허용, scripts/04_add_burst_kde.py)",
        "backbone": args.backbone,
        "year": args.year or "all",
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
    out = args.out or (RESULTS_DIR / f"{args.type_class}_{args.backbone}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장: {out} | 총 {time.time()-t0:.0f}초")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
