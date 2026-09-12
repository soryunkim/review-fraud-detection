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
비저활동(`--group high`)으로 비교한다(오동진, 2026-09-11 확정).

`--scope`로 그래프 범위를 두 가지 중 골라 **둘 다 돌려서 비교**할 수 있다
(2026-09-11, 팀 요청):
  - `isolated`(기본): 지정한 그룹 노드만 그래프에 남긴다(2014년 전체 그래프를
    그 그룹의 노드로 재인덱싱) — 그룹 안에서만 통하는 관계의 순수 효과를 잰다.
    기존 매트릭스 설계(유형별 독립 미니 실험)와 일관됨.
  - `full`: 2014년 전체 그래프(그룹 밖 이웃 포함)를 그대로 쓰되, 학습(loss)·
    val·test는 여전히 지정한 그룹 노드로만 계산한다 — 그룹 밖 이웃의 신호까지
    포함했을 때 얼마나 더 도움되는지를 잰다("이웃 데이터가 많아진 효과"까지 포함).
  두 scope의 차이 = "관계 자체의 순수 효과" vs "그 관계로 도달 가능한 전체
  이웃 데이터의 효과". 둘 다 실행해 비교하는 것을 권장.

피처는 `scripts/07_build_features_asof_2014.py`(오동진, 2026-09-11)가 만든
as-of Rayana Table 2 피처 37개(`features_rayana_asof_2014.parquet`)를 쓴다
— 미래 정보 없이 그 리뷰 작성 시점까지의 이력만으로 계산됨.

⚠️ `--exclude-cols`: 라벨 정의에 쓰인 피처를 빼는 병행 보고용(9/2 피드백).
`ISR`은 이제 "작성자의 첫 리뷰(=type_new)"와 동일해 `--group low/high`
안에서는 상수(그룹 내 모든 행이 같은 값)가 되므로 사실상 자동으로 무해하지만,
`--group all`로 두 그룹을 섞어 돌릴 때는 `--exclude-cols ISR`로 빼는 것을 권장.

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

from src.data.loaders import load_rayana_asof_features  # noqa: E402
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


def user_stratified_split(user_ids: np.ndarray, fraud: np.ndarray, seed: int,
                          val_frac: float = 0.15, test_frac: float = 0.15
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """작성자 단위 분할 — 한 작성자의 리뷰는 train/val/test 중 한 곳에만 들어간다.

    리뷰 단위 무작위 분할(`stratified_split`)은 같은 작성자의 다른 리뷰가 train 에
    들어가므로, 모델이 "이 작성자는 사기꾼"을 외워서 맞히는 효과가 성능에 섞인다
    (`results/mlp_baseline/README.md` 주의사항 3번). 비저활동형에서 작성자 수준
    피처와 R-U-R 의 높은 점수가 이 효과에 오염됐는지 검증하기 위한 분할 방식이다.

    사기 리뷰를 가진 작성자 / 없는 작성자 두 층으로 나눈 뒤 각 층에서 작성자를 섞어
    리뷰 수 기준 쿼터(test 15% → val 15% → 나머지 train)를 채운다. 리뷰 단위 분할만큼
    사기율이 정확히 맞지는 않으므로, 호출부에서 실제 분할 사기율을 함께 기록한다.
    저활동형은 작성자당 리뷰가 1건이라 이 분할이 리뷰 단위 분할과 사실상 같다(대조군).
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(user_ids, return_inverse=True)
    cnt = np.bincount(inv, minlength=len(uniq))
    fcnt = np.bincount(inv, weights=fraud, minlength=len(uniq))
    assign = np.zeros(len(uniq), dtype=np.int8)          # 0=train, 1=val, 2=test
    for stratum in (True, False):                        # 사기 보유 작성자부터
        idx_u = np.flatnonzero((fcnt > 0) == stratum)
        rng.shuffle(idx_u)
        total = cnt[idx_u].sum()
        q_test, q_val = total * test_frac, total * val_frac
        acc_t = acc_v = 0
        for u in idx_u:
            if acc_t < q_test:
                assign[u] = 2
                acc_t += cnt[u]
            elif acc_v < q_val:
                assign[u] = 1
                acc_v += cnt[u]
            else:
                assign[u] = 0
    a = assign[inv]
    return a == 0, a == 1, a == 2


TIME_SPLIT_MONTHS = {"train": range(1, 10), "val": (10, 11), "test": (12,)}


def temporal_split(dates: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """시간 단위 분할 — 2014년 1~9월 train / 10~11월 val / 12월 test (오동진, 2026-09-12).

    "미래 정보만 차단"한 분할로, 배포 상황(과거로 학습해 새 리뷰를 판정)에 가장 가깝고
    "작성 시점까지의 이력만" 원칙과 맞는다 → 주 결과 보고용(9/12 보고서 §10.1).
    같은 작성자의 과거 리뷰가 train 에 있는 것은 허용된다(배포 시에도 알 수 있는 정보).
    그래프가 과거 방향이라 train 노드는 val/test 기간 노드로부터 메시지를 받지 않는다.
    seed 와 무관하게 분할은 고정이다(seed 는 초기화·dropout 에만 영향).
    """
    m = pd.to_datetime(dates).dt.month.to_numpy()
    return (np.isin(m, list(TIME_SPLIT_MONTHS["train"])),
            np.isin(m, TIME_SPLIT_MONTHS["val"]),
            np.isin(m, TIME_SPLIT_MONTHS["test"]))


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


def sameday_tie_mask(nodes: pd.DataFrame) -> np.ndarray:
    """작성자의 '첫날'에 2건 이상 쓴 리뷰 — 저활동/비저활동 라벨이 같은 날 안에서
    review_id(=CSV 행 순서)로 갈리는 경계 리뷰들.

    2014년 18,172건이 해당하며 저활동 7,375 / 비저활동 10,797 로 나뉜다. 그런데
    사기율은 저활동(동률 아님) 20.9%, 저활동 동률 20.5%, 비저활동 동률 21.0%,
    비저활동 나머지 4.6% 로, 비저활동으로 분류된 10,797건이 사실상 저활동처럼
    행동한다. 게다가 이 10,797건은 **전부**(100%) R-U-R 과거 이웃이 같은 날
    리뷰뿐이다 — 즉 그 엣지의 시간 방향이 review_id 순서에만 의존한다.
    `--drop-sameday-ties` 로 이 리뷰들을 학습·평가 대상에서 빼고 결과가 어떻게
    달라지는지 확인한다(그래프에는 남으므로 이웃으로서의 역할은 유지)."""
    rev = pd.read_parquet(ROOT / "data" / "processed" / "reviews.parquet",
                          columns=["review_id", "user_id", "date"])
    d = pd.to_datetime(rev["date"]).dt.normalize()
    tie = (d == rev.groupby("user_id")["date"].transform("min").pipe(pd.to_datetime).dt.normalize()) &           (rev.groupby([rev.user_id, d])["review_id"].transform("size") > 1)
    m = dict(zip(rev["review_id"].to_numpy(), tie.to_numpy()))
    return nodes["review_id"].map(m).fillna(False).to_numpy().astype(bool)


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
    ap.add_argument("--scope", default="isolated", choices=["isolated", "full"],
                    help="isolated(기본): 그룹 부분그래프만 사용 — 그룹 내부 관계만의 순수 효과. "
                         "full: 2014년 전체 그래프(그룹 밖 이웃 포함)를 쓰되 학습/평가는 그룹으로 "
                         "한정 — 이웃 데이터가 많아지는 효과까지 포함. 둘 다 돌려서 비교 가능.")
    ap.add_argument("--backbone", default="gcn", choices=["gcn", "sage"])
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20,
                    help="val AUROC 개선 없이 버틸 최대 epoch 수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop-sameday-ties", action="store_true",
                    help="작성자 첫날 동률 리뷰(2014년 18,172건)를 학습·평가 대상에서 제외. "
                         "라벨이 review_id 순서로 갈리고 R-U-R 이웃이 전부 같은 날인 경계 리뷰들이라, "
                         "남은 신호가 여기서 오는지 확인하는 용도. 파일명에 _notie 가 붙는다.")
    ap.add_argument("--split", default="review", choices=["review", "user", "time"],
                    help="review(기본, 기존 결과와 동일): 리뷰 단위 무작위 분할. "
                         "user: 작성자 단위 분할 — 같은 작성자의 리뷰가 train/test 에 나뉘지 않게 해 "
                         "\"작성자 암기\" 효과를 제거(검증용). 결과 파일명에 _usersplit 이 붙는다. "
                         "time: 시간 단위 분할 — 1~9월 train / 10~11월 val / 12월 test(주 결과용). "
                         "결과 파일명에 _timesplit 이 붙는다.")
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
    feats = load_rayana_asof_features()
    feats_idx = feats.set_index("review_id")

    group_mask = build_group_mask(nodes, args.group, args.le2)
    if args.drop_sameday_ties:
        ties = sameday_tie_mask(nodes)
        n_drop = int((group_mask & ties).sum())
        group_mask = group_mask & ~ties
        print(f"      첫날 동률 리뷰 {n_drop:,}건을 대상에서 제외(그래프에는 유지)")
    sample_mask = subsample_users(nodes["user_id"].to_numpy(), args.frac, args.seed)
    target = group_mask & sample_mask   # 학습/평가 대상(그룹) — 항상 이 노드들로만 지도학습·채점
    n_target = int(target.sum())
    if n_target < 100:
        raise ValueError(f"부분집합이 너무 작습니다 (n={n_target}). --frac 을 키우세요.")

    # universe = 그래프에 실제로 올릴 노드 집합.
    #   isolated: target(그룹)만 그래프에 남긴다 — 그룹 밖 이웃은 아예 존재하지 않음.
    #   full    : 2014년 전체(표본 적용) 노드를 그래프에 남긴다 — 그룹 밖 이웃도 메시지를 보낼 수 있음.
    # 어느 쪽이든 loss·val·test는 target 노드로만 계산해 "무엇을 재는지"를 동일하게 유지한다.
    universe = target if args.scope == "isolated" else sample_mask
    n_universe = int(universe.sum())

    sub_nodes = nodes.loc[universe].reset_index(drop=True)
    target_local = target[universe]     # universe 로컬 좌표계에서 target 여부
    fraud_all = sub_nodes["fraud"].to_numpy().astype("float32")
    fraud = fraud_all[target_local]     # 리포트용 사기율은 target 기준

    feat_cols = [c for c in feats.columns if c != "review_id" and c not in args.exclude_cols]
    X = feats_idx.loc[sub_nodes["review_id"].to_numpy(), feat_cols].to_numpy(dtype="float32")

    group_label = f"{args.group}{'(<=2)' if args.le2 else ''}"
    print(f"[설정] group={group_label} scope={args.scope} relations={'+'.join(relations)} "
          f"frac={args.frac} n_target={n_target:,} n_universe={n_universe:,} "
          f"사기율(target)={fraud.mean():.1%} 피처={len(feat_cols)}개 backbone={args.backbone}")

    print("[1/4] 관계 그래프 결합 및 부분그래프 추출")
    combined = combine(adjs, relations)
    adj = subset_adjacency(combined, universe)
    n_edges = adj.nnz
    indeg = np.asarray(adj.sum(axis=1)).ravel()
    isolated = int((indeg[target_local] == 0).sum())
    print(f"      엣지 {n_edges:,}개(universe 기준) | target 중 과거 이웃 0개(고립) "
          f"{isolated:,}개 ({isolated/n_target:.1%}) | {time.time()-t0:.0f}초")

    self_loop = args.backbone == "gcn"
    adj_norm = scipy_to_torch_sparse(row_normalize(adj, self_loop=self_loop))

    print("[2/4] train/val/test 분할 (target 노드에서만, 사기율 유지, seed 고정)")
    target_idx_in_universe = np.flatnonzero(target_local)
    if args.split == "user":
        train_rel, val_rel, test_rel = user_stratified_split(
            sub_nodes.loc[target_local, "user_id"].to_numpy(), fraud, seed=args.seed)
    elif args.split == "time":
        train_rel, val_rel, test_rel = temporal_split(sub_nodes.loc[target_local, "date"])
    else:
        train_rel, val_rel, test_rel = stratified_split(fraud, seed=args.seed)
    train_idx = target_idx_in_universe[train_rel]
    val_idx = target_idx_in_universe[val_rel]
    test_idx = target_idx_in_universe[test_rel]
    print(f"      train {len(train_idx):,} / val {len(val_idx):,} / test {len(test_idx):,} "
          f"| 사기율 {fraud[train_rel].mean():.1%}/{fraud[val_rel].mean():.1%}/{fraud[test_rel].mean():.1%} "
          f"(split={args.split})")

    # float64 + 완화된 0-근사 판정(sigma < 1e-6): train 서브셋(예: 저활동형)에서는
    # 상수인 피처(ISR, U_MNR 등)가 float32 계산에서 정확히 0이 아닌 극소값(~4e-5)으로
    # 나올 수 있다. sigma == 0 만 걸러내면 이 극소값으로 나눠 scope=full처럼 표준화
    # 대상 밖 이웃(값이 실제로 다양한 비저활동 리뷰)의 피처가 최대 수만 배로 폭주한다
    # (오동진, 2026-09-11 — full scope epoch 1 loss 136 발견).
    mu = X[train_idx].astype("float64").mean(axis=0, keepdims=True)
    sigma = X[train_idx].astype("float64").std(axis=0, keepdims=True)
    sigma[sigma < 1e-6] = 1.0
    X = ((X - mu) / sigma).astype("float32")

    x_t = torch.from_numpy(X)
    y_t = torch.from_numpy(fraud_all)
    train_idx_t = torch.from_numpy(train_idx)

    pos = fraud_all[train_idx].sum()
    neg = len(train_idx) - pos
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
        loss = loss_fn(logits[train_idx_t], y_t[train_idx_t])
        loss.backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            logits = model(x_t, adj_norm)
            val_scores = torch.sigmoid(logits[val_idx]).numpy()
        val_metrics = evaluate(fraud_all[val_idx], val_scores)

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

    # 고정 임계값(Macro F1용): train 사기율 기준 상위 pos_rate 비율을 사기로 판정.
    # val/test 라벨을 보고 사후에 고르는 게 아니라 train 시점에 정해지는 값이라
    # best_f1(사후 최적 임계값)보다 실전에 가까운 평가다.
    train_pos_rate = float(fraud_all[train_idx].mean())
    val_final = evaluate(fraud_all[val_idx], scores[val_idx], pos_rate=train_pos_rate)
    test_final = evaluate(fraud_all[test_idx], scores[test_idx], pos_rate=train_pos_rate)
    print(f"      val  : AUROC {val_final['auroc']:.4f} | AUPRC {val_final['auprc']:.4f} "
          f"| best-F1 {val_final['best_f1']:.4f} | Macro-F1(고정) {val_final['macro_f1']:.4f}")
    print(f"      test : AUROC {test_final['auroc']:.4f} | AUPRC {test_final['auprc']:.4f} "
          f"| best-F1 {test_final['best_f1']:.4f} | Macro-F1(고정) {test_final['macro_f1']:.4f}")

    result = {
        "condition": "A",
        "group": args.group,
        "le2": args.le2,
        "scope": args.scope,
        "relations": relations,
        "graph_source": "data/processed/graph_2014 (과거 방향 temporal graph, 2026-09-11)",
        "backbone": args.backbone,
        "frac": args.frac,
        "n_target": n_target,
        "n_universe": n_universe,
        "n_edges": n_edges,
        "isolated_nodes_in_target": isolated,
        "fraud_rate_target": float(fraud.mean()),
        "n_features": len(feat_cols),
        "excluded_cols": args.exclude_cols,
        "epochs_run": epoch,
        "val": val_final,
        "test": test_final,
        "seed": args.seed,
        "split": args.split,
        "drop_sameday_ties": args.drop_sameday_ties,
        "fraud_rate_train": float(fraud[train_rel].mean()),
        "fraud_rate_test": float(fraud[test_rel].mean()),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    suffix = ({"user": "_usersplit", "time": "_timesplit"}.get(args.split, "")
              + ("_notie" if args.drop_sameday_ties else ""))
    out = args.out or (RESULTS_DIR /
                       f"{args.group}_{'-'.join(relations)}_{args.backbone}_{args.scope}{suffix}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장: {out} | 총 {time.time()-t0:.0f}초")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
