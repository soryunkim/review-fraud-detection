"""
R-U-R(같은 작성자) 관계 그래프 구성.

`docs/데이터_확인결과_오동진.md` §6: R-T-R(같은 업체+같은 달)·R-S-R(같은 업체+같은
평점)의 라벨 동질성은 무작위 기대치와 같아(0.05) 판별력이 없고, R-U-R만
유의미하다(0.90). 조건 A/B GNN은 R-U-R 하나만 쓴다.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def build_rur_adjacency(user_ids: np.ndarray) -> sp.csr_matrix:
    """같은 user_id를 공유하는 노드끼리 연결한 대칭 인접행렬(자기 자신 제외).

    user_ids 의 인덱스가 곧 그래프 노드 번호다(0..N-1). 호출 전에 원하는
    부분집합으로 이미 잘라낸 배열을 넘길 것 — 이 함수는 넘어온 배열 안에서만
    엣지를 만든다(부분집합 밖의 리뷰와는 연결하지 않음).
    """
    n = len(user_ids)
    order = np.argsort(user_ids, kind="stable")
    sorted_users = user_ids[order]
    boundaries = np.flatnonzero(np.diff(sorted_users)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [n]))

    rows, cols = [], []
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        idx = order[s:e]
        a, b = np.meshgrid(idx, idx)
        mask = a != b
        rows.append(a[mask])
        cols.append(b[mask])

    if not rows:
        return sp.csr_matrix((n, n), dtype="float32")
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.ones(len(rows), dtype="float32")
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def gcn_normalize(adj: sp.csr_matrix) -> sp.csr_matrix:
    """Kipf & Welling (2017) 대칭 정규화: D^-1/2 (A+I) D^-1/2."""
    n = adj.shape[0]
    adj = adj + sp.eye(n, format="csr", dtype="float32")
    deg = np.asarray(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    deg_inv_sqrt[nz] = np.power(deg[nz], -0.5)
    d = sp.diags(deg_inv_sqrt)
    return (d @ adj @ d).tocsr().astype("float32")


def mean_normalize(adj: sp.csr_matrix) -> sp.csr_matrix:
    """GraphSAGE(mean) 이웃 집계용 행 정규화: D^-1 A (자기 루프 없음).

    자기 자신은 SAGELayer 가 별도 선형변환으로 더하므로 여기서는 순수 이웃만 평균낸다.
    이웃이 없는 노드(싱글턴 작성자)는 0 벡터가 된다.
    """
    deg = np.asarray(adj.sum(axis=1)).flatten()
    deg_inv = np.zeros_like(deg)
    nz = deg > 0
    deg_inv[nz] = 1.0 / deg[nz]
    d = sp.diags(deg_inv)
    return (d @ adj).tocsr().astype("float32")


def scipy_to_torch_sparse(mat: sp.csr_matrix):
    """torch 는 여기서만 임포트한다 — 그래프 구성 자체는 torch 없이도 쓸 수 있게."""
    import torch

    coo = mat.tocoo()
    idx = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
    val = torch.from_numpy(coo.data).float()
    return torch.sparse_coo_tensor(idx, val, coo.shape).coalesce()
