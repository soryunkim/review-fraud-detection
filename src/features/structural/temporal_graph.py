"""
시간 순서를 보존하는(과거 방향) 관계 그래프.

각 리뷰는 **자기보다 먼저 작성된** 같은 관계의 리뷰에서만 메시지를 받는다.
미래 리뷰의 피처가 엣지를 통해 흘러들어오는 누수를 막기 위함이다
(9/9 지도교수 피드백: "해당 리뷰 작성 시점까지의 이력", "그 이후 것을 넣으면 안 됨").

인접행렬 규약: A[i, j] = 1  ⇔  j 가 i 보다 먼저 작성됨 (j → i 로 메시지 전달)
  → 행 i 는 i 가 받는 과거 이웃. `A @ X` 가 각 노드에 과거 이웃을 모은다.
  → 대칭이 아니므로 대칭 정규화(D^-1/2 A D^-1/2) 대신 행 정규화(D^-1 A)를 쓸 것.

"먼저"의 기준은 (date, review_id) 사전식 순서다. 같은 날 작성된 리뷰는 review_id 로
순서를 정하며, 이는 `scripts/01_build_reviews.py` 의 as-of 누적 순번과 동일한 규칙이다
(라벨과 엣지가 같은 시간 순서를 쓰도록 일치시킴).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

RELATIONS = {
    "rur": ["user_id"],             # 같은 작성자
    "rsr": ["prod_id", "rating"],   # 같은 업체 + 같은 평점
    "rtr": ["prod_id", "month"],    # 같은 업체 + 같은 달
}


def past_only_adjacency(nodes: pd.DataFrame, keys: list[str]) -> sp.csr_matrix:
    """같은 key 그룹 안에서 '먼저 쓴 리뷰 → 나중 리뷰' 방향 엣지만 만든다.

    nodes 는 0..n-1 로컬 인덱스를 가진 DataFrame 이어야 하며 date, review_id, keys 컬럼 필요.
    """
    n = len(nodes)
    sort_cols = keys + ["date", "review_id"]
    order = np.lexsort([nodes[c].to_numpy() for c in reversed(sort_cols)])
    key_arr = nodes[keys].to_numpy()[order]

    if len(keys) == 1:
        change = key_arr[1:, 0] != key_arr[:-1, 0]
    else:
        change = (key_arr[1:] != key_arr[:-1]).any(axis=1)
    bounds = np.r_[0, np.flatnonzero(change) + 1, n]

    rows, cols = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        m = b - a
        if m < 2:
            continue
        idx = order[a:b]                          # 시간순 정렬된 로컬 인덱스
        src, dst = np.triu_indices(m, k=1)        # src < dst  ⇒  src 가 먼저
        rows.append(idx[dst].astype(np.int32))    # 받는 쪽(나중)
        cols.append(idx[src].astype(np.int32))    # 보내는 쪽(먼저)

    if not rows:
        return sp.csr_matrix((n, n), dtype=np.float32)
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    return sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(n, n))


def row_normalize(adj: sp.csr_matrix, self_loop: bool = True) -> sp.csr_matrix:
    """방향 그래프용 행 정규화: D^-1 (A + I). 각 노드는 자기 + 과거 이웃의 평균을 받는다."""
    if self_loop:
        adj = adj + sp.eye(adj.shape[0], format="csr", dtype=np.float32)
    deg = np.asarray(adj.sum(axis=1)).ravel()
    inv = np.zeros_like(deg)
    nz = deg > 0
    inv[nz] = 1.0 / deg[nz]
    return (sp.diags(inv) @ adj).tocsr().astype(np.float32)


def combine(adjs: dict[str, sp.csr_matrix], names: list[str]) -> sp.csr_matrix:
    """여러 관계를 합친다(이진 OR). 7가지 조합 실험용."""
    out = None
    for nm in names:
        out = adjs[nm] if out is None else out + adjs[nm]
    out.data[:] = 1.0
    return out
