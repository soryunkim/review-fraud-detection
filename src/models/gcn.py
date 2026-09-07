"""
바닐라 GNN 백본 — 조건 A/B 공용 (GCN, GraphSAGE-mean).

9/2 미팅 피드백: CARE-GNN(강화학습 기반)은 성능 저하 원인을 "RL 문제"인지
"텍스트 신호 부족"인지 분리하기 어려워 제외. 가장 순수한 구조 신호 기준선으로
바닐라 GCN/GraphSAGE부터 적용한다. 불균형 보정(PC-GNN)은 다음 단계.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(adj_norm, self.lin(x))


class SAGELayer(nn.Module):
    """평균 집계 GraphSAGE. adj_mean 은 자기 루프 없는 행정규화 인접행렬이어야 한다."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj_mean: torch.Tensor) -> torch.Tensor:
        neigh = torch.sparse.mm(adj_mean, x)
        return self.lin_self(x) + self.lin_neigh(neigh)


class VanillaGNN(nn.Module):
    """2-layer GCN 또는 GraphSAGE + 이진 분류 헤드(로짓 1개)."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.3,
                backbone: str = "gcn"):
        super().__init__()
        if backbone not in ("gcn", "sage"):
            raise ValueError(f"알 수 없는 backbone: {backbone}")
        Layer = GCNLayer if backbone == "gcn" else SAGELayer
        self.conv1 = Layer(in_dim, hidden_dim)
        self.conv2 = Layer(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, adj_norm))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, adj_norm))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.out(h).squeeze(-1)
