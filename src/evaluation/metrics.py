"""이진 사기 분류 평가 지표 (AUROC / AUPRC / 최적 임계값 F1 / 고정 임계값 Macro F1)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def evaluate(y_true: np.ndarray, scores: np.ndarray, pos_rate: float | None = None) -> dict:
    """
    pos_rate: 고정 임계값에 쓸 "양성으로 판정할 비율"(보통 train 사기율).
      test/val 라벨을 보지 않고 정하는 값이므로 사후 임계값 최적화(best_f1)와
      달리 "정직한" 평가가 된다 — score 상위 pos_rate 비율을 사기로 예측.
      None 이면 고정 임계값 지표(macro_f1 등)는 계산하지 않는다.
    """
    auroc = roc_auc_score(y_true, scores)
    auprc = average_precision_score(y_true, scores)
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    best = int(np.argmax(f1))
    result = {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "best_f1": float(f1[best]),
        "best_threshold": float(thr[best]) if best < len(thr) else 1.0,
        "n": int(len(y_true)),
        "n_fraud": int(y_true.sum()),
    }
    if pos_rate is not None:
        cutoff = float(np.quantile(scores, 1 - pos_rate))
        pred = (scores >= cutoff).astype(int)
        result["fixed_threshold"] = cutoff
        result["fixed_pos_rate_used"] = float(pos_rate)
        result["macro_f1"] = float(f1_score(y_true, pred, average="macro"))
        result["fraud_f1_fixed"] = float(f1_score(y_true, pred, pos_label=1, zero_division=0))
        result["normal_f1_fixed"] = float(f1_score(y_true, pred, pos_label=0, zero_division=0))
    return result
