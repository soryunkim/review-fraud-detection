"""이진 사기 분류 평가 지표 (AUROC / AUPRC / 최적 임계값 F1)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def evaluate(y_true: np.ndarray, scores: np.ndarray) -> dict:
    auroc = roc_auc_score(y_true, scores)
    auprc = average_precision_score(y_true, scores)
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    best = int(np.argmax(f1))
    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "best_f1": float(f1[best]),
        "best_threshold": float(thr[best]) if best < len(thr) else 1.0,
        "n": int(len(y_true)),
        "n_fraud": int(y_true.sum()),
    }
