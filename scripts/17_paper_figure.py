"""
17_paper_figure.py — 논문 그림 1: 이력·관계별 관계 정보 기여 (오동진, 2026-10-02)

9/23 미팅에서 교수님이 "이 그림(슬라이드 5)을 논문에 넣을 것"이라고 지시한 그림을,
최종 수치(rolling 3회 + 동률 재분류)로 다시 그린다. 값은 results/bootstrap 의
부트스트랩 JSON에서 직접 읽어 수작업 전사 오류를 막는다.

출력: docs/paper/fig1_contribution_history.png
(10/15 김소륜: 교수님 수정본에 맞춰 저활동/비저활동 → 이력 미보유/보유, 가게 → 상점으로 라벨 변경.
 이전 라벨 그림은 fig1_contribution.png 로 남겨 둠)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BOOT = ROOT / "results" / "bootstrap"
OUT = ROOT / "docs" / "paper" / "fig1_contribution_history.png"

# (라벨, 부트스트랩 파일, 그룹)
ROWS = [
    ("이력 보유 · 같은 작성자", "high_rur_sage_vs_mlp37_rolling_sdf", "high"),
    ("이력 보유 · 같은 상점·평점", "high_rsr_sage_vs_mlp37_rolling_sdf", "high"),
    ("이력 보유 · 같은 상점·달", "high_rtr_sage_vs_mlp37_rolling_sdf", "high"),
    ("이력 미보유 · 같은 작성자", "low_rur_sage_vs_mlp37_rolling_sdf", "low"),
    ("이력 미보유 · 같은 상점·평점", "low_rsr_sage_vs_mlp37_rolling_sdf", "low"),
    ("이력 미보유 · 같은 상점·달", "low_rtr_sage_vs_mlp37_rolling_sdf", "low"),
]


def load(name: str) -> tuple[float, float, float]:
    d = json.loads((BOOT / f"{name}.json").read_text(encoding="utf-8"))
    lo, hi = d["ci95_delta_seed_and_test"]
    return d["delta_mean"], lo, hi


def main() -> None:
    plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "DejaVu Sans"]  # Windows / macOS, − 기호 대체
    plt.rcParams["axes.unicode_minus"] = False

    labels, vals, los, his = [], [], [], []
    for lab, f, _ in ROWS:
        v, lo, hi = load(f)
        labels.append(lab)
        vals.append(v)
        los.append(v - lo)
        his.append(hi - v)

    fig, ax = plt.subplots(figsize=(3.4, 2.3), dpi=300)
    y = range(len(labels))[::-1]
    # 색은 유의성 기준 — 95% CI가 0을 포함하면 회색(v2, 김소륜 10/3)
    colors = ["#9a9a9a" if v - lo <= 0 <= v + hi else ("#2f6f4e" if v > 0 else "#9b3b3b")
              for v, lo, hi in zip(vals, los, his)]
    ax.barh(list(y), vals, height=0.6, color=colors, alpha=0.85)
    ax.errorbar(vals, list(y), xerr=[los, his], fmt="none", ecolor="#333333",
                elinewidth=0.8, capsize=2.5)
    ax.axvline(0, color="#000000", linewidth=0.8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("관계 정보의 기여 (ΔPR-AUC)", fontsize=7)
    ax.tick_params(axis="x", labelsize=6.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for i, v, lo, hi in zip(y, vals, los, his):  # 값 표기는 오차막대 바깥에
        x = v + hi + 0.004 if v >= 0 else v - lo - 0.004
        ax.text(x, i, f"{v:+.3f}".replace("-", "\u2212"), va="center",
                ha="left" if v >= 0 else "right", fontsize=6)
    ax.set_xlim(-0.11, 0.16)
    fig.tight_layout(pad=0.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print("saved", OUT)
    for lab, v, lo, hi in zip(labels, vals, los, his):
        print(f"  {lab}: {v:+.4f} [{v - lo:+.4f}, {v + hi:+.4f}]")


if __name__ == "__main__":
    main()
