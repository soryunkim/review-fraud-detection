"""
12_type_intensity_gain.py — 유형 '강도'별 관계 정보 기여 (오동진, 2026-09-22)

버스트형·평점이탈형을 5% 꼬리 같은 작은 칸으로 자르면 test filtered 리뷰가 수십 건뿐이라
칸 단위 PR-AUC 비교로는 판단이 안 됐다. 여기서는 유형을 '있다/없다'가 아니라 연속적인
강도로 보고, 강도 사분위 구간마다 그래프 기여(GNN − MLP)를 잰다. 구간은 test 리뷰 전체를
나누므로 표본이 수백~수천 건 단위가 된다.

  강도
    burst    = −log10(max(p_life, p_90))  — 09의 교집합 정의(두 검정 모두 유의)를 연속값으로.
               09와 같은 평가 가능 조건(elapsed≥1, 직전90일 기준선 ≥30일)을 만족하는 리뷰만.
    dev      = rating − (그 리뷰 이전 가게 평균)  — 09 v2와 같은 정의, 과거 20건 이상만.
    absdev   = |dev|,  down = −dev (dev<0만),  up = dev (dev>0만)
  기여
    구간별 ΔAP, ΔAUC = (GNN − MLP), seed 5개 평균. 사기 비율이 구간마다 달라도 구간 안에서
    두 모델을 같은 리뷰로 비교하므로 짝지은 차이다.
  추세
    Q4 − Q1 = (상위 25% 구간의 Δ) − (하위 25% 구간의 Δ).
  신뢰구간
    상점 단위 cluster bootstrap(같은 상점 리뷰는 버스트 여부가 함께 움직이므로 리뷰 단위로
    뽑으면 CI가 과하게 좁아진다). 매 반복에서 seed도 복원추출.

사용 예
  python scripts/12_type_intensity_gain.py                 # 기본: low/high × rur/rsr/rtr
  python scripts/12_type_intensity_gain.py --groups high --relations rtr
출력: results/type_intensity/<group>_<rel>.json, results/type_intensity/summary.csv
      data/processed/type_intensity.parquet (강도 캐시, gitignore 대상)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REVIEWS = ROOT / "data" / "processed" / "reviews.parquet"
NODES = ROOT / "data" / "processed" / "graph_2014" / "nodes.parquet"
INTENSITY = ROOT / "data" / "processed" / "type_intensity.parquet"
SCORES = ROOT / "data" / "processed" / "scores"
OUT_DIR = ROOT / "results" / "type_intensity"
SEEDS = [42, 0, 1, 2, 3]


def load_09():
    spec = importlib.util.spec_from_file_location(
        "burst09", ROOT / "scripts" / "09_burst_trial_exploration.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_intensity() -> pd.DataFrame:
    """09의 함수를 그대로 써서 정의를 공유한다(임계값만 걷어낸 연속값)."""
    m09 = load_09()
    df = pd.read_parquet(REVIEWS, columns=["review_id", "prod_id", "date", "rating", "year"])
    df["date"] = pd.to_datetime(df["date"])
    ts = m09.compute_time_signal(df)
    ok = (ts["elapsed"] >= 1) & (ts["base90_days"] >= 30)
    p = np.maximum(ts["pval_life"].to_numpy(), ts["pval_90"].to_numpy())
    ts["burst"] = np.where(ok, -np.log10(np.clip(p, 1e-300, 1.0)), np.nan)

    w = df.sort_values(["prod_id", "date", "review_id"]).reset_index(drop=True)
    w["cum_n"] = w.groupby("prod_id").cumcount()
    w["cum_sum"] = w.groupby("prod_id")["rating"].cumsum() - w["rating"]
    w["past_avg"] = w["cum_sum"] / w["cum_n"].replace(0, np.nan)
    w["dev"] = np.where(w["cum_n"] >= m09.MIN_PAST_REVIEWS, w["rating"] - w["past_avg"], np.nan)

    out = w[["review_id", "prod_id", "dev"]].merge(ts[["review_id", "burst"]], on="review_id")
    out.to_parquet(INTENSITY, index=False)
    return out


def score_files(group: str, rel: str) -> tuple[list[Path], list[Path]]:
    g = [SCORES / "gnn" / f"{group}_{rel}_sage_full_timesplit{'' if s == 42 else f'_seed{s}'}.parquet"
         for s in SEEDS]
    m = [SCORES / "mlp" / f"{group}_all_seed{s}_timesplit.parquet" for s in SEEDS]
    return g, m


def load_test(files: list[Path]) -> pd.DataFrame:
    cols = {}
    for i, f in enumerate(files):
        s = pd.read_parquet(f)
        cols[i] = s[s["split"] == "test"].set_index("review_id")["score"]
    return pd.DataFrame(cols).sort_index()


# 가중 AP/AUC: 정렬 순서는 seed마다 고정이고, bootstrap 반복마다 가중치(복원추출 횟수)만 바뀐다.
def w_ap(order: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    yo, wo = y[order], w[order]  # 점수 내림차순
    tp = np.cumsum(wo * yo)
    fp = np.cumsum(wo * (1 - yo))
    if tp[-1] <= 0:
        return np.nan
    prec = tp / np.maximum(tp + fp, 1e-12)
    return float(np.sum(wo * yo * prec) / tp[-1])


def w_auc(order: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    yo, wo = y[order][::-1], w[order][::-1]  # 점수 오름차순
    neg = wo * (1 - yo)
    neg_before = np.cumsum(neg) - neg
    W1, W0 = np.sum(wo * yo), np.sum(neg)
    if W1 <= 0 or W0 <= 0:
        return np.nan
    return float(np.sum(wo * yo * neg_before) / (W1 * W0))


def bin_delta(A, B, y, w, seeds):
    """구간 하나의 seed 평균 ΔAP, ΔAUC. A, B: seed별 내림차순 order 리스트."""
    dap = np.mean([w_ap(A[s], y, w) - w_ap(B[s], y, w) for s in seeds])
    dauc = np.mean([w_auc(A[s], y, w) - w_auc(B[s], y, w) for s in seeds])
    return dap, dauc


def analyze(group: str, rel: str, inten: pd.DataFrame, labels: pd.Series,
            n_boot: int, rng: np.random.Generator) -> dict:
    gf, mf = score_files(group, rel)
    missing = [f.name for f in gf + mf if not f.exists()]
    if missing:
        return {"group": group, "relation": rel, "skipped": f"점수 파일 없음: {missing}"}
    GA, MB = load_test(gf), load_test(mf)
    if not GA.index.equals(MB.index):
        raise ValueError("GNN/MLP test 리뷰 집합이 다릅니다")
    d = inten.set_index("review_id").reindex(GA.index)
    y_all = labels.reindex(GA.index).to_numpy(dtype=np.float64)
    prod_all = d["prod_id"].to_numpy()
    dev = d["dev"].to_numpy()

    meas = {
        "burst": d["burst"].to_numpy(),
        "absdev": np.abs(dev),
        "down": np.where(dev < 0, -dev, np.nan),
        "up": np.where(dev > 0, dev, np.nan),
    }
    res = {"group": group, "relation": rel, "n_test": int(len(GA)),
           "test_filtered_rate": float(y_all.mean()), "n_boot": n_boot, "measures": {}}
    S = len(SEEDS)
    a_all, b_all = GA.to_numpy(), MB.to_numpy()
    for name, x in meas.items():
        ok = ~np.isnan(x)
        xs, ys, ps = x[ok], y_all[ok], prod_all[ok]
        a, b = a_all[ok], b_all[ok]
        # 동률이 많은 값(burst의 0 근처 등)은 순위로 끊어 네 구간 크기를 맞춘다
        q = pd.qcut(pd.Series(xs).rank(method="first"), 4, labels=False).to_numpy()
        uniq, pinv = np.unique(ps, return_inverse=True)
        bins, orders = [], []
        for k in range(4):
            sel = q == k
            A = [np.argsort(-a[sel, s], kind="stable") for s in range(S)]
            B = [np.argsort(-b[sel, s], kind="stable") for s in range(S)]
            orders.append((sel, A, B))
            dap, dauc = bin_delta(A, B, ys[sel], np.ones(sel.sum()), range(S))
            bins.append({"bin": k + 1, "range": [float(xs[sel].min()), float(xs[sel].max())],
                         "n": int(sel.sum()), "n_filtered": int(ys[sel].sum()),
                         "filtered_rate": float(ys[sel].mean()), "dAP": float(dap), "dAUC": float(dauc)})
        boot = np.empty((n_boot, 4, 2))
        for r in range(n_boot):
            cnt = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)))
            wall = cnt[pinv].astype(np.float64)
            sd = rng.integers(0, S, size=S)
            for k, (sel, A, B) in enumerate(orders):
                boot[r, k] = bin_delta(A, B, ys[sel], wall[sel], sd)
        for k in range(4):
            for j, key in enumerate(["dAP", "dAUC"]):
                v = boot[:, k, j]
                bins[k][key + "_ci"] = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
        trend = {}
        for j, key in enumerate(["dAP", "dAUC"]):
            v = boot[:, 3, j] - boot[:, 0, j]
            trend[key] = {"est": bins[3][key] - bins[0][key],
                          "ci": [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]}
        res["measures"][name] = {"n": int(ok.sum()), "bins": bins, "q4_minus_q1": trend}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["low", "high"])
    ap.add_argument("--relations", nargs="+", default=["rur", "rsr", "rtr"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--rebuild", action="store_true", help="강도 캐시 다시 계산")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    inten = pd.read_parquet(INTENSITY) if INTENSITY.exists() and not args.rebuild else build_intensity()
    labels = pd.read_parquet(NODES, columns=["review_id", "fraud"]).set_index("review_id")["fraud"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for g in args.groups:
        for rel in args.relations:
            res = analyze(g, rel, inten, labels, args.n_boot, np.random.default_rng(args.seed))
            (OUT_DIR / f"{g}_{rel}.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            if "skipped" in res:
                print(f"[skip] {g} {rel}: {res['skipped']}")
                continue
            for mname, m in res["measures"].items():
                for b in m["bins"]:
                    rows.append({"group": g, "relation": rel, "measure": mname, "bin": b["bin"],
                                 "lo": b["range"][0], "hi": b["range"][1], "n": b["n"],
                                 "n_filtered": b["n_filtered"], "filtered_rate": b["filtered_rate"],
                                 "dAP": b["dAP"], "dAP_lo": b["dAP_ci"][0], "dAP_hi": b["dAP_ci"][1],
                                 "dAUC": b["dAUC"], "dAUC_lo": b["dAUC_ci"][0], "dAUC_hi": b["dAUC_ci"][1]})
                t = m["q4_minus_q1"]["dAP"]
                print(f"{g:4s} {rel} {mname:6s} n={m['n']:5d}  dAP Q1..Q4 = "
                      + " ".join(f"{b['dAP']:+.3f}" for b in m["bins"])
                      + f"   Q4-Q1 {t['est']:+.3f} [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}]", flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
