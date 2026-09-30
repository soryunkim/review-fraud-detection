"""
16_graph_gain_regression.py — 그래프 기여의 회귀 분석 (9/21 제출요약 §8 질문2 선택지 ④)

칸을 나누지 않고, 비저활동형(high) 전체 test 리뷰에서 "그래프가 준 이득"
(Δscore = GNN 점수 − MLP 점수, seed 5개 평균, `15_review_level_gain.py`와 동일 정의)이
① 평점 이탈 폭(|rating − 과거 가게 평균|), ② 시간 집중도(버스트 강도,
−log10 p-value), ③ 작성자 이력 양(이 리뷰가 그 작성자의 몇 번째 리뷰인지, log1p)에
따라 달라지는지를 회귀로 본다. 칸으로 나누지 않으므로 표본 문제가 원천적으로 없다
(9/21 보고서 §8 질문2 선택지 ④ 설명 그대로) — 다만 유형별 비교표 형식과는 멀어진다.

⚠️ 범위: high(비저활동)형에서만 계산한다. low(저활동)형은 정의상 R-U-R 이웃이
없어 Δscore가 거의 0으로 수렴함이 이미 §3에서 확인됐고(별도 재확인 불필요),
low 그룹의 MLP 점수 파일(`data/processed/scores/mlp/low_all_seed*_timesplit.parquet`,
`--save-scores` 없이 실행됨)이 로컬에 없어 즉시 계산할 수 없다. "전체 18만 건" 회귀는
그 파일이 확보되면 이 스크립트를 `--groups low high`로 재실행해 확장할 수 있다.

회귀: OLS, Δscore ~ 표준화된 예측변수 (절편 포함). 신뢰구간은 상점 단위 cluster
bootstrap(다른 유형 분석 스크립트와 동일 관례) — 같은 가게 리뷰는 평점 이탈·버스트
여부가 함께 움직이므로 리뷰 단위 복원추출은 CI를 과소평가한다.
  1) 단변량: Δscore ~ absdev   (과거 가게 평균 리뷰 20건 이상인 리뷰만, 12 와 동일 정의)
  2) 단변량: Δscore ~ burst    (elapsed≥1 & 직전90일 기준선≥30일인 리뷰만, 12/09 와 동일)
  3) 단변량: Δscore ~ log1p(author_seq)  (모든 high 리뷰에 정의됨 — group 정의상 ≥1)
  4) 다변량: 위 세 예측변수 모두 정의된 교집합에서 함께

사용 예
    python scripts/16_graph_gain_regression.py                 # 기본 high × rur/rsr/rtr
출력: results/graph_gain_regression/<group>_<rel>.json
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
OUT_DIR = ROOT / "results" / "graph_gain_regression"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m12 = load_module("type_intensity12", "12_type_intensity_gain.py")   # score_files, load_test, build_intensity


def build_author_seq() -> pd.Series:
    """review_id -> 그 리뷰가 그 작성자의 몇 번째 리뷰인지(1부터, as-of). 12 의 prod_id
    cum_n 과 같은 패턴을 user_id 로 적용."""
    df = pd.read_parquet(REVIEWS, columns=["review_id", "user_id", "date"])
    df["date"] = pd.to_datetime(df["date"])
    w = df.sort_values(["user_id", "date", "review_id"]).reset_index(drop=True)
    w["author_seq"] = w.groupby("user_id").cumcount() + 1
    return w.set_index("review_id")["author_seq"]


def ols_beta(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """표준화된 x 에 대한 단순선형회귀 (intercept, slope)."""
    xm, ym = x.mean(), y.mean()
    xs = x.std()
    if xs < 1e-12:
        return float(ym), 0.0
    xz = (x - xm) / xs
    slope = np.sum(xz * (y - ym)) / np.sum(xz * xz)
    intercept = ym - slope * 0.0  # xz 평균은 0 이므로 절편 = ym
    return float(intercept), float(slope)


def ols_multi(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """표준화된 X(n,k)에 절편 열을 붙여 OLS 계수(k+1,) 반환. 첫 원소가 절편."""
    Xz = (X - X.mean(axis=0)) / X.std(axis=0)
    design = np.hstack([np.ones((len(y), 1)), Xz])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef


def regress_one(name: str, x: np.ndarray, y: np.ndarray, prod: np.ndarray,
                n_boot: int, rng: np.random.Generator) -> dict:
    intercept, slope = ols_beta(x, y)
    uniq, pinv = np.unique(prod, return_inverse=True)
    boot = np.empty(n_boot)
    group_idx = [np.flatnonzero(pinv == g) for g in range(len(uniq))]
    for r in range(n_boot):
        picked = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([group_idx[g] for g in picked])
        _, s = ols_beta(x[idx], y[idx])
        boot[r] = s
    return {
        "predictor": name, "n": int(len(y)),
        "slope_per_1sd": slope, "intercept": intercept,
        "slope_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "x_mean": float(x.mean()), "x_std": float(x.std()),
    }


def analyze(group: str, rel: str, inten: pd.DataFrame, author_seq: pd.Series,
            n_boot: int, rng: np.random.Generator) -> dict:
    gf, mf = m12.score_files(group, rel)
    missing = [f.name for f in gf + mf if not f.exists()]
    if missing:
        return {"group": group, "relation": rel, "skipped": f"점수 파일 없음: {missing}"}
    GA, MB = m12.load_test(gf), m12.load_test(mf)
    if not GA.index.equals(MB.index):
        raise ValueError("GNN/MLP test 리뷰 집합이 다릅니다")
    nodes = pd.read_parquet(NODES, columns=["review_id", "prod_id"]).set_index("review_id")
    prod_all = nodes["prod_id"].reindex(GA.index).to_numpy()
    delta_all = GA.to_numpy().mean(axis=1) - MB.to_numpy().mean(axis=1)

    d = inten.set_index("review_id").reindex(GA.index)
    absdev = np.abs(d["dev"].to_numpy())
    burst = d["burst"].to_numpy()
    log_hist = np.log1p(author_seq.reindex(GA.index).to_numpy(dtype=np.float64))

    out = {"group": group, "relation": rel, "n_test_total": int(len(GA)), "n_boot": n_boot, "predictors": {}}
    ok_dev = ~np.isnan(absdev)
    out["predictors"]["absdev"] = regress_one("absdev", absdev[ok_dev], delta_all[ok_dev], prod_all[ok_dev], n_boot, rng)
    ok_burst = ~np.isnan(burst)
    out["predictors"]["burst"] = regress_one("burst", burst[ok_burst], delta_all[ok_burst], prod_all[ok_burst], n_boot, rng)
    ok_hist = ~np.isnan(log_hist)
    out["predictors"]["log_author_seq"] = regress_one("log_author_seq", log_hist[ok_hist], delta_all[ok_hist],
                                                       prod_all[ok_hist], n_boot, rng)

    ok_all = ok_dev & ok_burst & ok_hist
    if ok_all.sum() >= 50:
        X = np.column_stack([absdev[ok_all], burst[ok_all], log_hist[ok_all]])
        y = delta_all[ok_all]
        coef = ols_multi(X, y)
        uniq, pinv = np.unique(prod_all[ok_all], return_inverse=True)
        group_idx = [np.flatnonzero(pinv == g) for g in range(len(uniq))]
        boot = np.empty((n_boot, 4))
        for r in range(n_boot):
            picked = rng.integers(0, len(uniq), len(uniq))
            idx = np.concatenate([group_idx[g] for g in picked])
            boot[r] = ols_multi(X[idx], y[idx])
        names = ["intercept", "absdev", "burst", "log_author_seq"]
        out["multivariate"] = {
            "n": int(ok_all.sum()),
            "coef": {n: float(c) for n, c in zip(names, coef)},
            "coef_ci": {n: [float(np.percentile(boot[:, i], 2.5)), float(np.percentile(boot[:, i], 97.5))]
                       for i, n in enumerate(names)},
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="+", default=["high"])
    ap.add_argument("--relations", nargs="+", default=["rur", "rsr", "rtr"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rebuild-intensity", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    inten_path = ROOT / "data" / "processed" / "type_intensity.parquet"
    inten = pd.read_parquet(inten_path) if inten_path.exists() and not args.rebuild_intensity else m12.build_intensity()
    author_seq = build_author_seq()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for g in args.groups:
        for rel in args.relations:
            res = analyze(g, rel, inten, author_seq, args.n_boot, np.random.default_rng(args.seed))
            (OUT_DIR / f"{g}_{rel}.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            if "skipped" in res:
                print(f"[skip] {g} {rel}: {res['skipped']}")
                continue
            print(f"\n{g} {rel} (n_test={res['n_test_total']:,})")
            for pname, p in res["predictors"].items():
                print(f"  {pname:16s} n={p['n']:6,d} slope/1sd {p['slope_per_1sd']:+.4f} "
                      f"[{p['slope_ci'][0]:+.4f}, {p['slope_ci'][1]:+.4f}]")
            if "multivariate" in res:
                mv = res["multivariate"]
                print(f"  다변량(n={mv['n']:,}): " +
                      ", ".join(f"{k} {v:+.4f} [{mv['coef_ci'][k][0]:+.4f},{mv['coef_ci'][k][1]:+.4f}]"
                               for k, v in mv["coef"].items() if k != "intercept"))
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
