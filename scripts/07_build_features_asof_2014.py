"""
07_build_features_asof_2014.py — Rayana & Akoglu(2015) Table 2 피처의 작성 시점(as-of) 재계산

03_build_features.py 는 작성자·업체의 '전체 기간' 리뷰로 피처를 계산해 미래 정보가 섞인다
(9/9 지도교수 피드백: "해당 리뷰 작성 시점까지의 이력만"). 이 스크립트는 2014년 리뷰
(= graph_2014 노드) 각각에 대해, 그 리뷰 작성 시점까지의 이력만으로 피처를 다시 계산한다.

시간 순서: (date, review_id) 사전식, 자기 자신 포함.
  = 01_build_reviews.py 의 n_reviews_user_asof, 06 의 과거 방향 엣지와 같은 규칙.
이력 범위: 2014년 이전 전체(2004~) + 2014년 중 그 리뷰보다 먼저 쓴 리뷰. 이후 리뷰는 제외.

03 과 달라진 점
  [미래 정보 제거]
    RD, DEV     업체 전체 평균 → 그 시점까지의 업체 평균
    ISR         작성자의 '평생' 유일 리뷰 → 작성자의 첫 리뷰 (= type_new)
    F           전체 코퍼스의 유사 리뷰 수 → 그 시점까지 쓰인 유사 리뷰 수
    DLu, DLb    전체 코퍼스 언어모델 → 2014년 이전 리뷰만으로 만든 언어모델 (+1 스무딩)
    U_*, P_*    전체 기간 → 그 시점까지의 작성자/업체 리뷰.
                avgRD·WRD 에 들어가는 각 과거 리뷰의 RD 는 그 리뷰가 쓰인 시점의 업체 평균 기준.
    CDF 변환    60만 건 분포 → 분석 대상인 2014년 노드 분포
                (논문: "empirical probability distribution ... over all the nodes of the given type")
  [원 논문 정의에 맞춘 수정]
    P_BST 제거  논문 본문 "One feature (BST) applies only to users" → 38개가 아니라 37개
    ACS/MCS/F   유니그램+바이그램(어휘 min_df=5) → 바이그램만 (논문 "bag-of-bigrams").
                해싱 벡터라 코사인이 두 리뷰 텍스트에만 의존 (코퍼스 어휘 선택에 미래가 끼지 않음)
    ACS/MCS     그룹당 200건 무작위 표본 근사 → 모든 쌍 정확 계산
  [그대로 — 원래 미래 정보 없음]
    Rank, EXT, ETF, PCW, PC, L, PP1, RES, SW, OW
    (ETF 는 논문 표기 (T-F)/delta 를 그대로 따른다. 03 과 동일)

주의: 저활동형(첫 리뷰)은 작성자 이력이 자기 1건뿐이라 U_* 가 상수이거나 리뷰 피처의 복사본이다
  U_MNR=1, U_BST=1, U_ERD=U_ETG=U_ACS=U_MCS=0, U_avgRD=U_WRD=RD, U_RL=L, U_PR/U_NR=자기 평점
  오류가 아니라, 작성 시점에 실제로 알 수 있는 작성자 정보가 그것뿐이라는 뜻이다.

출력 (data/processed/):
  features_rayana_asof_2014.parquet       180,659 x (review_id + 37), CDF 변환 [0,1]
  features_rayana_asof_2014_raw.parquet   같은 모양, 변환 전 원시값
  features_rayana_asof_2014_meta.json     피처별 H/L·수준
  행 순서 = review_id 오름차순 = graph_2014/nodes.parquet 행 순서 (그래프 노드 i ↔ i번째 행)

검증: `--check`
  (1) 미래 절단 테스트: 2014-07-01 이후 리뷰를 지운 데이터로 다시 계산해도 그 이전 리뷰의 값이 같은가
  (2) 정의 대조: 표본 리뷰마다 이력을 직접 잘라 정의대로 계산한 값과 같은가
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy, rankdata, spearmanr
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loaders import load_reviews  # noqa: E402

# 03 의 텍스트 피처 함수·상수를 그대로 재사용 (정의 일치)
_spec = importlib.util.spec_from_file_location("build03", ROOT / "scripts" / "03_build_features.py")
b03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b03)

YEAR = 2014
PROCESSED = ROOT / "data" / "processed"
OUT = PROCESSED / f"features_rayana_asof_{YEAR}.parquet"
OUT_RAW = PROCESSED / f"features_rayana_asof_{YEAR}_raw.parquet"
OUT_META = PROCESSED / f"features_rayana_asof_{YEAR}_meta.json"
OLD = PROCESSED / "features_rayana.parquet"

TAU_BST, ALPHA_WRD, DELTA_ETF = b03.TAU_BST, b03.ALPHA_WRD, b03.DELTA_ETF
BETA1_DEV, BETA2_ETF, MAX_GAP = b03.BETA1_DEV, b03.BETA2_ETF, b03.MAX_GAP
SIM_SEED, LSH_BITS = b03.SIM_SEED, b03.LSH_BITS
HASH_DIM = 2 ** 20      # 바이그램 해싱 차원 (충돌 무시 가능한 크기)
SIM_BLOCK = 2048        # 유사도 블록 크기 (행)
LEAK_CUTOFF = f"{YEAR}-07-01"

REVIEW = [("Rank", "L"), ("RD", "H"), ("EXT", "H"), ("DEV", "H"), ("ETF", "H"), ("ISR", "H"),
          ("PCW", "H"), ("PC", "H"), ("L", "L"), ("PP1", "L"), ("RES", "H"), ("SW", "H"),
          ("OW", "L"), ("DLu", "L"), ("DLb", "L"), ("F", "H")]
ENTITY = [("MNR", "H"), ("PR", "H"), ("NR", "H"), ("avgRD", "H"), ("WRD", "H"), ("BST", "H"),
          ("ERD", "L"), ("ETG", "L"), ("RL", "L"), ("ACS", "H"), ("MCS", "H")]
SPEC = {n: (hl, "review") for n, hl in REVIEW}
SPEC.update({f"U_{n}": (hl, "user") for n, hl in ENTITY})
SPEC.update({f"P_{n}": (hl, "product") for n, hl in ENTITY if n != "BST"})
COLUMNS = list(SPEC)
HISTORY_COLS = ["Rank", "RD", "DEV", "ETF", "ISR", "F"] + [c for c in COLUMNS if c[:2] in ("U_", "P_")]

_TOK = re.compile(r"\b\w+\b")


# ───────────────────────────── 텍스트 표현 ─────────────────────────────

def bigram_vectors(texts: list[str]):
    """L2 정규화된 이진 bag-of-bigrams (해싱). 코사인 = 두 텍스트에만 의존."""
    hv = HashingVectorizer(ngram_range=(2, 2), token_pattern=r"\b\w+\b", lowercase=True,
                           n_features=HASH_DIM, binary=True, norm="l2",
                           alternate_sign=False, dtype=np.float32)
    return hv.transform(texts).tocsr()


def fit_language_model(texts: list[str]):
    uni, bi = Counter(), Counter()
    for t in texts:
        d = _TOK.findall(t.lower())
        uni.update(d)
        bi.update(zip(d, d[1:]))
    return uni, bi


def description_lengths(texts: list[str], uni: Counter, bi: Counter) -> pd.DataFrame:
    """DLu/DLb — 03 과 같은 부호 길이(bits)이되, 언어모델은 과거(2014년 이전) 리뷰로만 만든다.
    처음 보는 단어가 있으므로 +1 (라플라스) 스무딩."""
    lu = math.log2(sum(uni.values()) + len(uni) + 1)
    lb = math.log2(sum(bi.values()) + len(bi) + 1)
    log2 = math.log2
    dlu, dlb = [], []
    for t in texts:
        d = _TOK.findall(t.lower())
        dlu.append(sum(lu - log2(uni.get(w, 0) + 1) for w in d))
        dlb.append(sum(lb - log2(bi.get(p, 0) + 1) for p in zip(d, d[1:])))
    return pd.DataFrame({"DLu": dlu, "DLb": dlb}, dtype="float64")


# ───────────────────────────── as-of 계산 도구 ─────────────────────────────

def _gsorted(df: pd.DataFrame, key: str):
    """key 그룹 → 시간순 정렬. (원래 행 위치, 정렬된 key) 반환."""
    s = df.sort_values([key, "date", "review_id"], kind="mergesort")
    return s.index.to_numpy(), s[key].to_numpy()


def _gcum(values: np.ndarray, keys: np.ndarray, how: str = "cumsum") -> np.ndarray:
    return getattr(pd.Series(values).groupby(keys, sort=False), how)().to_numpy()


def product_pass(df: pd.DataFrame, day: np.ndarray):
    """Rank / 그 시점까지 업체 평균 대비 RD / 업체 첫 리뷰로부터 경과일."""
    n = len(df)
    idx, keys = _gsorted(df, "prod_id")
    r = df["rating"].to_numpy("float64")[idx]
    k = _gcum(np.ones(n), keys)
    rd = np.abs(r - _gcum(r, keys) / k)
    first = pd.Series(day[idx]).groupby(keys, sort=False).transform("first").to_numpy()
    rank, rd_all, elapsed = (np.empty(n) for _ in range(3))
    rank[idx], rd_all[idx], elapsed[idx] = k, rd, day[idx] - first
    return rank, rd_all, elapsed


def content_similarity(X, idx: np.ndarray, gstart: np.ndarray, need: np.ndarray):
    """그룹(시간순)마다 '자기 + 과거 리뷰' 사이 모든 쌍의 평균/최대 코사인 (정확 계산).

    새 리뷰 p 가 들어올 때 추가되는 쌍은 (p, q<p) 뿐이므로
    ACS_p = sum_{p'<=p} sum_{q<p'} cos / C(p+1, 2),  MCS_p = max_{p'<=p} max_{q<p'} cos.
    need(정렬 순서) 가 True 인 마지막 행까지만 계산한다.
    """
    n = len(idx)
    acs, mcs = np.zeros(n), np.zeros(n)
    gend = np.r_[gstart[1:], n]
    for a, b in zip(gstart, gend):
        nz = np.flatnonzero(need[a:b])
        if nz.size == 0 or nz[-1] == 0:
            continue
        m = nz[-1] + 1
        Xg = X[idx[a:a + m]]
        row_sum, row_max = np.zeros(m), np.zeros(m)
        for c0 in range(0, m, SIM_BLOCK):
            c1 = min(m, c0 + SIM_BLOCK)
            S = (Xg[c0:c1] @ Xg[:c1].T).toarray()
            S[np.arange(c1)[None, :] >= np.arange(c0, c1)[:, None]] = 0.0   # q < p 만
            row_sum[c0:c1] = S.sum(axis=1, dtype=np.float64)
            row_max[c0:c1] = S.max(axis=1)
        kk = np.arange(1, m + 1, dtype=np.float64)
        npairs = kk * (kk - 1) / 2
        acs[a:a + m] = np.where(npairs > 0, np.cumsum(row_sum) / np.maximum(npairs, 1), 0.0)
        mcs[a:a + m] = np.maximum.accumulate(row_max)
    return acs, mcs


def entity_pass(df, key, day, rd_all, words, prank, X, need):
    """MNR PR NR avgRD WRD BST ERD ETG RL ACS MCS — 그 시점까지의 작성자/업체 리뷰로."""
    n = len(df)
    idx, keys = _gsorted(df, key)
    r = df["rating"].to_numpy("float64")[idx]
    d = day[idx]
    start = np.r_[True, keys[1:] != keys[:-1]]
    gstart = np.flatnonzero(start)
    gid = np.cumsum(start) - 1
    ones = np.ones(n)
    k = _gcum(ones, keys)

    out = {}
    per_day = pd.Series(ones).groupby([keys, d], sort=False).cumsum().to_numpy()
    out["MNR"] = _gcum(per_day, keys, "cummax")
    out["PR"] = _gcum((r >= 4).astype(float), keys) / k
    out["NR"] = _gcum((r <= 2).astype(float), keys) / k
    rdi = rd_all[idx]
    out["avgRD"] = _gcum(rdi, keys) / k
    w = np.power(np.maximum(prank[idx], 1.0), -ALPHA_WRD)
    out["WRD"] = _gcum(rdi * w, keys) / _gcum(w, keys)
    span = d - d[gstart][gid]
    out["BST"] = np.where(span > TAU_BST, 0.0, 1.0 - span / TAU_BST)

    C = np.stack([_gcum((r == v).astype(float), keys) for v in (1, 2, 3, 4, 5)], axis=1)
    P = C / k[:, None]
    out["ERD"] = -np.where(P > 0, P * np.log(np.where(P > 0, P, 1.0)), 0.0).sum(axis=1)

    # ETG: 히스토그램 엔트로피를 누적 갱신. H = ln N - (sum c ln c) / N
    gap = np.r_[0, np.diff(d)]
    valid = (~start) & (gap < MAX_GAP)          # 정렬돼 있으므로 gap >= 0
    c = np.zeros(n)
    vp = np.flatnonzero(valid)
    c[vp] = pd.Series(np.ones(len(vp))).groupby([keys[vp], gap[vp]], sort=False).cumsum().to_numpy() - 1
    xlogx = lambda x: np.where(x > 0, x * np.log(np.where(x > 0, x, 1.0)), 0.0)  # noqa: E731
    delta = np.where(valid, xlogx(c + 1) - xlogx(c), 0.0)
    S, N = _gcum(delta, keys), _gcum(valid.astype(float), keys)
    out["ETG"] = np.where(N > 0, np.log(np.maximum(N, 1)) - S / np.maximum(N, 1), 0.0)

    out["RL"] = _gcum(words[idx], keys) / k
    out["ACS"], out["MCS"] = content_similarity(X, idx, gstart, need[idx])

    res = {}
    for nm, v in out.items():
        a = np.empty(n)
        a[idx] = v
        res[nm] = a
    kk = np.empty(n)
    kk[idx] = k
    return res, kk


def review_frequency(X, df: pd.DataFrame, day: np.ndarray) -> np.ndarray:
    """F — 03 과 같은 SimHash 서명. 전체 코퍼스가 아니라 '그 시점까지' 같은 서명의 리뷰 수."""
    rng = np.random.default_rng(SIM_SEED)
    P = rng.standard_normal((X.shape[1], LSH_BITS)).astype(np.float32)
    sig = (np.asarray(X @ P) > 0).astype(np.int64) @ (1 << np.arange(LSH_BITS, dtype=np.int64))
    sig[X.getnnz(axis=1) == 0] = -1                  # 바이그램 없는 리뷰(1단어 이하)는 별도 버킷
    order = np.lexsort((df["review_id"].to_numpy(), day))
    f = np.empty(len(df))
    f[order] = pd.Series(sig[order]).groupby(sig[order], sort=False).cumcount().to_numpy() + 1
    return f, sig


# ───────────────────────────── 전체 계산 ─────────────────────────────

def compute_raw(df: pd.DataFrame, X, lm, lex, target: np.ndarray, text: bool = True,
                log=print) -> pd.DataFrame:
    """df: review_id 오름차순, index 0..n-1. X: df 행과 정렬된 바이그램 벡터.
    target 행(분석 대상)의 as-of 원시 피처를 review_id 순으로 반환."""
    t0 = time.time()
    day = df["date"].to_numpy("datetime64[D]").astype(np.int64)
    tpos = np.flatnonzero(target)
    texts = df["text"].tolist()
    words = np.array([len(t.split(" ")) for t in texts], dtype=np.float64)   # 03 의 L 과 동일 정의

    rank, rd_all, elapsed = product_pass(df, day)
    f, _ = review_frequency(X, df, day)
    log(f"      Rank/RD/ETF/F {time.time() - t0:.0f}초")

    user, k_user = entity_pass(df, "user_id", day, rd_all, words, rank, X, target)
    log(f"      작성자 11개 {time.time() - t0:.0f}초")
    prod, _ = entity_pass(df, "prod_id", day, rd_all, words, rank, X, target)
    log(f"      업체 10개 {time.time() - t0:.0f}초")

    star = df["rating"].to_numpy("float64")
    fe = np.where(elapsed > DELTA_ETF, 0.0, elapsed / DELTA_ETF)
    raw = pd.DataFrame({
        "Rank": rank[tpos],
        "RD": rd_all[tpos],
        "EXT": np.isin(star[tpos], (4.0, 5.0)).astype(float),
        "DEV": ((rd_all[tpos] / 4.0) > BETA1_DEV).astype(float),
        "ETF": (fe[tpos] > BETA2_ETF).astype(float),
        "ISR": (k_user[tpos] == 1).astype(float),
    })
    if text:
        tt = [texts[i] for i in tpos]
        raw = pd.concat([raw, b03.review_text_features(tt, lex),
                         description_lengths(tt, *lm)], axis=1)
        log(f"      텍스트 {time.time() - t0:.0f}초")
    raw["F"] = f[tpos]
    for nm, _ in ENTITY:
        raw[f"U_{nm}"] = user[nm][tpos]
    for nm, _ in ENTITY:
        if nm != "BST":
            raw[f"P_{nm}"] = prod[nm][tpos]
    raw.insert(0, "review_id", df["review_id"].to_numpy()[tpos])
    cols = ["review_id"] + [c for c in COLUMNS if c in raw.columns]
    return raw[cols]


def cdf_transform(raw: pd.DataFrame) -> pd.DataFrame:
    """논문 f(x): H → 1 - P(X<=x), L → P(X<=x). 분포는 분석 대상 노드(2014년) 기준."""
    out = {"review_id": raw["review_id"].to_numpy()}
    n = len(raw)
    for c in COLUMNS:
        p = rankdata(raw[c].to_numpy(), method="average") / n
        out[c] = (1.0 - p) if SPEC[c][0] == "H" else p
    return pd.DataFrame(out)


# ───────────────────────────── 검증 ─────────────────────────────

def check_future_truncation(df, X, lm, lex, raw_full):
    """cutoff 이후 리뷰를 모두 지워도 cutoff 이전 대상 리뷰의 피처가 똑같아야 한다."""
    keep = np.flatnonzero((df["date"] < LEAK_CUTOFF).to_numpy())
    dft = df.iloc[keep].reset_index(drop=True)
    tgt = (dft["year"] == YEAR).to_numpy()
    rt = compute_raw(dft, X[keep], lm, lex, tgt, text=False, log=lambda *_: None)
    rf = raw_full.set_index("review_id").loc[rt["review_id"], HISTORY_COLS].to_numpy()
    diff = np.abs(rt[HISTORY_COLS].to_numpy() - rf)
    tol = 1e-6 * np.maximum(1.0, np.abs(rf))
    bad = (diff > tol).sum(axis=0)
    print(f"  (1) 미래 절단 테스트: {LEAK_CUTOFF} 이전 2014년 리뷰 {len(rt):,}건, 이력 의존 피처 "
          f"{len(HISTORY_COLS)}개 → 불일치 {int(bad.sum())}건 (최대 차이 {diff.max():.2e})")
    assert bad.sum() == 0, dict(zip(HISTORY_COLS, bad))


def check_against_definition(df, X, raw_full, n_sample=300, seed=0, max_hist=4000):
    """표본 리뷰마다 이력을 직접 잘라 Table 2 정의대로 계산해 비교 (벡터화 구현과 독립)."""
    day = df["date"].to_numpy("datetime64[D]").astype(np.int64)
    rid = df["review_id"].to_numpy()
    okey = day * 10 ** 7 + rid
    star = df["rating"].to_numpy("float64")
    words = np.array([len(t.split(" ")) for t in df["text"]], dtype=np.float64)
    groups = {k: df.groupby(k).indices for k in ("user_id", "prod_id")}
    _, sig = review_frequency(X, df, day)

    def hist(key, i):
        g = groups[key][df[key].iat[i]]
        return np.sort(g[okey[g] <= okey[i]])      # review_id 순 = 인덱스 순

    def rd_naive(j):
        h = hist("prod_id", j)
        return abs(star[j] - star[h].mean())

    def prank_naive(j):
        return len(hist("prod_id", j))

    def entity_naive(h, with_bst):
        dd, s = day[h], star[h]
        o = {"MNR": pd.Series(dd).value_counts().max(),
             "PR": (s >= 4).mean(), "NR": (s <= 2).mean()}
        rds = np.array([rd_naive(j) for j in h])
        w = np.array([1.0 / max(prank_naive(j), 1) ** ALPHA_WRD for j in h])
        o["avgRD"], o["WRD"] = rds.mean(), (rds * w).sum() / w.sum()
        if with_bst:
            span = dd.max() - dd.min()
            o["BST"] = 0.0 if span > TAU_BST else 1.0 - span / TAU_BST
        o["ERD"] = entropy(np.array([(s == v).sum() for v in (1, 2, 3, 4, 5)], float))
        gaps = np.diff(np.sort(dd))
        gaps = gaps[(gaps >= 0) & (gaps < MAX_GAP)]
        o["ETG"] = entropy(np.bincount(gaps, minlength=MAX_GAP).astype(float)) if gaps.size else 0.0
        o["RL"] = words[h].mean()
        if len(h) > 1 and len(h) <= max_hist:
            Sm = (X[h] @ X[h].T).toarray()
            np.fill_diagonal(Sm, 0.0)
            o["ACS"], o["MCS"] = Sm.sum() / (len(h) * (len(h) - 1)), Sm.max()
        elif len(h) == 1:
            o["ACS"] = o["MCS"] = 0.0
        return o

    rng = np.random.default_rng(seed)
    tpos = np.flatnonzero((df["year"] == YEAR).to_numpy())
    sample = rng.choice(tpos, n_sample, replace=False)
    R = raw_full.set_index("review_id")
    worst, n_cmp, skipped = {}, 0, 0
    for i in sample:
        hu, hp = hist("user_id", i), hist("prod_id", i)
        exp = {"Rank": len(hp), "RD": rd_naive(i), "ISR": float(len(hu) == 1),
               "F": float(((sig == sig[i]) & (okey <= okey[i])).sum())}
        for nm, v in entity_naive(hu, True).items():
            exp[f"U_{nm}"] = v
        pe = entity_naive(hp, False)
        skipped += "ACS" not in pe
        for nm, v in pe.items():
            exp[f"P_{nm}"] = v
        got = R.loc[rid[i]]
        for c, v in exp.items():
            d = abs(float(got[c]) - float(v)) / max(1.0, abs(float(v)))
            worst[c] = max(worst.get(c, 0.0), d)
            n_cmp += 1
    bad = {c: v for c, v in worst.items() if v > 1e-4}
    print(f"  (2) 정의 대조: 표본 {n_sample}건 × 피처 → {n_cmp:,}개 값 비교, "
          f"최대 상대오차 {max(worst.values()):.2e} "
          f"(업체 이력 {max_hist}건 초과로 P_ACS/P_MCS 대조 생략 {skipped}건)")
    assert not bad, bad


# ───────────────────────────── main ─────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="미래 절단 테스트 + 정의 대조 검증")
    args = ap.parse_args()

    t0 = time.time()
    df = load_reviews(columns=["review_id", "user_id", "prod_id", "rating", "date", "text",
                               "year", "fraud", "type_new"]).reset_index(drop=True)
    df["text"] = df["text"].fillna("")
    target = (df["year"] == YEAR).to_numpy()
    print(f"전체 {len(df):,}건 | 분석 대상 {YEAR}년 {target.sum():,}건")

    print("[1/4] 언어모델(2014년 이전 리뷰) + 바이그램 벡터")
    lm = fit_language_model(df.loc[df["date"] < f"{YEAR}-01-01", "text"].tolist())
    X = bigram_vectors(df["text"].tolist())
    lex = b03.sentiment_lexicon()
    print(f"      유니그램 {len(lm[0]):,} / 바이그램 {len(lm[1]):,} | {time.time() - t0:.0f}초")

    print("[2/4] as-of 피처 계산")
    raw = compute_raw(df, X, lm, lex, target)
    raw = raw.replace([np.inf, -np.inf], np.nan)
    assert raw[COLUMNS].notna().all().all(), raw[COLUMNS].isna().sum()[lambda s: s > 0]
    tn = df.loc[target, "type_new"].to_numpy()
    assert (raw["ISR"].to_numpy() == tn).all(), "ISR(as-of) 와 type_new 불일치"
    assert raw["review_id"].is_monotonic_increasing

    print("[3/4] 저장")
    feats = cdf_transform(raw)
    feats[COLUMNS] = feats[COLUMNS].astype("float32")
    feats.to_parquet(OUT, index=False)
    raw.astype({c: "float64" for c in COLUMNS}).to_parquet(OUT_RAW, index=False)
    meta = {c: {"suspicious": SPEC[c][0], "level": SPEC[c][1]} for c in COLUMNS}
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    nodes_path = PROCESSED / f"graph_{YEAR}" / "nodes.parquet"
    if nodes_path.exists():
        nodes = pd.read_parquet(nodes_path, columns=["review_id"])
        assert (nodes["review_id"].to_numpy() == feats["review_id"].to_numpy()).all()
        print("      graph_2014/nodes.parquet 과 행 순서 일치 확인")
    lv = pd.Series([SPEC[c][1] for c in COLUMNS]).value_counts()
    print(f"      {len(COLUMNS)}개 = review {lv['review']} + user {lv['user']} + product {lv['product']}"
          f" | {OUT.name}")

    print("[4/4] 요약")
    low = tn == 1
    ucols = [f"U_{n}" for n, _ in ENTITY]
    print("  저활동(첫 리뷰) 노드의 작성자 피처 원시값 고유값 수:",
          {c: int(raw.loc[low, c].nunique()) for c in ucols})
    fraud = df.loc[target, "fraud"].to_numpy()
    if OLD.exists():
        old = pd.read_parquet(OLD).set_index("review_id").loc[raw["review_id"]]
        print(f"  기존(전체 기간) vs as-of — 단일 피처 PR-AUC (2014, 기준선 = 사기율 {fraud.mean():.3f})")
        print(f"    {'피처':<8} {'기존':>7} {'as-of':>7} {'순위상관':>8}")
        for c in COLUMNS:
            if c not in old.columns:
                continue
            rho = spearmanr(old[c], feats[c]).correlation
            if rho > 0.999:
                continue
            a_old = average_precision_score(fraud, 1 - old[c].to_numpy())
            a_new = average_precision_score(fraud, 1 - feats[c].to_numpy())
            print(f"    {c:<8} {a_old:7.3f} {a_new:7.3f} {rho:8.3f}")
    print(f"  총 {time.time() - t0:.0f}초")

    if args.check:
        print("\n[검증]")
        check_future_truncation(df, X, lm, lex, raw)
        check_against_definition(df, X, raw)
        print("  검증 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
