"""
03_build_features.py — Rayana & Akoglu (2015) Table 2 피처 재구현 (조건 A 입력)

KDD 2015 "Collective Opinion Spam Detection" Table 2 의 정의를 그대로 구현한다.
표는 27개 피처를 정의한다:
  User & Product Features 11개 — MNR PR NR avgRD WRD BST ERD ETG | RL ACS MCS
  Review Features         16개 — Rank RD EXT DEV ETF ISR | PCW PC L PP1 RES SW OW F DLu DLb

리뷰를 노드로 쓰므로 User 11 + Product 11 + Review 16 = 38개 값이 리뷰마다 붙는다.
(YelpChi `.mat` 은 이 중 32개를 쓰지만 어느 6개인지 논문에 명시가 없어 특정 불가)

Table 2 는 각 피처에 H/L 을 표시한다 — 값이 높을수록(H) 또는 낮을수록(L) 사기 의심.
논문은 이를 경험적 CDF 로 [0,1] 점수로 변환한다:
    f(x) = 1 - P(X <= x)   if H
    f(x) = P(X <= x)       if L
`.mat` 의 값이 모두 [0,1] 범위인 것과 일치한다. 기본으로 이 변환을 적용하며
`--raw` 로 원시값도 저장할 수 있다.

전체 데이터에서 계산해야 한다. 부분 표본에서 계산하면 업체 평균·작성자 통계가
왜곡된다.

출력:
  data/processed/features_rayana.parquet        (608458 x 38, review_id 순서)
  data/processed/features_rayana_meta.json      피처별 정의·H/L·수준 분류
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy, rankdata
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "reviews.parquet"
OUT = ROOT / "data" / "processed" / "features_rayana.parquet"
OUT_META = ROOT / "data" / "processed" / "features_rayana_meta.json"

TAU_BST = 28          # BST: burstiness 임계 (일). Table 2: tau = 28 days
ALPHA_WRD = 1.5       # WRD: 가중 감쇠율. Table 2: alpha = 1.5
DELTA_ETF = 7 * 30    # ETF: early time frame (일). Table 2: delta = 7 months
BETA1_DEV = 0.63      # DEV 임계. 논문은 recursive minimal entropy partitioning 으로
BETA2_ETF = 0.5       # ETF 임계. 학습하나 라벨을 쓰므로 고정값 사용(누수 방지)
MAX_GAP = 1000        # ETG 히스토그램 상한 (일)
SIM_CAP = 200         # ACS/MCS 계산 시 그룹당 표본 상한 (업체 최대 7,378건)
SIM_SEED = 42
LSH_BITS = 16         # F: 근사 중복 탐지용 랜덤 투영 비트 수

PRONOUNS = {"i", "me", "we", "us", "my", "mine", "our", "ours", "myself", "ourselves"}

# 피처명 -> (H/L, 수준)  ※ 수준: review / user / product
SPEC: dict[str, tuple[str, str]] = {}


def _reg(name: str, hl: str, level: str) -> str:
    SPEC[name] = (hl, level)
    return name


# ─────────────────────────── 텍스트 기반 리뷰 피처 ───────────────────────────

def sentiment_lexicon() -> dict[str, float]:
    from nltk.corpus import sentiwordnet as swn
    lex: dict[str, list[float]] = {}
    for s in swn.all_senti_synsets():
        subj = s.pos_score() + s.neg_score()
        for lemma in s.synset.lemma_names():
            lex.setdefault(lemma.lower().replace("_", " "), []).append(subj)
    return {w: float(np.mean(v)) for w, v in lex.items()}


def review_text_features(texts: list[str], lex: dict[str, float]) -> pd.DataFrame:
    """PCW PC L PP1 RES SW OW — Table 2 Review/Text."""
    word_re = re.compile(r"\b\w+\b")
    letters_re = re.compile(r"[^a-zA-Z]")
    # 문장 끝 부호 '뒤에서' 자른다. 구분자를 문장에 남겨야 '!' 포함 여부를 셀 수 있다.
    sent_re = re.compile(r"(?<=[.!?])\s+")

    pcw, pc, ln, pp1, res, sw, ow = ([] for _ in range(7))
    for t in texts:
        parts = t.split(" ")
        n = max(len(parts), 1)
        pcw.append(sum(w.isupper() for w in parts) / n)
        pc.append(sum(1 for c in t if c.isupper()) / len(t) if t else 0.0)
        ln.append(len(parts))
        low = [letters_re.sub("", w).lower() for w in parts]
        pp1.append(sum(w in PRONOUNS for w in low) / n)
        sents = sent_re.split(t)
        res.append(sum("!" in s for s in sents) / max(len(sents), 1))
        ws = word_re.findall(t.lower())
        if ws:
            sc = [lex.get(w, 0.0) for w in ws]
            sw.append(float(np.mean(sc)))
            ow.append(float(np.mean([1.0 - s if s > 0 else 0.0 for s in sc])))
        else:
            sw.append(0.0)
            ow.append(0.0)

    return pd.DataFrame({
        _reg("PCW", "H", "review"): pcw,
        _reg("PC", "H", "review"): pc,
        _reg("L", "L", "review"): ln,
        _reg("PP1", "L", "review"): pp1,
        _reg("RES", "H", "review"): res,
        _reg("SW", "H", "review"): sw,
        _reg("OW", "L", "review"): ow,
    })


def description_lengths(texts: list[str]) -> pd.DataFrame:
    """DLu / DLb — 코퍼스 유니그램·바이그램 모델 하에서의 부호 길이(bits).

    Table 2 는 'information-theoretic description length [new]' 라고만 적고 산출식을
    주지 않는다. 표준적인 해석인 음의 로그우도(부호 길이)로 구현한다.
    템플릿·복제 텍스트일수록 짧아지므로 L(낮을수록 의심)이 된다.
    """
    tok = re.compile(r"\b\w+\b")
    docs = [tok.findall(t.lower()) for t in texts]

    uni: dict[str, int] = {}
    bi: dict[tuple[str, str], int] = {}
    for d in docs:
        for w in d:
            uni[w] = uni.get(w, 0) + 1
        for a, b in zip(d, d[1:]):
            bi[(a, b)] = bi.get((a, b), 0) + 1
    nu = sum(uni.values()) or 1
    nb = sum(bi.values()) or 1
    log2 = np.log2

    dlu, dlb = [], []
    for d in docs:
        if not d:
            dlu.append(0.0)
            dlb.append(0.0)
            continue
        dlu.append(float(-sum(log2(uni[w] / nu) for w in d)))
        pairs = list(zip(d, d[1:]))
        dlb.append(float(-sum(log2(bi[p] / nb) for p in pairs)) if pairs else 0.0)

    return pd.DataFrame({
        _reg("DLu", "L", "review"): dlu,
        _reg("DLb", "L", "review"): dlb,
    })


def review_frequency(Xn, bits: int = LSH_BITS, seed: int = SIM_SEED) -> np.ndarray:
    """F — 유사 리뷰의 빈도. Table 2 는 locality sensitive hashing 근사라고 명시.

    정규화된 bag-of-bigrams 에 랜덤 초평면 투영(SimHash)을 적용하고, 같은 서명을
    가진 리뷰 수를 센다. 서명이 같으면 코사인 유사도가 높을 확률이 크다.
    """
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((Xn.shape[1], bits)).astype("float32")
    sign = (Xn @ P) > 0
    key = (sign * (1 << np.arange(bits))).sum(axis=1)
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    return cnt[inv].astype("float64")


# ────────────────────────── 행동 기반 리뷰 피처 ──────────────────────────

def review_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rank RD EXT DEV ETF ISR — Table 2 Review/Behavior."""
    n = len(df)
    star = df["rating"].to_numpy("float64")
    day = df["date"].to_numpy("datetime64[D]").astype("int64")

    rank = np.zeros(n)
    rd = np.zeros(n)
    etf = np.zeros(n)
    for _, g in df.groupby("prod_id", sort=False):
        pos = g.index.to_numpy()
        s = star[pos]
        d = day[pos]
        # Rank: 업체 리뷰 중 시간 순서 (1 = 가장 이른 리뷰)
        rank[pos] = rankdata(d, method="ordinal")
        # RD: 업체 평균 대비 절대 편차
        rd[pos] = np.abs(s - s.mean())
        # ETF: 업체 첫 리뷰로부터의 경과가 delta 이내면 (T-F)/delta, 넘으면 0
        elapsed = d - d.min()
        f = np.where(elapsed > DELTA_ETF, 0.0, elapsed / DELTA_ETF)
        etf[pos] = (f > BETA2_ETF).astype("float64")

    ext = np.isin(star, (4.0, 5.0)).astype("float64")   # Table 2: {4,5} = 1
    dev = ((rd / 4.0) > BETA1_DEV).astype("float64")
    isr = (df["n_reviews_user"].to_numpy() == 1).astype("float64")

    return pd.DataFrame({
        _reg("Rank", "L", "review"): rank,
        _reg("RD", "H", "review"): rd,
        _reg("EXT", "H", "review"): ext,
        _reg("DEV", "H", "review"): dev,
        _reg("ETF", "H", "review"): etf,
        _reg("ISR", "H", "review"): isr,
    })


# ───────────────────── User / Product 공통 11개 피처 ─────────────────────

def entity_features(df: pd.DataFrame, key: str, prefix: str, Xn,
                    rd: np.ndarray, length_words: np.ndarray,
                    prod_rank: np.ndarray) -> pd.DataFrame:
    """MNR PR NR avgRD WRD BST ERD ETG RL ACS MCS — 작성자 또는 업체 단위."""
    n = len(df)
    names = ["MNR", "PR", "NR", "avgRD", "WRD", "BST", "ERD", "ETG", "RL", "ACS", "MCS"]
    hl = ["H", "H", "H", "H", "H", "H", "L", "L", "L", "H", "H"]
    level = "user" if key == "user_id" else "product"
    cols = [_reg(f"{prefix}_{nm}", h, level) for nm, h in zip(names, hl)]
    F = np.zeros((n, len(names)))

    star = df["rating"].to_numpy("float64")
    day = df["date"].to_numpy("datetime64[D]").astype("int64")
    rng = np.random.default_rng(SIM_SEED)

    for _, g in df.groupby(key, sort=False):
        pos = g.index.to_numpy()
        k = len(pos)
        s = star[pos]
        d = day[pos]

        _, cnt = np.unique(d, return_counts=True)
        F[pos, 0] = cnt.max()                       # MNR
        F[pos, 1] = (s >= 4).sum() / k              # PR
        F[pos, 2] = (s <= 2).sum() / k              # NR
        F[pos, 3] = rd[pos].mean()                  # avgRD
        # WRD: 업체 내 리뷰 순위로 가중 (w = 1/rank^alpha)
        w = 1.0 / np.power(np.maximum(prod_rank[pos], 1.0), ALPHA_WRD)
        F[pos, 4] = (rd[pos] * w).sum() / w.sum()   # WRD
        span = d.max() - d.min()                    # BST
        F[pos, 5] = 0.0 if span > TAU_BST else 1.0 - span / TAU_BST
        pk = np.array([(s == v).sum() for v in (1, 2, 3, 4, 5)], dtype="float64")
        F[pos, 6] = entropy(pk) if pk.sum() > 0 else 0.0    # ERD
        if k > 1:                                   # ETG
            gaps = np.diff(np.sort(d))
            gaps = gaps[(gaps >= 0) & (gaps < MAX_GAP)]
            if gaps.size:
                F[pos, 7] = entropy(np.bincount(gaps, minlength=MAX_GAP).astype("float64"))
        F[pos, 8] = length_words[pos].mean()        # RL
        if k > 1:                                   # ACS / MCS
            sel = pos if k <= SIM_CAP else rng.choice(pos, SIM_CAP, replace=False)
            S = (Xn[sel] @ Xn[sel].T).toarray()
            np.fill_diagonal(S, 0.0)
            m = len(sel)
            F[pos, 9] = S.sum() / (m * (m - 1))
            F[pos, 10] = S.max()

    return pd.DataFrame(F, columns=cols)


# ─────────────────────────────── CDF 변환 ───────────────────────────────

def cdf_transform(feats: pd.DataFrame) -> pd.DataFrame:
    """Table 2 의 H/L 에 따라 f(x) 를 계산. 의심스러운 값일수록 낮은 점수."""
    out = {}
    n = len(feats)
    for c in feats.columns:
        hl = SPEC[c][0]
        p = rankdata(feats[c].to_numpy(), method="average") / n   # P(X <= x)
        out[c] = (1.0 - p) if hl == "H" else p
    return pd.DataFrame(out, index=feats.index)


# ─────────────────────────────────── main ───────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true", help="CDF 변환 전 원시값도 저장")
    args = ap.parse_args()

    t0 = time.time()
    df = pd.read_parquet(SRC)
    if not df["review_id"].is_monotonic_increasing:
        print("review_id 순서가 깨졌습니다", file=sys.stderr)
        return 1
    df = df.reset_index(drop=True)
    texts = df["text"].fillna("").tolist()
    print(f"입력 {len(df):,}행")

    print("[1/6] 텍스트 피처 (PCW PC L PP1 RES SW OW)")
    lex = sentiment_lexicon()
    f_text = review_text_features(texts, lex)
    print(f"      {time.time()-t0:.0f}초")

    print("[2/6] description length (DLu DLb)")
    f_dl = description_lengths(texts)
    print(f"      {time.time()-t0:.0f}초")

    print("[3/6] 행동 피처 (Rank RD EXT DEV ETF ISR)")
    f_beh = review_behavior_features(df)
    print(f"      {time.time()-t0:.0f}초")

    print("[4/6] bigram 벡터화 (ACS MCS F 용)")
    vec = CountVectorizer(ngram_range=(1, 2), token_pattern=r"\b\w+\b",
                          min_df=5, max_features=200_000, binary=True, dtype=np.float32)
    Xn = normalize(vec.fit_transform(texts)).tocsr()
    f_freq = pd.DataFrame({_reg("F", "H", "review"): review_frequency(Xn)})
    print(f"      어휘 {Xn.shape[1]:,} | {time.time()-t0:.0f}초")

    rd = f_beh["RD"].to_numpy()
    lw = f_text["L"].to_numpy("float64")
    prank = f_beh["Rank"].to_numpy()

    print("[5/6] 작성자 수준 11개")
    f_user = entity_features(df, "user_id", "U", Xn, rd, lw, prank)
    print(f"      {time.time()-t0:.0f}초")

    print("[6/6] 업체 수준 11개")
    f_prod = entity_features(df, "prod_id", "P", Xn, rd, lw, prank)
    print(f"      {time.time()-t0:.0f}초")

    raw = pd.concat([f_beh, f_text, f_dl, f_freq, f_user, f_prod], axis=1)
    raw = raw.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    feats = cdf_transform(raw).astype("float32")
    feats.insert(0, "review_id", df["review_id"].to_numpy())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(OUT, index=False)

    if args.raw:
        r = raw.astype("float32")
        r.insert(0, "review_id", df["review_id"].to_numpy())
        r.to_parquet(OUT.with_name("features_rayana_raw.parquet"), index=False)

    meta = {c: {"suspicious": SPEC[c][0], "level": SPEC[c][1]} for c in raw.columns}
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    lv = pd.Series([SPEC[c][1] for c in raw.columns]).value_counts()
    print(f"\n총 {raw.shape[1]}개 = review {lv.get('review',0)} + user {lv.get('user',0)} "
          f"+ product {lv.get('product',0)}")
    print(f"저장: {OUT}  ({OUT.stat().st_size/1e6:.0f} MB)  |  총 {time.time()-t0:.0f}초")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
