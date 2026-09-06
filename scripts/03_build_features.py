"""
03_build_features.py — 수작업 행동/텍스트 통계 피처 생성 (조건 A 입력)

Rayana & Akoglu (2015) 계열의 피처를 YelpZip 전체(608,458행)에서 재계산한다.
YelpChi `.mat` 에 들어있는 32차원 피처는 YelpZip 에는 없으므로 직접 만들어야 한다.

산출 피처 (총 38개):
  리뷰 수준 12개 = 행동 5 + 텍스트 파생 7
  작성자 수준 13개
  업체 수준   13개

★ 조건 A("구조/메타데이터")가 실제로는 텍스트 파생 통계를 포함한다는 점을 확인하기
  위해, 각 피처를 behavioral / text 로 분류해 `feature_groups.json` 에 함께 기록한다.
  (9/2 미팅 지도교수 지적: "구조 계열도 이미 텍스트에서 파생된 정보를 쓰고 있다")

참조 구현(FraudSquad)의 버그를 고쳐서 구현했다:
  - avg_rating / rating_std 자리에 내용 유사도를 덮어쓰던 문제
  - 업체 루프에서 직전 작성자의 bigram 을 참조하던 문제
  - 평점 엔트로피를 range(5)(=0~4)로 계산해 5점을 빠뜨리던 문제

전체 데이터에서 계산해야 한다. 부분 표본에서 계산하면 업체 평균·작성자 리뷰 수가
왜곡되어 값이 틀어진다.

출력:
  data/processed/features_handcrafted.parquet   (608458 x 38, review_id 순서)
  data/processed/feature_groups.json            피처별 behavioral/text 분류
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import entropy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "reviews.parquet"
OUT = ROOT / "data" / "processed" / "features_handcrafted.parquet"
OUT_GROUPS = ROOT / "data" / "processed" / "feature_groups.json"

SIM_CAP = 200        # 내용 유사도 계산 시 그룹당 최대 표본 수 (업체 최대 7,378건)
SIM_SEED = 42
MAX_GAP = 1000       # 시간 간격 엔트로피 최대 구간(일/월)

PRONOUNS = {"i", "me", "we", "us", "my", "mine", "our", "ours", "myself", "ourselves"}


# ────────────────────────────── 텍스트 통계 ──────────────────────────────

def build_sentiment_lexicon() -> dict[str, float]:
    """SentiWordNet 에서 단어 -> 주관성 점수(0~1) 사전을 만든다."""
    from nltk.corpus import sentiwordnet as swn
    lex: dict[str, list[float]] = {}
    for s in swn.all_senti_synsets():
        subj = s.pos_score() + s.neg_score()
        for lemma in s.synset.lemma_names():
            lex.setdefault(lemma.lower().replace("_", " "), []).append(subj)
    return {w: float(np.mean(v)) for w, v in lex.items()}


def review_text_features(texts: list[str], lex: dict[str, float]) -> pd.DataFrame:
    word_re = re.compile(r"\b\w+\b")
    letters_re = re.compile(r"[^a-zA-Z]")
    # 문장 끝 부호 '뒤에서' 자른다. 구분자를 문장에 남겨야 '!' 포함 여부를 셀 수 있다.
    # (구분자에 '!' 를 넣고 split 하면 '!' 가 제거되어 비율이 항상 0이 된다)
    sent_re = re.compile(r"(?<=[.!?])\s+")

    allcap, capletter, n_words, pronoun, excla, subj, obj = ([] for _ in range(7))
    for t in texts:
        parts = t.split(" ")
        n = max(len(parts), 1)
        allcap.append(sum(w.isupper() for w in parts) / n)
        capletter.append(sum(1 for c in t if c.isupper()) / len(t) if t else 0.0)
        n_words.append(len(parts))
        low = [letters_re.sub("", w).lower() for w in parts]
        pronoun.append(sum(w in PRONOUNS for w in low) / n)
        sents = sent_re.split(t)
        excla.append(sum("!" in s for s in sents) / max(len(sents), 1))
        ws = word_re.findall(t.lower())
        if ws:
            sc = [lex.get(w, 0.0) for w in ws]
            subj.append(float(np.mean(sc)))
            obj.append(float(np.mean([1.0 - s if s > 0 else 0.0 for s in sc])))
        else:
            subj.append(0.0)
            obj.append(0.0)

    return pd.DataFrame({
        "allcap_words_ratio": allcap,
        "cap_letter_ratio": capletter,
        "length_in_words": n_words,
        "first_pronoun_ratio": pronoun,
        "excla_sentence_ratio": excla,
        "subjective_word_ratio": subj,
        "objective_word_ratio": obj,
    })


# ───────────────────────────── 리뷰 수준 행동 ─────────────────────────────

def review_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    star = df["rating"].values.astype("float64")
    t = df["date"].values.astype("datetime64[s]").astype("int64")

    extremity = ((star > 4.9) | (star < 1.1)).astype("float64")

    rating_dev = np.zeros(n)
    deviated = np.zeros(n)
    early = np.zeros(n)
    for _, g in df.groupby("prod_id", sort=False):
        pos = g.index.values
        s = star[pos]
        rating_dev[pos] = s - s.mean()
        if len(pos) > 1:
            # 자기 자신을 뺀 평균과의 편차 (Rayana & Akoglu)
            loo = (s.sum() - s) / (len(s) - 1)
            deviated[pos[np.abs(s - loo) > 4 * 0.63]] = 1.0
        first = t[pos].min()
        early[pos[(t[pos] - first) <= 3600 * 24 * 30 * 2]] = 1.0

    singleton = (df["n_reviews_user"].values == 1).astype("float64")

    return pd.DataFrame({
        "extremity": extremity,
        "rating_deviation": rating_dev,
        "deviated_rating": deviated,
        "early_time": early,
        "singleton": singleton,
    })


# ─────────────────────── 그룹(작성자/업체) 수준 피처 ───────────────────────

def group_features(df: pd.DataFrame, key: str, Xn: sp.csr_matrix,
                   rating_dev: np.ndarray, length_words: np.ndarray,
                   prefix: str) -> pd.DataFrame:
    n = len(df)
    cols = ["max_reviews_day", "max_reviews_month", "ratio_positive", "ratio_negative",
            "avg_rating", "rating_std", "rating_entropy", "gap_entropy_day",
            "gap_entropy_month", "max_content_sim", "avg_content_sim",
            "avg_rating_deviation", "avg_review_length"]
    F = np.zeros((n, len(cols)))

    star = df["rating"].values.astype("float64")
    tsec = df["date"].values.astype("datetime64[s]").astype("int64")
    day = df["date"].values.astype("datetime64[D]")
    month = df["date"].values.astype("datetime64[M]")
    rng = np.random.default_rng(SIM_SEED)

    for _, g in df.groupby(key, sort=False):
        pos = g.index.values
        k = len(pos)
        s = star[pos]

        F[pos, 0] = np.unique(day[pos], return_counts=True)[1].max()
        F[pos, 1] = np.unique(month[pos], return_counts=True)[1].max()
        F[pos, 2] = (s >= 4).sum() / k
        F[pos, 3] = (s <= 2).sum() / k
        F[pos, 4] = s.mean()
        F[pos, 11] = rating_dev[pos].mean()
        F[pos, 12] = length_words[pos].mean()

        if k > 1:
            F[pos, 5] = s.std(ddof=1)
            pk = np.array([(s == i).sum() for i in range(1, 6)], dtype="float64")
            F[pos, 6] = entropy(pk) if pk.sum() > 0 else 0.0
            for interval, dim in ((3600 * 24, 7), (3600 * 24 * 30, 8)):
                gaps = np.diff(np.sort(tsec[pos] / interval)).round().astype(int)
                gaps = gaps[(gaps >= 0) & (gaps < MAX_GAP)]
                if gaps.size:
                    pk2 = np.bincount(gaps, minlength=MAX_GAP).astype("float64")
                    F[pos, dim] = entropy(pk2)
            # 내용 유사도: 그룹이 크면 표본을 뽑아 계산 (비용 상한)
            sel = pos if k <= SIM_CAP else rng.choice(pos, SIM_CAP, replace=False)
            S = (Xn[sel] @ Xn[sel].T).toarray()
            np.fill_diagonal(S, 0.0)
            m = len(sel)
            F[pos, 9] = S.max()
            F[pos, 10] = S.sum() / (m * (m - 1))

    return pd.DataFrame(F, columns=[f"{prefix}_{c}" for c in cols])


# ────────────────────────────────── main ──────────────────────────────────

def main() -> int:
    t0 = time.time()
    df = pd.read_parquet(SRC)
    if not df["review_id"].is_monotonic_increasing:
        print("review_id 순서가 깨졌습니다", file=sys.stderr)
        return 1
    df = df.reset_index(drop=True)
    texts = df["text"].fillna("").tolist()
    print(f"입력 {len(df):,}행")

    print("[1/5] 텍스트 통계")
    lex = build_sentiment_lexicon()
    f_text = review_text_features(texts, lex)
    print(f"      {time.time()-t0:.0f}초")

    print("[2/5] 리뷰 수준 행동")
    f_beh = review_behavior_features(df)
    print(f"      {time.time()-t0:.0f}초")

    print("[3/5] bigram 벡터화 (내용 유사도용)")
    vec = CountVectorizer(ngram_range=(1, 2), token_pattern=r"\b\w+\b",
                          min_df=5, max_features=200_000, binary=True, dtype=np.float32)
    X = vec.fit_transform(texts)
    Xn = normalize(X).tocsr()
    print(f"      어휘 {X.shape[1]:,} | {time.time()-t0:.0f}초")

    rd = f_beh["rating_deviation"].values
    lw = f_text["length_in_words"].values.astype("float64")

    print("[4/5] 작성자 수준")
    f_user = group_features(df, "user_id", Xn, rd, lw, "user")
    print(f"      {time.time()-t0:.0f}초")

    print("[5/5] 업체 수준")
    f_prod = group_features(df, "prod_id", Xn, rd, lw, "prod")
    print(f"      {time.time()-t0:.0f}초")

    feats = pd.concat([f_beh, f_text, f_user, f_prod], axis=1).astype("float32")
    feats = feats.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    feats.insert(0, "review_id", df["review_id"].values)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(OUT, index=False)

    groups = {
        "behavioral": list(f_beh.columns) + list(f_user.columns) + list(f_prod.columns),
        "text_derived": list(f_text.columns),
    }
    # 그룹 수준 피처 중 텍스트에서 파생된 것 재분류
    for c in list(groups["behavioral"]):
        if c.endswith(("content_sim", "avg_review_length")):
            groups["behavioral"].remove(c)
            groups["text_derived"].append(c)
    OUT_GROUPS.write_text(json.dumps(groups, indent=2, ensure_ascii=False), encoding="utf-8")

    nb, nt = len(groups["behavioral"]), len(groups["text_derived"])
    print(f"\n총 {feats.shape[1]-1}개 피처  =  행동 {nb}개 + 텍스트 파생 {nt}개 "
          f"({100*nt/(nb+nt):.0f}%)")
    print(f"저장: {OUT}  ({OUT.stat().st_size/1e6:.0f} MB)  |  총 {time.time()-t0:.0f}초")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
