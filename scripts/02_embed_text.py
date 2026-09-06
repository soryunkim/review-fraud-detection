"""
02_embed_text.py — 리뷰 텍스트 -> 문장 임베딩 (.npy)

reviews.parquet 의 행 순서를 그대로 따르는 (N, D) float32 배열을 만든다.
.npy 에는 ID가 없으므로 순서가 유일한 정합 수단이다. 절대 재정렬하지 말 것.

5,000건 단위로 중간 저장하므로, 중간에 끊겨도 다시 실행하면 남은 구간부터 이어간다.

사용:
    python scripts/02_embed_text.py                 # MiniLM(384), 약 4.7시간 (CPU 8코어)
    python scripts/02_embed_text.py --model bert    # BERT-base(768), 훨씬 오래 걸림
    python scripts/02_embed_text.py --limit 5000    # 짧게 동작 확인

임베딩은 전부 로컬 연산이다. API 호출도 토큰 소비도 없다.
(최초 1회 모델 다운로드에만 인터넷이 필요)
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "reviews.parquet"
PARTS = ROOT / "data" / "processed" / "_emb_parts"

MODELS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", 384),
    "bert": ("sentence-transformers/bert-base-nli-mean-tokens", 768),
}
CHUNK = 5_000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="minilm", choices=list(MODELS))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="앞 N건만 처리 (동작 확인용)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    name, dim = MODELS[args.model]
    out_path = ROOT / "data" / "processed" / f"emb_{args.model}_{dim}.npy"
    parts_dir = PARTS / args.model
    parts_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(SRC, columns=["review_id", "text"])
    assert df.review_id.is_monotonic_increasing, "review_id 순서가 깨졌습니다"
    texts = df["text"].fillna("").tolist()
    if args.limit:
        texts = texts[: args.limit]
    total = len(texts)

    print(f"모델 {name} (dim={dim}) | 대상 {total:,}건 | 청크 {CHUNK:,}")
    model = SentenceTransformer(name)

    t0 = time.time()
    done = 0
    for start in range(0, total, CHUNK):
        part = parts_dir / f"{start:08d}.npy"
        end = min(start + CHUNK, total)
        if part.exists():
            # 길이가 맞는 청크만 신뢰한다 (--limit 로 만든 조각 등 오염 방지)
            if np.load(part, mmap_mode="r").shape[0] == end - start:
                done = end
                continue
            print(f"  [재계산] {part.name}: 길이 불일치")
        emb = model.encode(
            texts[start:end], batch_size=args.batch_size, show_progress_bar=False
        )
        np.save(part, np.asarray(emb, dtype=np.float32))
        done = end
        el = time.time() - t0
        rate = (end - start) / max(el, 1e-9) if start == 0 else None
        eta = (total - done) / max(done / el, 1e-9) / 3600 if el > 0 else 0
        print(f"  {done:,} / {total:,}  ({100*done/total:5.1f}%)  경과 {el/60:.1f}분  남은 예상 {eta:.1f}시간", flush=True)

    parts = sorted(parts_dir.glob("*.npy"))
    emb = np.vstack([np.load(p) for p in parts])[:total]
    assert emb.shape[0] == total, f"행 수 불일치: {emb.shape[0]} != {total}"
    np.save(out_path, emb)
    print(f"\n저장: {out_path}  shape={emb.shape}  ({out_path.stat().st_size/1e6:.0f} MB)")
    print("중간 파일을 지우려면:", parts_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
