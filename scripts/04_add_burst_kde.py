"""
04_add_burst_kde.py — 버스트형 라벨 재정의 (Fei et al. 2013 기반)

01_build_reviews.py 가 만든 reviews.parquet 을 읽어서,
버스트형을 KDE+평점 방식으로 다시 정의한 컬럼을 추가한다.

[배경 — 왜 임시 버스트를 교체하나]
01_build_reviews.py 의 type_burst 는 전역 고정 임계값(7일/5건)이라,
버스트 구간 사기율이 오히려 전체 평균(13.2%)을 하회한다(약 11.6%).
YelpZip 식당 리뷰에서는 '리뷰가 몰리는 것'이 대부분 정상 인기 상승이지
조직적 사기가 아니기 때문이다 — Fei et al.(2013)이 지적한 문제.

[해결 — Fei 방식: 버스트 + 평점 조작 신호 결합]
버스트형 = ① 버스트(같은 상점 7일 내 3건 이상, 시간 집중)
         + ② 극단평점(1점 또는 5점)
         + ③ 평점이탈(상점 평균에서 1.5 이상 벗어남, Fei의 Rating Deviation)
→ 버스트형 사기율 28.1% (전체 평균의 2.1배), 표본 약 19,000건

[주의]
- 엄밀히는 '순수 버스트'가 아니라 '버스트+평점조작(coordinated burst)'이다.
  논문/발표에 이 정의를 명시할 것.
- 세 유형(신규/버스트/믹스)은 겹침을 허용한다. 각 유형별 탐지 실험을
  독립적으로 수행하므로 겹쳐도 문제없다.
- 전체 데이터에서 계산한다. review_id 순서를 보존한다.

출력:
  data/processed/reviews.parquet  (기존 + type_burst_kde, type_class_kde 컬럼)
  ※ 기존 type_burst(임시)도 남겨두어 비교 가능하게 함
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "reviews.parquet"   # 01이 만든 파일
OUT = ROOT / "data" / "processed" / "reviews.parquet"   # 덮어쓰기 (컬럼 추가)

# ── 버스트형 파라미터 (검증된 값) ──
BURST_WINDOW_DAYS = 7      # 같은 상점 7일 내
BURST_MIN_COUNT   = 3      # 3건 이상 (시간 집중)
RDEV_THRESHOLD    = 1.5    # 상점 평균 평점에서의 최소 이탈


def main():
    print("reviews.parquet 로드 중...")
    df = pd.read_parquet(SRC)
    df['date'] = pd.to_datetime(df['date'])
    assert df['review_id'].is_monotonic_increasing, "review_id 순서가 깨졌습니다"

    overall = df['fraud'].mean()
    print(f"  전체 {len(df):,}건, 사기율 {overall*100:.2f}%")

    # ── 신호 계산 (원래 순서 보존) ──
    work = df[['review_id', 'prod_id', 'rating', 'date']].copy()
    work = work.sort_values(['prod_id', 'date'])

    # 신호1: 버스트 (같은 상점 W일 내 리뷰 수)
    print("버스트 카운트 계산 중...")
    burst_counts = []
    for pid, sub in work.groupby('prod_id'):
        dates = sub['date'].values
        cnt = np.ones(len(dates))
        j = 0
        for i in range(len(dates)):
            while dates[i] - dates[j] > np.timedelta64(BURST_WINDOW_DAYS, 'D'):
                j += 1
            cnt[i] = i - j + 1
        burst_counts.append(pd.Series(cnt, index=sub.index))
    work['burst_count'] = pd.concat(burst_counts)

    # 신호2: 극단평점
    work['is_extreme'] = work['rating'].isin([1.0, 5.0]).astype(int)

    # 신호3: 평점 이탈 (상점 평균 대비 절대 편차)
    prod_avg = work.groupby('prod_id')['rating'].transform('mean')
    work['rating_dev'] = (work['rating'] - prod_avg).abs()

    # 버스트형 = 세 신호 모두 만족
    work['type_burst_kde'] = (
        (work['burst_count'] >= BURST_MIN_COUNT) &
        (work['is_extreme'] == 1) &
        (work['rating_dev'] > RDEV_THRESHOLD)
    ).astype('int8')

    # review_id 순서로 복원 후 원본에 병합
    work = work.sort_values('review_id')
    df['type_burst_kde'] = work.set_index('review_id')['type_burst_kde'].reindex(df['review_id']).values

    # ── 3유형 클래스 (겹침 허용이므로 참고용 — 실험은 각 유형 독립) ──
    # type_class_kde: 상호배타 버전(참고용). 실제 실험은 type_new, type_burst_kde 각각 사용.
    df['type_class_kde'] = np.select(
        [
            (df.type_new == 1) & (df.type_burst_kde == 0),
            (df.type_new == 0) & (df.type_burst_kde == 1),
            (df.type_new == 1) & (df.type_burst_kde == 1),
        ],
        ['new', 'burst', 'mixed'], default='general'
    )

    # ── 결과 요약 ──
    print("\n=== 버스트 재정의 결과 ===")
    print(f"  임시 type_burst:   {df.type_burst.sum():>7,}건, 사기율 {df[df.type_burst==1].fraud.mean()*100:.2f}%")
    print(f"  새 type_burst_kde: {df.type_burst_kde.sum():>7,}건, 사기율 {df[df.type_burst_kde==1].fraud.mean()*100:.2f}%")

    print("\n=== 3×3 매트릭스용 유형 (겹침 허용, 각각 독립 실험) ===")
    for name, mask in [
        ('신규계정형 (type_new)', df.type_new == 1),
        ('버스트형 (type_burst_kde)', df.type_burst_kde == 1),
        ('믹스형 (신규 AND 버스트)', (df.type_new == 1) & (df.type_burst_kde == 1)),
    ]:
        n = mask.sum()
        fr = df[mask].fraud.mean()
        print(f"  {name:>26}: {n:>8,}건, 사기율 {fr*100:.1f}% (×{fr/overall:.1f})")

    # ── 저장 (review_id 순서 유지) ──
    df.to_parquet(OUT, index=False)
    print(f"\n저장 완료: {OUT}")
    print(f"  추가된 컬럼: type_burst_kde, type_class_kde")
    print(f"  전체 컬럼: {list(df.columns)}")


if __name__ == "__main__":
    main()
