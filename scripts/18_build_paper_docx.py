"""
18_build_paper_docx.py — KSC 2026 투고용 .docx 생성 (오동진, 2026-10-02)

한국정보과학회 학술대회 양식(docs/paper/DocForm_1.docx)의 서식(함초롬돋움, 1단 제목·요약 /
2단 본문, A4 여백 위30·아래20·좌우10mm)을 그대로 쓰고 본문만 교체한다. 양식 규정상
참고문헌 포함 2~3쪽, 글자 9pt 이상, PDF 제출.

본문 내용은 docs/paper_draft.md 를 3쪽 분량으로 압축한 것이며, 수치는
results/bootstrap 의 rolling + sameday-fix 기준 최종값이다. 그림 1은
scripts/17_paper_figure.py 가 먼저 생성해야 한다.

사용법
  python scripts/17_paper_figure.py          # 그림 먼저
  python scripts/18_build_paper_docx.py      # docs/paper/KSC2026_초안.docx 생성
"""
import re, shutil, zipfile, os, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORM = ROOT / "docs" / "paper" / "DocForm_1.docx"
OUT = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "docs" / "paper" / "KSC2026_초안.docx")
TPL = tempfile.mkdtemp(prefix="ksc_tpl_")
with zipfile.ZipFile(FORM) as z:
    z.extractall(TPL)

F = '<w:rFonts w:ascii="함초롬돋움" w:eastAsia="함초롬돋움" w:hAnsi="함초롬돋움" w:cs="함초롬돋움"/>'
BODY_SZ = '<w:sz w:val="18"/><w:szCs w:val="18"/>'          # 9pt (양식 최소값)
SECT1 = ('<w:sectPr><w:pgSz w:w="11906" w:h="16838" w:code="9"/>'
         '<w:pgMar w:top="1701" w:right="567" w:bottom="1134" w:left="567" '
         'w:header="0" w:footer="0" w:gutter="0"/><w:cols w:space="720"/>'
         '<w:docGrid w:type="lines" w:linePitch="360"/></w:sectPr>')
SECT2 = ('<w:sectPr><w:type w:val="continuous"/><w:pgSz w:w="11906" w:h="16838" w:code="9"/>'
         '<w:pgMar w:top="1701" w:right="567" w:bottom="1134" w:left="567" '
         'w:header="0" w:footer="0" w:gutter="0"/><w:cols w:num="2" w:space="720"/>'
         '<w:docGrid w:type="lines" w:linePitch="280"/></w:sectPr>')
COLW = 4980
FIG_CX, FIG_CY = 3020000, 2000000   # 2단 한 칸 폭(트윕) — 11906 − 567*2 − 720 을 2로 나눈 값에서 여유분


def esc(t):
    return t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def run(text, bold=False, sz=None, italic=False):
    rpr = F + ('<w:b/><w:bCs/>' if bold else '') + ('<w:i/>' if italic else '') + (sz or BODY_SZ)
    return (f'<w:r><w:rPr>{rpr}</w:rPr>'
            f'<w:t xml:space="preserve">{esc(text)}</w:t></w:r>')


def para(text='', bold=False, sz=None, center=False, indent=True, line=300,
         space_after=0, keep_next=False, sectpr=''):
    p = ['<w:p><w:pPr><w:wordWrap/>']
    if line:
        p.append(f'<w:spacing w:line="{line}" w:lineRule="exact"'
                 + (f' w:after="{space_after}"' if space_after else '') + '/>')
    if indent:
        p.append('<w:ind w:firstLineChars="100" w:firstLine="180"/>')
    if center:
        p.append('<w:jc w:val="center"/>')
    else:
        p.append('<w:jc w:val="both"/>')
    if keep_next:
        p.append('<w:keepNext/>')
    p.append(f'<w:rPr>{F}{sz or BODY_SZ}</w:rPr>')
    if sectpr:
        p.append(sectpr)
    p.append('</w:pPr>')
    if text:
        p.append(run(text, bold=bold, sz=sz))
    p.append('</w:p>')
    return ''.join(p)


def heading(text, sz='<w:sz w:val="20"/><w:szCs w:val="20"/>'):
    return ('<w:p><w:pPr><w:wordWrap/><w:spacing w:before="120" w:line="300" w:lineRule="exact"/>'
            f'<w:rPr>{F}<w:b/><w:bCs/>{sz}</w:rPr></w:pPr>'
            f'{run(text, bold=True, sz=sz)}</w:p>')


def cell(text, w, bold=False, center=False, shade=False):
    sh = '<w:shd w:val="clear" w:color="auto" w:fill="F2F2F2"/>' if shade else ''
    jc = '<w:jc w:val="center"/>' if center else '<w:jc w:val="left"/>'
    sz = '<w:sz w:val="16"/><w:szCs w:val="16"/>'
    return (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{sh}'
            '<w:vAlign w:val="center"/></w:tcPr>'
            f'<w:p><w:pPr><w:spacing w:line="240" w:lineRule="exact"/>{jc}'
            f'<w:rPr>{F}{sz}</w:rPr></w:pPr>{run(text, bold=bold, sz=sz)}</w:p></w:tc>')


def table(rows, widths):
    grid = ''.join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    body = []
    for i, r in enumerate(rows):
        tcs = ''.join(cell(c, widths[j], bold=(i == 0), center=(i == 0 or j > 0), shade=(i == 0))
                      for j, c in enumerate(r))
        trpr = ('<w:trPr><w:cantSplit/>' + ('<w:tblHeader/>' if i == 0 else '') + '</w:trPr>')
        body.append(f'<w:tr>{trpr}{tcs}</w:tr>')
    borders = ('<w:tblBorders>'
               + ''.join(f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
                         for s in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'))
               + '</w:tblBorders>')
    return (f'<w:tbl><w:tblPr><w:tblW w:w="{sum(widths)}" w:type="dxa"/>'
            f'<w:jc w:val="center"/>{borders}'
            '<w:tblCellMar><w:left w:w="40" w:type="dxa"/><w:right w:w="40" w:type="dxa"/>'
            '</w:tblCellMar></w:tblPr>'
            f'<w:tblGrid>{grid}</w:tblGrid>{"".join(body)}</w:tbl>')


def caption(text, before=True):
    sz = '<w:sz w:val="16"/><w:szCs w:val="16"/>'
    keep = '<w:keepNext/><w:keepLines/>' if before else ''
    return ('<w:p><w:pPr><w:wordWrap/>' + keep +
            f'<w:spacing w:{"before" if before else "after"}="80" w:line="260" w:lineRule="exact"/>'
            f'<w:jc w:val="center"/><w:rPr>{F}{sz}</w:rPr></w:pPr>{run(text, sz=sz)}</w:p>')



def picture(rel_id, cx, cy, name="그림"):
    """단 폭에 맞춘 인라인 이미지 문단. cx, cy 는 EMU."""
    return ('<w:p><w:pPr><w:keepNext/><w:jc w:val="center"/>'
            '<w:spacing w:before="120" w:line="240" w:lineRule="auto"/></w:pPr>'
            '<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
            f'<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="1" name="{name}"/>'
            '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:nvPicPr><pic:cNvPr id="1" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            '<pic:spPr><a:xfrm><a:off x="0" y="0"/>'
            f'<a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
            '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')


# ───────────────────────── 본문 내용 ─────────────────────────
TITLE_KR = "그래프 신경망 기반 사기 탐지의 조건적 유효성"
TITLE_KR2 = "― 시간적 데이터 누수 통제 환경에서의 실증 분석 ―"
TITLE_EN1 = "Conditional Effectiveness of Graph Neural Network-based"
TITLE_EN2 = "Fraud Detection under Temporal Data Leakage Control"

ABSTRACT = (
    "온라인 리뷰 사기 탐지에서 그래프 신경망(GNN)은 리뷰 간 관계 정보를 활용해 탐지 성능을 높이는 "
    "방법으로 널리 쓰인다. 그러나 대표적인 선행 연구들은 리뷰를 무작위로 분할해 평가하므로, 채점 대상보다 "
    "나중에 작성된 같은 작성자의 리뷰가 학습 그래프에 유입되는 시간적 데이터 누수를 통제하지 못한다. "
    "본 연구는 YelpZip 2014년 리뷰 180,659건에 대해 피처·유형 라벨·그래프 연결을 모두 작성 시점까지의 "
    "정보로만 재구성하고 시간 순서로 평가해 이를 통제한 뒤, 관계 정보의 기여를 그래프를 사용한 모델과 "
    "동일 피처·무그래프 모델의 PR-AUC 차이로 측정했다. 그 결과 기여는 세 조건으로 뚜렷이 갈렸다. 과거 "
    "이력이 있는 작성자의 리뷰에 같은 작성자 관계를 적용하면 +0.110(95% 신뢰구간 [+0.089, +0.133])으로 "
    "크게 증폭되지만(성공 조건), 같은 가게·같은 평점 및 같은 가게·같은 시기 관계는 이웃 대부분이 정상 "
    "리뷰여서 각각 −0.026, −0.035로 오히려 성능을 떨어뜨렸으며(실패 조건), 과거 이력이 없는 첫 리뷰에서는 "
    "이력 기반 관계의 기여가 관측되지 않았다(한계 조건). 또한 누수를 통제하지 않은 평가에서 관측되는 "
    "기여의 28%가 미래 정보 유입에서 기인함을 정량화했다. 본 연구는 그래프 구조가 보편적으로 유효하다는 "
    "전제를 반증하고, 과거 이력의 유무와 이웃 관계의 동질성에 따른 조건적 유효성을 실증한다."
)
KEYWORDS = "주제어: 그래프 신경망, 리뷰 사기 탐지, 데이터 누수, 호모필리, 콜드 스타트"

parts = []
# ── 1단 영역 ──
parts.append(para(TITLE_KR, bold=True, sz='<w:sz w:val="32"/><w:szCs w:val="32"/>',
                  center=True, indent=False, line=400))
parts.append(para(TITLE_KR2, bold=True, sz='<w:sz w:val="24"/><w:szCs w:val="24"/>',
                  center=True, indent=False, line=320))
parts.append(para('', indent=False, line=200))
parts.append(para(TITLE_EN1, bold=True, sz='<w:sz w:val="24"/><w:szCs w:val="24"/>',
                  center=True, indent=False, line=300))
parts.append(para(TITLE_EN2, bold=True, sz='<w:sz w:val="24"/><w:szCs w:val="24"/>',
                  center=True, indent=False, line=300))
parts.append(para('', indent=False, line=200))
parts.append(para("요   약", bold=True, center=True, indent=False, line=280,
                  sz='<w:sz w:val="20"/><w:szCs w:val="20"/>'))
parts.append(para(ABSTRACT, line=260))
parts.append(para(KEYWORDS, line=260, indent=False))
parts.append(para('', indent=False, line=140, sectpr=SECT1))

# ── 2단 영역 ──
parts.append(heading("1. 서  론"))
for t in [
    "온라인 리뷰는 소비자의 구매 결정과 플랫폼 신뢰에 직접 영향을 미치지만, 조작된 가짜 리뷰는 소비자를 "
    "오도하고 정상 사업자와의 공정한 경쟁을 저해한다. 이를 막기 위해 리뷰·작성자·상점 사이의 관계를 "
    "그래프로 모델링하고 그래프 신경망(GNN)으로 탐지 성능을 높이는 연구가 다수 축적되었다[3,4,5].",
    "그러나 이 흐름에는 두 가지 구조적 한계가 있다. 첫째, 대다수의 GNN 기반 탐지 모델은 이웃이 자신과 "
    "같은 라벨을 가진다는 호모필리(동질성)를 암묵적으로 전제한다. 같은 가게에 비슷한 시기·비슷한 평점으로 "
    "작성된 리뷰처럼 이웃이 수십 개씩 붙는 관계는 대부분 정상 리뷰로 구성되어 사기 신호가 묻힌다. 또한 "
    "과거 이력이 없는 신규 작성자(콜드 스타트)는 참조할 관계 자체가 존재하지 않는다.",
    "둘째, 기존 연구의 다수는 리뷰를 무작위로 나누어 학습·평가한다. 같은 작성자 관계(R-U-R)를 사용하는 "
    "모델에서 이 방식은 채점 대상보다 미래에 작성된 같은 작성자의 리뷰가 학습 그래프에 포함되는 시간적 "
    "데이터 누수를 구조적으로 발생시켜, 관계 정보의 실제 기여를 과대평가하게 만든다.",
    "본 연구는 그래프가 무조건 성능을 높인다는 전제를 검증 대상으로 삼아, 관계 정보가 어떤 조건에서 실제로 "
    "작동하는지를 통제된 환경에서 실증한다. 기여는 다음 세 가지다. (1) 피처·유형 라벨·그래프 연결을 모두 "
    "작성 시점까지의 정보로 재구성하고 시간 순서 3회 분할로 평가해 누수를 통제했으며, 무통제 평가에서 "
    "관측되는 기여의 28%가 미래 정보 유입임을 정량화했다. (2) 이 환경에서 관계 정보의 유효성이 과거 이력 "
    "유무와 이웃 관계의 동질성에 따라 성공·실패·한계 세 조건으로 갈림을 보였다. (3) 이를 바탕으로 조건부 "
    "하이브리드 구조라는 향후 설계 방향을 제시한다.",
]:
    parts.append(para(t))

parts.append(heading("2. 선행 연구"))
parts.append(heading("2.1 무작위 분할 기반 평가", '<w:sz w:val="18"/><w:szCs w:val="18"/>'))
parts.append(para(
    "리뷰 사기 탐지의 대표적 GNN 모델들은 공통적으로 리뷰(노드)를 무작위로 층화 분할해 평가한다. "
    "CARE-GNN[3]은 공개 구현에서 층화 무작위 추출로 학습 40%·평가 60%를 나누며, PC-GNN[4]과 "
    "BWGNN[5] 역시 같은 방식을 사용한다. 세 모델 모두 작성 시각을 분할에 사용하지 않는다. 이들이 사용하는 "
    "YelpChi는 본 연구와 동일하게 R-U-R·R-S-R·R-T-R 관계로 구성되므로, 리뷰를 무작위로 나누는 순간 "
    "채점 대상보다 나중에 작성된 같은 작성자의 리뷰가 학습 그래프에 포함된다. 본 연구는 이 유입분이 관측 "
    "기여의 28%에 해당함을 정량적으로 보인다(3.4절)."))
parts.append(heading("2.2 호모필리 가정", '<w:sz w:val="18"/><w:szCs w:val="18"/>'))
parts.append(para(
    "선행 연구들도 호모필리 전제가 관계에 따라 성립하지 않음을 인지하고 있다. CARE-GNN[3]은 관계별 라벨 "
    "유사도가 크게 갈린다는 점을 사기꾼의 관계 위장으로 설명하고, BWGNN[5]은 이상치가 그래프 스펙트럼을 "
    "고주파 쪽으로 이동시킨다는 관측에서 저주파 가정의 한계를 지적하며, Wang 등[6]은 낮은 호모필리 "
    "환경에서 기존 탐지기의 일반화 실패를 보인다. 그러나 이들은 호모필리의 결여를 모델 구조 개선의 동기로 "
    "삼을 뿐, 관계 정보의 기여가 관계 종류와 대상 리뷰의 성격에 따라 어떻게 갈리는지를 분해해 제시하지 "
    "않는다. 특히 과거 이력이 없는 콜드 스타트 리뷰에서 관계 정보가 갖는 값은 다루어지지 않는다. 본 연구는 "
    "두 갭을 하나의 통제된 설계로 동시에 다룬다."))

parts.append(heading("3. 데이터 및 실험 설계"))
parts.append(heading("3.1 데이터와 유형 정의", '<w:sz w:val="18"/><w:szCs w:val="18"/>'))
parts.append(para(
    "YelpZip의 2014년 리뷰 180,659건을 사용한다. 정답은 Yelp 필터가 걸러낸 리뷰(12.7%)로, 사기로 "
    "단정하지 않고 '걸러진 리뷰'로 표기한다. 작성 시점까지의 이력을 기준으로 작성자의 첫 리뷰를 저활동형"
    "(89,530건, 걸러진 비율 20.9%), 이전 리뷰가 있는 작성자의 리뷰를 비저활동형(91,129건, 4.6%)으로 "
    "나눈다. 작성자가 가입 첫날 여러 건을 쓴 경우 날짜가 일 단위여서 선행 관계가 불분명하므로, 해당 "
    "10,797건은 모두 저활동형으로 분류했다."))
parts.append(heading("3.2 관계와 노드 피처", '<w:sz w:val="18"/><w:szCs w:val="18"/>'))
parts.append(para(
    "관계는 같은 작성자(R-U-R), 같은 가게·같은 평점(R-S-R), 같은 가게·같은 달(R-T-R) 세 가지이며, 모두 "
    "과거 방향으로만 연결한다. 각 리뷰는 자신보다 먼저 작성된 리뷰에서만 정보를 받으므로 그래프 구조 자체에 "
    "미래 정보가 들어오지 않는다. 노드 피처는 Rayana와 Akoglu[1]가 정의한 37개를 사용하되 전부 작성 "
    "시점까지의 이력만으로 재계산했다. 텍스트 임베딩(384차원)을 추가하거나 대체하는 조건도 실측했으나 두 "
    "경우 모두 성능이 저하되어(비저활동형 각각 −0.059, −0.141) 수작업 피처만 사용한다."))
parts.append(heading("3.3 모델과 평가", '<w:sz w:val="18"/><w:szCs w:val="18"/>'))
parts.append(para(
    "백본은 GraphSAGE[2]를 사용한다. 자기 피처를 이웃 집계와 분리해 변환하므로, 호모필리가 약한 관계를 "
    "가져올 때도 리뷰 자신의 신호가 희석되지 않는다. 본 연구의 목적은 최신 아키텍처와의 성능 경쟁이 아니라 "
    "기본적인 구조에서 관계 정보의 조건적 유효성을 확인하는 것이므로 모든 실험을 단일 백본으로 통일했다. "
    "평가는 2014년 1~9월을 학습, 10~11월을 조기 종료 기준, 10~12월을 한 달씩 밀어 3회 채점하며, 모델마다 "
    "5회 반복한다. 관계 정보의 기여는 그래프를 사용한 모델과 동일 피처·무그래프 모델의 PR-AUC 차이로 "
    "정의하고, 같은 리뷰로 짝지은 부트스트랩으로 95% 신뢰구간을 구한다."))
parts.append(heading("3.4 데이터 누수의 통제와 정량화", '<w:sz w:val="18"/><w:szCs w:val="18"/>'))
parts.append(caption("표 1  학습 데이터 조건별 관계 정보 기여"))
parts.append(table([
    ["학습 데이터 조건", "기여(ΔPR-AUC)"],
    ["A. 원본(무작위 분할에 해당)", "+0.129"],
    ["B. 채점 대상 작성자의 미래 리뷰 제거", "+0.093"],
    ["C. 채점 대상 작성자의 리뷰 전부 제거", "+0.001"],
], [3300, 1680]))
parts.append(para(
    "채점 대상을 고정한 채 학습 데이터에서 정보를 단계적으로 제거해 누수 규모를 분해했다(표 1). A−B"
    "(0.036)는 전체의 28%로 미래 정보 유입에 의한 거품이고, B−C(0.092, 72%)는 실제 운영 환경에서도 "
    "사용할 수 있는 과거 이력 효과다. 학습 데이터를 무작위로 40% 줄인 대조 실험에서는 기여가 17%만 "
    "감소해, 작성자 이력을 표적으로 제거했을 때에만 성능이 무너짐을 확인했다. 즉 28%는 데이터 양이 아니라 "
    "미래 정보 자체의 효과다."))

parts.append(heading("4. 실험 결과: 성공·실패·한계 조건"))
parts.append(caption("표 2  조건별 관계 정보의 기여"))
parts.append(table([
    ["조건", "유형 · 관계", "ΔPR-AUC", "95% 신뢰구간"],
    ["성공", "비저활동 · R-U-R", "+0.110", "[+0.089, +0.133]"],
    ["실패", "비저활동 · R-S-R", "−0.026", "[−0.038, −0.015]"],
    ["실패", "비저활동 · R-T-R", "−0.035", "[−0.047, −0.022]"],
    ["한계", "저활동 · R-U-R", "−0.005", "[−0.010, +0.000]"],
], [620, 1800, 1000, 1560]))
parts.append(para(
    "표 2는 세 조건을 요약한다. 과거 이력이 있는 작성자의 리뷰에서 같은 작성자 관계는 기여가 +0.110으로 "
    "가장 크다(성공 조건). 과거 이력이 앵커가 되어 관계 정보가 증폭되는 경우다. 반면 같은 가게 기반 관계는 "
    "이웃이 평균 34개까지 붙지만 그 대부분이 정상 리뷰여서 신호가 묻히고, 기여가 음수가 된다(실패 조건). "
    "과거 이력이 없는 첫 리뷰에서는 같은 작성자 관계가 구조적으로 성립하지 않아 기여가 관측되지 않는다"
    "(한계 조건)."))
parts.append(heading("4.1 성공 조건의 세부", '<w:sz w:val="18"/><w:szCs w:val="18"/>'))
parts.append(para(
    "이력이 있어도 기여는 균일하지 않다. 평점이 가게 평균과 비슷한 평범한 리뷰에서는 기여가 +0.123"
    "([+0.100, +0.145])인 반면, 평점이 가게 평균에서 크게 벗어난 리뷰(2014년 이전 분포의 양 끝 10%)에서는 "
    "+0.038([+0.005, +0.072])로 1/3 수준에 그친다. 두 값 모두 유의하지만 그 차이 또한 유의하다"
    "(+0.086, [+0.044, +0.123]). 평점 이탈 구간은 그래프 없이 측정한 성능 자체도 낮아(0.153 대 0.203) "
    "걸러진 비율이 더 높음에도(5.98% 대 4.62%) 피처로도 관계로도 덜 탐지되는 가장 어려운 구간이다. "
    "행동 패턴 축의 또 다른 후보였던 버스트형(가게의 평소 발생량 대비 포아송 검정)은 분위수가 아닌 "
    "유의수준으로 정의되어 동일한 확장 기준이 없고, 대응되는 완화(p<0.05)에서도 채점 대상 중 걸러진 리뷰가 "
    "145건에 그쳐 신뢰구간 반폭이 ±0.065에 달해 부호조차 판정할 수 없었다. 따라서 행동 패턴 축은 평점 "
    "이탈형으로 단일화했다."))
parts.append(heading("4.2 한계 조건과 그 예외", '<w:sz w:val="18"/><w:szCs w:val="18"/>'))
parts.append(para(
    "저활동형은 정의상 같은 작성자의 과거 리뷰가 없어 이력에 의존하는 관계가 무의미하다. 그러나 이력에 "
    "의존하지 않는 상점 기반 관계는 저활동형에서도 작지만 유의한 기여를 보인다. 같은 가게·같은 평점은 "
    "+0.0085([+0.0016, +0.0152]), 같은 가게·같은 달은 +0.0063([−0.0000, +0.0124])이다. 즉 한계 조건은 "
    "관계 정보 전체가 무의미하다는 뜻이 아니라, 이력을 앵커로 삼는 관계만 무의미하다는 뜻으로 정교화해야 "
    "한다. 다만 이 기여의 절대 크기는 작다(PR-AUC 0.374→0.382, 상대 2.3%). 채점 대상 중 걸러진 리뷰가 "
    "5,316건으로 많아 통계적으로 유의하게 관측되는 것이며, 성공 조건의 상대 증가율(약 60%)의 1/13 "
    "수준이다. 이웃 수를 수십 배로 늘려 얻은 개선이 작성자 이력 한 가지가 주는 개선에 크게 못 미친다는 "
    "것으로, 연결의 양이 아니라 그 연결이 어떤 종류의 앵커를 제공하는지가 관건임을 보인다."))
parts.append(picture("rId100", FIG_CX, FIG_CY, "그림 1"))
parts.append(caption("그림 1  유형·관계별 관계 정보의 기여(95% 신뢰구간)", before=False))
parts.append(heading("4.3 강건성", '<w:sz w:val="18"/><w:szCs w:val="18"/>'))
parts.append(para(
    "같은 날 작성된 리뷰의 순서 모호성 처리를 3.1절의 방식으로 바꾸면 세 조건 모두 더 뚜렷해지는 방향으로 "
    "변한다. 성공 조건은 +0.065에서 +0.110으로, 실패 조건은 −0.011·−0.022에서 −0.026·−0.035로 "
    "바뀌었고, 한계 조건은 0 근방을 유지했다. 재분류로 이동한 10,797건의 걸러진 비율이 비저활동형 평균보다 "
    "높아 해당 군의 신호가 깨끗해진 결과로 해석한다. 조건적 유효성의 결론은 동률 처리 방식에 강건하다."))

parts.append(heading("5. 결론 및 향후 연구"))
parts.append(para(
    "본 연구는 시간적 데이터 누수를 통제한 환경에서 GNN 기반 사기 탐지의 관계 정보 기여가 보편적이지 않고, "
    "과거 이력의 유무와 이웃 관계의 동질성에 따라 성공·실패·한계 세 조건으로 갈림을 실증했다. 이는 GNN이 "
    "사기 탐지에 효과적이라는 통념에 대한 반증이라기보다, 관계 정보를 언제 어떻게 적용해야 하는지에 대한 "
    "사용 지침에 해당한다."))
parts.append(para(
    "향후 시스템은 이 조건성을 반영한 조건부 하이브리드 구조로 발전할 필요가 있다. 이력이 없는 콜드 스타트 "
    "리뷰에는 관계 정보 대신 피처 기반 앙상블을 우선 적용하고, 호모필리가 약한 이웃에서 신호가 묻히는 "
    "문제에는 직접 이웃만 보는 1-hop 구조를 넘어서는 멀티홉 또는 스펙트럴 계열 GNN[5]의 도입이 요구된다. "
    "본 연구의 한계로, 데이터 구득의 어려움 때문에 비교적 오래된 2014년 데이터를 사용했으므로 최신 플랫폼 "
    "데이터에서의 재현이 필요하다."))

parts.append(heading("참고문헌"))
REFS = [
    '[1] S. Rayana and L. Akoglu, "Collective opinion spam detection: Bridging review networks and '
    'metadata," Proc. ACM SIGKDD, 2015.',
    '[2] W. L. Hamilton, R. Ying, and J. Leskovec, "Inductive representation learning on large '
    'graphs," Proc. NeurIPS, 2017.',
    '[3] Y. Dou, Z. Liu, L. Sun, Y. Deng, H. Peng, and P. S. Yu, "Enhancing graph neural '
    'network-based fraud detectors against camouflaged fraudsters," Proc. ACM CIKM, 2020.',
    '[4] Y. Liu, X. Ao, Z. Qin, J. Chi, J. Feng, H. Yang, and Q. He, "Pick and choose: A GNN-based '
    'imbalanced learning approach for fraud detection," Proc. WWW, 2021.',
    '[5] J. Tang, J. Li, Z. Gao, and J. Li, "Rethinking graph neural networks for anomaly '
    'detection," Proc. ICML, 2022.',
    '[6] Y. Wang, J. Zhang, Z. Huang, W. Li, S. Feng, Z. Ma, Y. Sun, D. Yu, F. Dong, J. Jin, '
    'B. Wang, and J. Luo, "Label information enhanced fraud detection against low homophily in '
    'graphs," Proc. WWW, 2023.',
    '[7] G. Fei, A. Mukherjee, B. Liu, M. Hsu, M. Castellanos, and R. Ghosh, "Exploiting burstiness '
    'in reviews for review spammer detection," Proc. ICWSM, 2013.',
]
for r in REFS:
    parts.append('<w:p><w:pPr><w:wordWrap/><w:spacing w:line="240" w:lineRule="exact"/>'
                 '<w:ind w:left="200" w:hanging="200"/><w:jc w:val="both"/>'
                 f'<w:rPr>{F}<w:sz w:val="16"/><w:szCs w:val="16"/></w:rPr></w:pPr>'
                 + run(r, sz='<w:sz w:val="16"/><w:szCs w:val="16"/>') + '</w:p>')

doc = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n'
       '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
       'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
       'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
       'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
       '<w:body>' + ''.join(parts) + SECT2 + '</w:body></w:document>')

FIG = str(ROOT / "docs" / "paper" / "fig1_contribution.png")
work = os.path.join(tempfile.mkdtemp(prefix='ksc_build_'), 'build')
if os.path.exists(work):
    shutil.rmtree(work)
shutil.copytree(TPL, work)
with open(os.path.join(work, 'word', 'document.xml'), 'w', encoding='utf-8') as f:
    f.write(doc)

# 이미지 파일·관계·콘텐츠 타입 등록
os.makedirs(os.path.join(work, 'word', 'media'), exist_ok=True)
shutil.copy(FIG, os.path.join(work, 'word', 'media', 'image1.png'))
rels_p = os.path.join(work, 'word', '_rels', 'document.xml.rels')
rels = open(rels_p, encoding='utf-8').read()
if 'rId100' not in rels:
    rels = rels.replace('</Relationships>',
        '<Relationship Id="rId100" Type="http://schemas.openxmlformats.org/officeDocument/'
        '2006/relationships/image" Target="media/image1.png"/></Relationships>')
    open(rels_p, 'w', encoding='utf-8').write(rels)
ct_p = os.path.join(work, '[Content_Types].xml')
ct = open(ct_p, encoding='utf-8').read()
if 'Extension="png"' not in ct:
    i = ct.index('<Types')
    j = ct.index('>', i) + 1
    ct = ct[:j] + '<Default Extension="png" ContentType="image/png"/>' + ct[j:]
    open(ct_p, 'w', encoding='utf-8').write(ct)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
if os.path.exists(OUT):
    os.remove(OUT)
zf = zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED)
for root, _, files in os.walk(work):
    for fn in files:
        p = os.path.join(root, fn)
        zf.write(p, os.path.relpath(p, work).replace('\\', '/'))
zf.close()
print('wrote', OUT)
