#!/usr/bin/env python3
"""RAG와 LLM 파인튜닝 실전 책용 이미지 생성 스크립트 (PIL 기반)."""
import os
import math
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.path.dirname(HERE), "assets")
os.makedirs(ASSETS, exist_ok=True)

BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
REG = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

BG = (13, 19, 38)
PANEL = (24, 33, 62)
CORAL = (224, 120, 86)
TEAL = (78, 205, 196)
BLUE = (96, 150, 255)
VIOLET = (167, 139, 250)
WHITE = (245, 247, 250)
MUTED = (154, 167, 199)
LINE = (70, 88, 140)
LIGHT_BG = (250, 251, 253)
INK = (30, 41, 59)
SUB = (100, 116, 139)
CARD_LINE = (203, 213, 225)


def font(path, size):
    return ImageFont.truetype(path, size, index=1)


def text_center(d, cx, y, s, f, fill):
    bb = d.textbbox((0, 0), s, font=f)
    w = bb[2] - bb[0]
    d.text((cx - w / 2 - bb[0], y), s, font=f, fill=fill)


def rrect(d, box, radius, fill, outline=None, width=1):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(d, x1, y1, x2, y2, fill=CORAL, width=4):
    d.line([x1, y1, x2, y2], fill=fill, width=width)
    ang = math.atan2(y2 - y1, x2 - x1)
    sz = 14
    for da in (2.6, -2.6):
        d.line([x2, y2, x2 + sz * math.cos(ang + da), y2 + sz * math.sin(ang + da)],
               fill=fill, width=width)


def box_with_label(d, x, y, w, h, title, sub=None, fill=(255, 255, 255),
                   title_size=30, sub_size=22):
    rrect(d, [x, y, x + w, y + h], 14, fill, outline=CARD_LINE, width=2)
    text_center(d, x + w / 2, y + 16, title, font(BOLD, title_size), INK)
    if sub:
        text_center(d, x + w / 2, y + 16 + title_size + 14, sub,
                    font(REG, sub_size), SUB)


# ---------------------------------------------------------------- 표지 1000x1300
def make_cover():
    W, H = 1000, 1300
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    for y in range(H):
        t = y / H
        r = int(13 + (30 - 13) * t)
        g = int(19 + (44 - 19) * t)
        b = int(38 + (82 - 38) * t)
        d.line([(0, y), (W, y)], fill=(r, g, b))
    d.ellipse([-180, -180, 320, 320], outline=(224, 120, 86, 90), width=3)
    d.ellipse([-120, -120, 260, 260], outline=(78, 205, 196, 70), width=2)
    d.ellipse([760, 980, 1180, 1400], outline=(96, 150, 255, 80), width=3)
    d.ellipse([820, 1040, 1120, 1340], outline=(224, 120, 86, 60), width=2)
    for i in range(6):
        x = 700 + i * 45
        d.line([(x, 0), (x + 220, 420)], fill=(255, 255, 255, 14), width=2)

    fb = lambda s: font(BOLD, s)
    fr = lambda s: font(REG, s)

    text_center(d, W / 2, 66, "W I K I D O C S   T E C H   B O O K", fr(26), CORAL)
    d.line([(120, 128), (880, 128)], fill=LINE, width=2)

    text_center(d, W / 2, 200, "RAG와 LLM", fb(96), WHITE)
    text_center(d, W / 2, 320, "파인튜닝 실전", fb(96), TEAL)

    text_center(d, W / 2, 500, "검색으로 지식을 연결하고", fb(42), WHITE)
    text_center(d, W / 2, 562, "학습으로 모델을 다듬는 실전서", fb(42), WHITE)

    rrect(d, [160, 680, 840, 748], 34, PANEL, outline=LINE, width=2)
    text_center(d, W / 2, 696, "언제 무엇을 쓸지까지 정리한다", fb(32), WHITE)

    topics = [
        "임베딩 · 벡터 DB와 청킹 전략",
        "하이브리드 검색 · 리랭킹 · 검색 평가",
        "LoRA · QLoRA와 학습 설정",
        "RAG vs 파인튜닝 선택 기준",
        "실전 프로젝트: 사내 문서 Q&A 챗봇",
    ]
    y = 800
    text_center(d, W / 2, y, "이 책에서 다루는 내용", fr(26), MUTED)
    y += 44
    for i, t in enumerate(topics):
        rrect(d, [170, y, 830, y + 46], 23, (20, 28, 54),
              outline=(60, 76, 120), width=1)
        d.ellipse([194, y + 11, 228, y + 45], fill=TEAL)
        fnum = fb(26)
        num = str(i + 1)
        bb = d.textbbox((0, 0), num, font=fnum)
        d.text((211 - (bb[2] - bb[0]) / 2 - bb[0], y + 13), num, font=fnum,
               fill=(255, 255, 255))
        d.text((244, y + 15), t, font=fr(24), fill=WHITE)
        y += 56

    d.line([(120, 1130), (880, 1130)], fill=LINE, width=2)
    text_center(d, W / 2, 1158, "저자  이준수", fr(30), MUTED)
    text_center(d, W / 2, 1202, "기준일  2026-10-09", fr(26), MUTED)

    img.save(os.path.join(ASSETS, "cover.png"))
    print("cover.png saved")


# ------------------------------------------------- RAG 파이프라인 1600x620
def make_rag_pipeline():
    W, H = 1600, 620
    img = Image.new("RGB", (W, H), LIGHT_BG)
    d = ImageDraw.Draw(img)
    text_center(d, W / 2, 24, "RAG 전체 파이프라인", font(BOLD, 40), INK)

    # 인덱싱 단계
    idx_boxes = [("문서 수집", "PDF · 웹 · 사내 문서"),
                 ("청킹", "400~512 토큰"),
                 ("임베딩", "벡터로 변환"),
                 ("벡터 DB 저장", "Chroma · Qdrant 등")]
    x = 60
    for title, sub in idx_boxes:
        box_with_label(d, x, 120, 260, 120, title, sub)
        x += 300
    for i in range(3):
        arrow(d, 60 + 260 + i * 300, 180, 60 + 300 + i * 300, 180, fill=TEAL)
    text_center(d, 690, 84, "1단계: 인덱싱 (미리 준비)", font(BOLD, 26), TEAL)

    # 검색 + 생성 단계
    q_boxes = [("질문 입력", "사용자 질문"),
               ("질문 임베딩", "같은 모델 사용"),
               ("유사도 검색", "Top-k 청크"),
               ("답변 생성", "LLM + 근거")]
    x = 60
    for title, sub in q_boxes:
        box_with_label(d, x, 360, 260, 120, title, sub,
                       fill=(255, 243, 235))
        x += 300
    for i in range(3):
        arrow(d, 60 + 260 + i * 300, 420, 60 + 300 + i * 300, 420, fill=CORAL)
    text_center(d, 690, 324, "2단계: 검색과 생성 (실시간)", font(BOLD, 26), CORAL)
    # 벡터 DB에서 유사도 검색으로 연결되는 점선 (L자)
    def dash_line(x1, y1, x2, y2):
        if x1 == x2:
            for yy in range(min(y1, y2), max(y1, y2), 16):
                d.line([(x1, yy), (x1, yy + 8)], fill=INK, width=3)
        else:
            for xx in range(min(x1, x2), max(x1, x2), 16):
                d.line([(xx, y1), (xx + 8, y1)], fill=INK, width=3)
    dash_line(1090, 240, 1090, 300)
    dash_line(1090, 300, 790, 300)
    dash_line(790, 300, 790, 360)

    img.save(os.path.join(ASSETS, "fig-rag-pipeline.png"))
    print("fig-rag-pipeline.png saved")


# ------------------------------------------------- 임베딩 벡터 공간 1200x620
def make_embedding():
    W, H = 1200, 620
    img = Image.new("RGB", (W, H), LIGHT_BG)
    d = ImageDraw.Draw(img)
    text_center(d, W / 2, 24, "임베딩: 비슷한 뜻은 벡터 공간에서 가까워진다",
                font(BOLD, 36), INK)

    # 축
    d.line([(120, 540), (1080, 540)], fill=CARD_LINE, width=2)
    d.line([(120, 120), (120, 540)], fill=CARD_LINE, width=2)

    groups = [
        ((300, 220), TEAL, ["반도체 공정 질문", "식각 장비 매뉴얼", "수율 개선 사례"]),
        ((850, 380), CORAL, ["점심 메뉴 추천", "맛집 지도", "레시피 모음"]),
        ((600, 430), BLUE, ["파이썬 문법 질문", "에러 로그 해석"]),
    ]
    for (cx, cy), color, labels in groups:
        import random
        random.seed(hash(labels[0]) % 1000)
        for i, lab in enumerate(labels):
            px = cx + random.randint(-90, 90)
            py = cy + random.randint(-60, 60)
            d.ellipse([px - 9, py - 9, px + 9, py + 9], fill=color)
            d.text((px + 14, py - 12), lab, font=font(REG, 20), fill=INK)
        d.ellipse([cx - 130, cy - 100, cx + 130, cy + 100], outline=color, width=2)

    # 코사인 유사도 설명
    rrect(d, [120, 120, 480, 210], 12, (255, 255, 255), outline=CARD_LINE, width=2)
    d.text((140, 134), "가까울수록 코사인 유사도가 높다", font=font(BOLD, 22), fill=INK)
    d.text((140, 166), "질문 벡터와 가장 가까운", font=font(REG, 20), fill=SUB)
    d.text((140, 192), "청크부터 순서대로 가져온다", font=font(REG, 20), fill=SUB)

    img.save(os.path.join(ASSETS, "fig-embedding.png"))
    print("fig-embedding.png saved")


# ------------------------------------------------- 청킹 전략 1500x680
def make_chunking():
    W, H = 1500, 680
    img = Image.new("RGB", (W, H), LIGHT_BG)
    d = ImageDraw.Draw(img)
    text_center(d, W / 2, 24, "청킹 전략 비교", font(BOLD, 40), INK)

    strategies = [
        ("고정 크기 청킹", "500토큰씩 일정하게 자르기",
         "간단하지만 문장 중간이 잘릴 수 있음", (250, 200, 200)),
        ("문장 경계 청킹", "문장 끝에서 자르기 + 오버랩",
         "의미 단위가 지켜져 검색 품질이 안정적", (200, 235, 220)),
        ("계층 청킹 (부모-자식)", "자식 256토큰으로 검색, 부모로 답변",
         "검색은 정밀하게, 답변은 맥락 있게", (200, 220, 250)),
    ]
    y = 110
    for title, desc, note, tint in strategies:
        rrect(d, [60, y, 1440, y + 150], 14, (255, 255, 255),
              outline=CARD_LINE, width=2)
        d.text((90, y + 18), title, font=font(BOLD, 30), fill=INK)
        d.text((90, y + 62), desc, font=font(REG, 24), fill=SUB)
        d.text((90, y + 100), note, font=font(REG, 24), fill=TEAL)
        # 청크 블록 시각화
        bx = 880
        if "고정" in title:
            for i in range(5):
                rrect(d, [bx + i * 105, y + 45, bx + i * 105 + 95, y + 105],
                      8, tint, outline=CARD_LINE)
        elif "문장" in title:
            widths = [150, 110, 150, 110]
            xx = bx
            for i, w in enumerate(widths):
                rrect(d, [xx, y + 45, xx + w, y + 105], 8, tint,
                      outline=CARD_LINE)
                xx += w + 12
        else:
            rrect(d, [bx, y + 30, bx + 480, y + 60], 8, tint, outline=CARD_LINE)
            for i in range(4):
                rrect(d, [bx + i * 122, y + 80, bx + i * 122 + 112, y + 120],
                      8, (255, 255, 255), outline=TEAL, width=2)
        y += 175

    img.save(os.path.join(ASSETS, "fig-chunking.png"))
    print("fig-chunking.png saved")


# ------------------------------------------------- 하이브리드 검색 1500x640
def make_hybrid():
    W, H = 1500, 640
    img = Image.new("RGB", (W, H), LIGHT_BG)
    d = ImageDraw.Draw(img)
    text_center(d, W / 2, 24, "하이브리드 검색과 리랭킹", font(BOLD, 40), INK)

    box_with_label(d, 620, 100, 260, 100, "사용자 질문", "질의 입력")
    # 두 갈래
    box_with_label(d, 180, 300, 300, 120, "벡터 검색", "의미 유사도 Top-50",
                   fill=(235, 248, 255))
    box_with_label(d, 1020, 300, 300, 120, "BM25 검색", "키워드 일치 Top-50",
                   fill=(255, 243, 235))
    arrow(d, 700, 200, 330, 300, fill=BLUE)
    arrow(d, 800, 200, 1170, 300, fill=CORAL)
    d.text((420, 250), "의미가 통하면", font=font(REG, 22), fill=SUB)
    d.text((1050, 250), "정확한 용어는", font=font(REG, 22), fill=SUB)

    box_with_label(d, 620, 300, 260, 120, "RRF 융합", "순위 기반 점수 합산")
    arrow(d, 480, 360, 620, 360, fill=BLUE)
    arrow(d, 1020, 360, 880, 360, fill=CORAL)

    box_with_label(d, 620, 480, 260, 110, "리랭킹", "크로스엔코더 Top 5~10")
    arrow(d, 750, 420, 750, 480, fill=TEAL)

    img.save(os.path.join(ASSETS, "fig-hybrid.png"))
    print("fig-hybrid.png saved")


# ------------------------------------------------- 에이전틱 RAG 1400x700
def make_agentic_rag():
    W, H = 1400, 700
    img = Image.new("RGB", (W, H), LIGHT_BG)
    d = ImageDraw.Draw(img)
    text_center(d, W / 2, 24, "에이전틱 RAG: 에이전트가 검색을 판단한다",
                font(BOLD, 36), INK)

    steps = [
        ("질문 접수", "사용자 질문을 받는다"),
        ("생각", "검색이 필요한지 판단한다"),
        ("도구 호출", "벡터 검색 도구를 실행한다"),
        ("관찰", "검색 결과를 읽고 평가한다"),
        ("답변", "충분하면 답변, 아니면 재검색"),
    ]
    cx, cy, rx, ry = 700, 400, 430, 190
    n = len(steps)
    for i, (title, sub) in enumerate(steps):
        ang = -math.pi / 2 + i * 2 * math.pi / n
        x = cx + rx * math.cos(ang)
        y = cy + ry * math.sin(ang)
        box_with_label(d, x - 150, y - 55, 300, 110, title, sub,
                       title_size=26, sub_size=19)
        ang2 = -math.pi / 2 + (i + 1) * 2 * math.pi / n
        x2 = cx + rx * math.cos(ang2)
        y2 = cy + ry * math.sin(ang2)
        mx = cx + (rx + 60) * math.cos((ang + ang2) / 2)
        my = cy + (ry + 60) * math.sin((ang + ang2) / 2)
        arrow(d, x, y, mx, my, fill=TEAL, width=3)
        arrow(d, mx, my, x2, y2, fill=TEAL, width=3)

    rrect(d, [520, 620, 880, 668], 12, (255, 255, 255), outline=CARD_LINE, width=2)
    text_center(d, 700, 632, "한 번 검색이 아니라 필요할 때까지 반복한다",
                font(REG, 22), SUB)

    img.save(os.path.join(ASSETS, "fig-agentic-rag.png"))
    print("fig-agentic-rag.png saved")


# ------------------------------------------------- LoRA 1400x640
def make_lora():
    W, H = 1400, 640
    img = Image.new("RGB", (W, H), LIGHT_BG)
    d = ImageDraw.Draw(img)
    text_center(d, W / 2, 24, "LoRA: 저랭크 분해로 가중치를 조금만 학습한다",
                font(BOLD, 34), INK)

    # W0 박스
    rrect(d, [80, 180, 380, 480], 14, (235, 240, 250), outline=CARD_LINE, width=2)
    text_center(d, 230, 300, "W0", font(BOLD, 54), INK)
    text_center(d, 230, 380, "원래 가중치", font(REG, 24), SUB)
    text_center(d, 230, 414, "고정 (학습 안 함)", font(REG, 24), SUB)

    d.text((430, 320), "+", font=font(BOLD, 60), fill=INK)

    # B x A 박스
    rrect(d, [520, 180, 700, 480], 14, (255, 243, 235), outline=CORAL, width=2)
    text_center(d, 610, 300, "B", font(BOLD, 54), CORAL)
    text_center(d, 610, 380, "d x r 행렬", font(REG, 24), SUB)
    text_center(d, 610, 414, "학습함", font(REG, 24), SUB)

    d.text((735, 320), "x", font=font(BOLD, 48), fill=INK)

    rrect(d, [800, 180, 980, 480], 14, (255, 243, 235), outline=CORAL, width=2)
    text_center(d, 890, 300, "A", font(BOLD, 54), CORAL)
    text_center(d, 890, 380, "r x k 행렬", font(REG, 24), SUB)
    text_center(d, 890, 414, "학습함", font(REG, 24), SUB)

    d.text((1030, 320), "=", font=font(BOLD, 60), fill=INK)

    rrect(d, [1090, 180, 1340, 480], 14, (235, 248, 255), outline=BLUE, width=2)
    text_center(d, 1215, 300, "W", font(BOLD, 54), BLUE)
    text_center(d, 1215, 380, "W0 + BA", font(REG, 24), SUB)
    text_center(d, 1215, 414, "추론용 가중치", font(REG, 24), SUB)

    rrect(d, [80, 520, 1340, 590], 12, (255, 255, 255), outline=CARD_LINE, width=2)
    text_center(d, 710, 534,
                "r이 작을수록 학습 파라미터가 적다 (예: r=16이면 전체의 1% 미만)",
                font(REG, 24), SUB)

    img.save(os.path.join(ASSETS, "fig-lora.png"))
    print("fig-lora.png saved")


# ------------------------------------------------- 파인튜닝 흐름 1600x560
def make_finetuning_flow():
    W, H = 1600, 560
    img = Image.new("RGB", (W, H), LIGHT_BG)
    d = ImageDraw.Draw(img)
    text_center(d, W / 2, 24, "파인튜닝 전체 흐름", font(BOLD, 40), INK)

    steps = [("데이터 준비", "인스트럭션 데이터 수집·정제"),
             ("SFT 학습", "LoRA·QLoRA로 지도 학습"),
             ("평가", "held-out 테스트로 검증"),
             ("DPO (선택)", "선호도 데이터로 정렬"),
             ("배포", "어댑터 병합 후 서빙")]
    colors = [TEAL, CORAL, BLUE, VIOLET, TEAL]
    x = 50
    for i, (title, sub) in enumerate(steps):
        box_with_label(d, x, 180, 240, 140, title, sub)
        d.line([(x, 180), (x + 240, 180)], fill=colors[i], width=8)
        x += 292
    for i in range(4):
        arrow(d, 50 + 240 + i * 292, 250, 50 + 292 + i * 292, 250, fill=INK)

    rrect(d, [50, 400, 1550, 500], 12, (255, 255, 255), outline=CARD_LINE, width=2)
    text_center(d, 800, 420, "학습 전 평가 베이스라인을 먼저 기록한다",
                font(BOLD, 26), INK)
    text_center(d, 800, 458, "베이스라인이 있어야 학습이 효과가 있었는지 알 수 있다",
                font(REG, 22), SUB)

    img.save(os.path.join(ASSETS, "fig-finetuning-flow.png"))
    print("fig-finetuning-flow.png saved")


# ------------------------------------------------- RAG vs 파인튜닝 1400x960
def make_rag_vs_ft():
    W, H = 1400, 960
    img = Image.new("RGB", (W, H), LIGHT_BG)
    d = ImageDraw.Draw(img)
    text_center(d, W / 2, 24, "RAG와 파인튜닝, 무엇을 쓸까", font(BOLD, 38), INK)

    box_with_label(d, 550, 100, 300, 90, "해결하고 싶은 문제", "하나씩 질문한다")
    arrow(d, 700, 190, 700, 240, fill=INK)

    # 질문 1
    rrect(d, [380, 240, 1020, 330], 14, (255, 255, 255), outline=CARD_LINE,
          width=2)
    text_center(d, 700, 256, "지식이 자주 바뀌는가?", font(BOLD, 28), INK)
    text_center(d, 700, 294, "예: 신제품 문서, 매일 바뀌는 규정", font(REG, 22), SUB)

    box_with_label(d, 80, 400, 360, 130, "예 -> RAG", "문서만 갈아끼우면 된다",
                   fill=(235, 248, 255))
    box_with_label(d, 960, 400, 360, 130, "아니오 -> 다음 질문", "지식은 안정적이다",
                   fill=(255, 243, 235))
    arrow(d, 480, 300, 260, 400, fill=TEAL)
    arrow(d, 920, 300, 1140, 400, fill=CORAL)

    # 질문 2 (오른쪽 흐름 아래)
    arrow(d, 1140, 530, 1140, 580, fill=INK)
    rrect(d, [880, 580, 1320, 680], 14, (255, 255, 255), outline=CARD_LINE,
          width=2)
    text_center(d, 1100, 596, "말투·형식을 고정해야 하는가?",
                font(BOLD, 24), INK)
    text_center(d, 1100, 630, "예: 보고서 양식, 상담 말투", font(REG, 20), SUB)

    box_with_label(d, 880, 730, 200, 120, "예", "파인튜닝",
                   fill=(245, 235, 255), title_size=26)
    box_with_label(d, 1120, 730, 200, 120, "아니오", "RAG로 시작",
                   fill=(235, 248, 255), title_size=26)
    arrow(d, 980, 680, 980, 730, fill=VIOLET)
    arrow(d, 1220, 680, 1220, 730, fill=TEAL)

    rrect(d, [80, 880, 1320, 936], 12, (255, 255, 255), outline=CARD_LINE,
          width=2)
    text_center(d, 700, 894, "둘 다 필요하면 RAG + 파인튜닝을 결합한다",
                font(REG, 24), SUB)

    img.save(os.path.join(ASSETS, "fig-rag-vs-ft.png"))
    print("fig-rag-vs-ft.png saved")


if __name__ == "__main__":
    make_cover()
    make_rag_pipeline()
    make_embedding()
    make_chunking()
    make_hybrid()
    make_agentic_rag()
    make_lora()
    make_finetuning_flow()
    make_rag_vs_ft()
    print("done")
