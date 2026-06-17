"""
轻量知识库构建器（适配 Render 512MB 内存限制）
Excel → jieba 分词 → TF-IDF 向量 → pickle 持久化
"""
import pickle
import hashlib
import re
from pathlib import Path

import openpyxl
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import jieba_tokenize

# ============================================================
# 配置
# ============================================================
EXCEL_PATH = "亚瑟爱数学考研数学规划.xlsx"
KB_PATH = "knowledge_base.pkl"

PLAN_SHEETS = {
    "张宇": "张宇",
    "武忠祥A1": "武忠祥（方案A1）",
    "武忠祥A2": "武忠祥（方案A2）",
}

SUBJECT_COL_MAP = {1: "高数", 2: "线代", 3: "概率论"}


def safe_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text in ("\\", "-", "—", "/"):
        return ""
    return text


# ============================================================
# 解析 Excel（逻辑和之前完全一样）
# ============================================================
def parse_plan_sheet(ws, teacher_name: str) -> list[dict]:
    chunks = []
    score_targets = safe_text(ws.cell(1, 5).value)

    for row_idx in range(2, 7):
        stage_name = safe_text(ws.cell(row_idx, 1).value)
        if not stage_name:
            continue
        for col_idx in range(1, 4):
            content = safe_text(ws.cell(row_idx, col_idx + 1).value)
            if not content:
                continue
            subject = SUBJECT_COL_MAP.get(col_idx, "通用")
            chunk_text = f"【{teacher_name}】{stage_name} — {subject}\n{content}"
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    "type": "规划", "teacher": teacher_name,
                    "stage": stage_name.split("\n")[0], "subject": subject,
                    "score_targets": score_targets[:200],
                }
            })

    # 真题
    zhenti = safe_text(ws.cell(8, 2).value)
    if zhenti:
        chunks.append({
            "content": f"【{teacher_name}】真题阶段\n{zhenti}",
            "metadata": {"type": "规划", "teacher": teacher_name, "stage": "真题", "subject": "全部"}
        })

    # 模拟卷
    moni = safe_text(ws.cell(9, 2).value)
    if moni:
        chunks.append({
            "content": f"【{teacher_name}】模拟卷阶段\n{moni}",
            "metadata": {"type": "规划", "teacher": teacher_name, "stage": "模拟卷", "subject": "全部"}
        })

    # 分数目标
    if score_targets:
        chunks.append({
            "content": f"【{teacher_name}】分数目标与总体规划思路\n{score_targets}",
            "metadata": {"type": "规划", "teacher": teacher_name, "stage": "分数目标", "subject": "全部"}
        })

    return chunks


def parse_teacher_review_sheet(ws) -> list[dict]:
    chunks = []
    current_subject = ""
    for row_idx in range(2, ws.max_row + 1):
        sv = safe_text(ws.cell(row_idx, 1).value)
        if sv:
            current_subject = sv
        teacher = safe_text(ws.cell(row_idx, 2).value)
        review = safe_text(ws.cell(row_idx, 3).value)
        if not teacher or not review:
            continue
        chunks.append({
            "content": f"【名师测评】{current_subject} — {teacher}\n{review}",
            "metadata": {"type": "名师测评", "subject": current_subject, "teacher": teacher}
        })
    return chunks


def parse_book_review_sheet(ws) -> list[dict]:
    chunks = []
    for row_idx in range(1, ws.max_row + 1):
        raw = safe_text(ws.cell(row_idx, 1).value)
        if not raw:
            continue
        cat = "习题册" if row_idx == 1 else "模拟卷"
        entries = re.split(r'(?=[^\s：]+(?:题|卷|套卷|讲义|系列)[：:])', raw)
        entries = [e.strip() for e in entries if e.strip()]
        if len(entries) <= 1:
            entries = [raw]
        for e in entries:
            if len(e) < 20:
                continue
            chunks.append({
                "content": f"【书籍测评】{cat}\n{e}",
                "metadata": {"type": "书籍测评", "category": cat}
            })
    return chunks


# ============================================================
# 构建
# ============================================================
def build():
    print(f"📂 加载 Excel: {EXCEL_PATH}")
    wb = openpyxl.load_workbook(EXCEL_PATH)

    all_chunks = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        print(f"📄 解析: {sheet_name}")
        if sheet_name in PLAN_SHEETS:
            chunks = parse_plan_sheet(ws, PLAN_SHEETS[sheet_name])
        elif sheet_name == "名师测评":
            chunks = parse_teacher_review_sheet(ws)
        elif sheet_name == "书籍测评":
            chunks = parse_book_review_sheet(ws)
        else:
            continue
        all_chunks.extend(chunks)
        print(f"   → {len(chunks)} 条")

    print(f"\n📊 共 {len(all_chunks)} 条 chunk，构建 TF-IDF 索引...")

    documents = [c["content"] for c in all_chunks]
    vectorizer = TfidfVectorizer(
        tokenizer=jieba_tokenize,
        max_features=3000,
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(documents)

    # 保存
    kb_data = {
        "chunks": all_chunks,
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
    }
    with open(KB_PATH, "wb") as f:
        pickle.dump(kb_data, f)

    size_mb = Path(KB_PATH).stat().st_size / 1024 / 1024
    print(f"\n✅ 知识库已保存: {KB_PATH} ({size_mb:.1f} MB)")
    print(f"   词汇量: {len(vectorizer.vocabulary_)}")
    print(f"   矩阵形状: {tfidf_matrix.shape}")


if __name__ == "__main__":
    build()
