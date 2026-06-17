"""
RAG 引擎：检索 + 门控判定 + LLM 生成 + 答案缓存
"""
import os
import hashlib
import json
import sqlite3
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ============================================================
# 配置
# ============================================================
CHROMA_DIR = "./chroma_data"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
COLLECTION_NAME = "kaoyan_math"
CACHE_DB = "./cache.db"

# DeepSeek API 配置
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-chat"

# 检索参数
TOP_K = 8  # 检索返回条数

# 门控阈值
MIN_HITS = 2       # 最少命中条数
MIN_AVG_SCORE = 0.45  # 最低平均相似度（bge 在领域文本上 0.4~0.7 为正常范围）
# 注意：Chroma 默认用余弦距离（越小越相似），我们会转成相似度

# ============================================================
# 答案缓存（SQLite）
# ============================================================
class AnswerCache:
    """基于 SQLite 的问答缓存，持久化到磁盘"""

    def __init__(self, db_path: str = CACHE_DB, ttl_days: int = 30):
        self.db_path = db_path
        self.ttl_seconds = ttl_days * 86400
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    question_hash TEXT PRIMARY KEY,
                    question TEXT,
                    answer TEXT,
                    source_type TEXT,
                    created_at REAL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created 
                ON cache(created_at)
            """)

    def _normalize(self, question: str) -> str:
        """标准化问题字符串"""
        q = question.strip().lower()
        # 去掉多余空白
        q = " ".join(q.split())
        return q

    def _hash(self, question: str) -> str:
        return hashlib.md5(self._normalize(question).encode()).hexdigest()

    def get(self, question: str) -> dict | None:
        h = self._hash(question)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT answer, source_type, created_at FROM cache WHERE question_hash = ?",
                (h,)
            ).fetchone()

        if row is None:
            return None

        answer, source_type, created_at = row
        # 检查是否过期
        if time.time() - created_at > self.ttl_seconds:
            self._delete(h)
            return None

        return {"answer": answer, "source_type": source_type}

    def set(self, question: str, answer: str, source_type: str):
        h = self._hash(question)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?, ?)",
                (h, self._normalize(question), answer, source_type, time.time())
            )

    def _delete(self, h: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache WHERE question_hash = ?", (h,))

    def stats(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            # 清理过期
            cutoff = time.time() - self.ttl_seconds
            conn.execute("DELETE FROM cache WHERE created_at < ?", (cutoff,))
            valid = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        return {"total": total, "valid": valid}


# ============================================================
# RAG 引擎
# ============================================================
class RAGEngine:
    def __init__(self):
        print("🤖 加载 Embedding 模型...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)

        print("📚 连接 Chroma 知识库...")
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.chroma_client.get_collection(COLLECTION_NAME)

        print("💾 初始化缓存...")
        self.cache = AnswerCache()

        print("🔗 初始化 DeepSeek 客户端...")
        self.llm_client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )

        print(f"✅ RAG 引擎就绪（知识库: {self.collection.count()} 条）")

    # ---- 检索 ----
    def search(self, question: str, top_k: int = TOP_K) -> list[dict]:
        """语义检索，返回带相似度的结果列表"""
        q_embedding = self.embed_model.encode([question]).tolist()

        results = self.collection.query(
            query_embeddings=q_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]  # 余弦距离，越小越相似
                similarity = 1.0 - distance  # 转为相似度
                hits.append({
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity": round(similarity, 4),
                })

        # 按相似度降序
        hits.sort(key=lambda x: x["similarity"], reverse=True)
        return hits

    # ---- 门控判定 ----
    def decide_source(self, hits: list[dict]) -> tuple[str, bool]:
        """
        判定答案来源。
        返回 (source_type, is_kb)
          - "knowledge_base" → 走知识库
          - "ai_generated"    → AI 自主生成
        """
        if len(hits) < MIN_HITS:
            return "ai_generated", False

        # 取前 MIN_HITS 条的平均相似度
        avg_score = sum(h["similarity"] for h in hits[:MIN_HITS]) / MIN_HITS

        if avg_score >= MIN_AVG_SCORE:
            return "knowledge_base", True
        else:
            return "ai_generated", False

    # ---- 构建 Prompt ----
    def build_prompt_kb(self, question: str, hits: list[dict]) -> str:
        """知识库模式 prompt"""
        context_parts = []
        for i, hit in enumerate(hits[:TOP_K], 1):
            meta = hit["metadata"]
            source_info = f"{meta.get('teacher', '')} | {meta.get('stage', '')} | {meta.get('subject', '')}"
            context_parts.append(f"【来源 {i}】{source_info}\n{hit['content']}")

        context = "\n\n---\n\n".join(context_parts)

        return f"""你是一个专业的考研数学备考规划助手。请严格根据以下从全年规划表中检索到的信息回答用户问题。

【检索到的规划内容】
{context}

【用户问题】
{question}

要求：
1. 只基于上面提供的内容回答，不要编造规划表中不存在的信息
2. 如果检索内容涉及多个阶段/老师，按时间顺序或逻辑关系组织回答
3. 回答简洁实用，直接给可执行的建议
4. 如果涉及不同目标分数（130+/110-130/110-），请分别说明"""

    def build_prompt_ai(self, question: str, hits: list[dict]) -> str:
        """AI 自主模式 prompt（检索不足时）"""
        # 即使检索不足，也把少量检索结果给出来当参考
        ref_text = ""
        if hits:
            ref_parts = []
            for h in hits[:3]:
                ref_parts.append(h["content"][:300])
            ref_text = "\n\n".join(ref_parts)
            ref_text = f"\n\n【仅供参考的弱相关条目】\n{ref_text}\n"

        return f"""你是一个专业的考研数学备考规划助手。用户的问题在现有规划表中没有找到足够匹配的内容，请根据你对考研数学的通用知识给出合理建议。{ref_text}

【用户问题】
{question}

要求：
1. 给出具体的、可执行的建议（时间安排、资料推荐、复习方法等）
2. 说明建议的依据
3. 区分不同目标分数（130+/110-130/110-）给出差异化建议
4. 建议用户将此问题与自己的实际情况结合判断"""

    # ---- LLM 调用 ----
    def call_llm(self, prompt: str) -> str:
        """调用 DeepSeek API"""
        try:
            response = self.llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个专业的考研数学备考规划助手，帮助考生制定科学的全年复习计划。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ LLM 调用失败: {str(e)}\n\n请检查 API Key 是否正确配置。"

    # ---- 主入口 ----
    def ask(self, question: str) -> dict:
        """
        主问答接口。
        返回 {
            "answer": str,       # 最终答案
            "source_type": str,  # "knowledge_base" 或 "ai_generated"
            "hits": list[dict],  # 检索结果（供展示用）
        }
        """
        # 1. 查缓存
        cached = self.cache.get(question)
        if cached:
            print(f"  💾 命中缓存")
            return {
                "answer": cached["answer"],
                "source_type": cached["source_type"],
                "hits": [],
                "from_cache": True,
            }

        # 2. 检索
        hits = self.search(question)

        # 3. 门控判定
        source_type, is_kb = self.decide_source(hits)

        # 4. 构建 prompt 并生成
        if is_kb:
            prompt = self.build_prompt_kb(question, hits)
        else:
            prompt = self.build_prompt_ai(question, hits)

        answer = self.call_llm(prompt)

        # 5. 写入缓存
        self.cache.set(question, answer, source_type)

        return {
            "answer": answer,
            "source_type": source_type,
            "hits": hits,
            "from_cache": False,
        }


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    engine = RAGEngine()
    
    test_questions = [
        "我基础很差，线代应该跟哪个老师？",
        "7月份高数强化阶段应该怎么做？",
        "张宇1000题和880题哪个更好？",
    ]
    
    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"❓ {q}")
        result = engine.ask(q)
        print(f"📌 来源: {result['source_type']} | 缓存: {result.get('from_cache', False)}")
        print(f"📊 命中: {len(result['hits'])} 条")
        if result['hits']:
            print(f"   Top3 相似度: {[h['similarity'] for h in result['hits'][:3]]}")
        print(f"💬 答案:\n{result['answer'][:500]}...")
