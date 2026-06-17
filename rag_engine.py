"""
轻量 RAG 引擎（适配 Render 512MB）
TF-IDF 检索 + DeepSeek LLM + SQLite 缓存
"""
import os
import pickle
import hashlib
import sqlite3
import time

from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# ============================================================
# 配置
# ============================================================
KB_PATH = "knowledge_base.pkl"
CACHE_DB = "./cache.db"

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-chat"

TOP_K = 8
MIN_HITS = 2
MIN_AVG_SCORE = 0.10  # TF-IDF 余弦相似度偏低，阈值相应调低


# ============================================================
# 缓存
# ============================================================
class AnswerCache:
    def __init__(self, db_path: str = CACHE_DB, ttl_days: int = 30):
        self.db_path = db_path
        self.ttl_seconds = ttl_days * 86400
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    question_hash TEXT PRIMARY KEY,
                    question TEXT, answer TEXT,
                    source_type TEXT, created_at REAL
                )
            """)

    def _hash(self, question: str) -> str:
        q = " ".join(question.strip().lower().split())
        return hashlib.md5(q.encode()).hexdigest()

    def get(self, question: str) -> dict | None:
        h = self._hash(question)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT answer, source_type, created_at FROM cache WHERE question_hash=?", (h,)
            ).fetchone()
        if row is None:
            return None
        answer, st, ct = row
        if time.time() - ct > self.ttl_seconds:
            return None
        return {"answer": answer, "source_type": st}

    def set(self, question: str, answer: str, source_type: str):
        h = self._hash(question)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?)",
                (h, " ".join(question.strip().lower().split()), answer, source_type, time.time())
            )

    def stats(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            valid = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        return {"valid": valid}


# ============================================================
# 引擎
# ============================================================
class RAGEngine:
    def __init__(self):
        print("📚 加载知识库...")
        with open(KB_PATH, "rb") as f:
            kb = pickle.load(f)
        self.chunks = kb["chunks"]
        self.vectorizer = kb["vectorizer"]
        self.tfidf_matrix = kb["tfidf_matrix"]
        print(f"   {len(self.chunks)} 条 chunk, {len(self.vectorizer.vocabulary_)} 个词")

        print("💾 初始化缓存...")
        self.cache = AnswerCache()

        print("🔗 初始化 DeepSeek...")
        self.llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        print("✅ 引擎就绪")

    def search(self, question: str, top_k: int = TOP_K) -> list[dict]:
        """TF-IDF 检索"""
        q_vec = self.vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]

        # 取 top_k
        top_indices = sims.argsort()[::-1][:top_k]
        hits = []
        for idx in top_indices:
            score = float(sims[idx])
            if score < 0.01:  # 几乎无关，跳过
                continue
            hits.append({
                "content": self.chunks[idx]["content"],
                "metadata": self.chunks[idx]["metadata"],
                "similarity": round(score, 4),
            })
        return hits

    def decide_source(self, hits: list[dict]) -> tuple[str, bool]:
        if len(hits) < MIN_HITS:
            return "ai_generated", False
        avg = sum(h["similarity"] for h in hits[:MIN_HITS]) / MIN_HITS
        return ("knowledge_base", True) if avg >= MIN_AVG_SCORE else ("ai_generated", False)

    def build_prompt_kb(self, question: str, hits: list[dict]) -> str:
        parts = []
        for i, h in enumerate(hits[:TOP_K], 1):
            m = h["metadata"]
            src = f"{m.get('teacher','')} | {m.get('stage','')} | {m.get('subject','')}"
            parts.append(f"【来源 {i}】{src}\n{h['content']}")
        context = "\n\n---\n\n".join(parts)
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
        ref = ""
        if hits:
            ref = "\n\n".join(h["content"][:300] for h in hits[:3])
            ref = f"\n\n【仅供参考的弱相关条目】\n{ref}\n"
        return f"""你是一个专业的考研数学备考规划助手。用户的问题在现有规划表中没有找到足够匹配的内容，请根据你对考研数学的通用知识给出合理建议。{ref}

【用户问题】
{question}

要求：
1. 给出具体的、可执行的建议（时间安排、资料推荐、复习方法等）
2. 说明建议的依据
3. 区分不同目标分数给出差异化建议"""

    def call_llm(self, prompt: str) -> str:
        try:
            resp = self.llm.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是考研数学备考规划助手，帮助考生制定科学的复习计划。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1500,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ LLM 调用失败: {e}"

    def ask(self, question: str) -> dict:
        # 缓存
        cached = self.cache.get(question)
        if cached:
            return {"answer": cached["answer"], "source_type": cached["source_type"],
                    "hits": [], "from_cache": True}

        hits = self.search(question)
        source_type, is_kb = self.decide_source(hits)
        prompt = self.build_prompt_kb(question, hits) if is_kb else self.build_prompt_ai(question, hits)
        answer = self.call_llm(prompt)
        self.cache.set(question, answer, source_type)

        return {"answer": answer, "source_type": source_type,
                "hits": hits, "from_cache": False}
