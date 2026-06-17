"""
考研数学 Q&A — Gradio 前端
"""
import gradio as gr
from rag_engine import RAGEngine

# ============================================================
# 初始化引擎（全局单例）
# ============================================================
engine = RAGEngine()

# ============================================================
# 对话处理
# ============================================================
def chat(message: str, history: list[list[str]]):
    """处理一轮对话"""
    if not message.strip():
        yield "请输入你的考研数学问题～"
        return

    # 调用 RAG 引擎
    result = engine.ask(message)

    answer = result["answer"]
    source_type = result["source_type"]
    hits = result["hits"]
    from_cache = result.get("from_cache", False)

    # 构建展示用的答案
    # 追加来源标注
    if source_type == "ai_generated":
        answer += "\n\n---\n🤖 **注意**：以上建议由 AI 基于通用知识生成，未在全年规划表中找到充分匹配的内容，请结合实际情况判断。"

    # 追加检索来源（可折叠提示）
    if hits and not from_cache:
        answer += "\n\n---\n📋 **参考来源**（来自规划表）：\n"
        seen = set()
        for h in hits[:5]:
            meta = h["metadata"]
            key = f"{meta.get('teacher','')}|{meta.get('stage','')}|{meta.get('subject','')}"
            if key not in seen:
                seen.add(key)
                sim = h["similarity"]
                answer += f"- {meta.get('teacher','')} · {meta.get('stage','')} · {meta.get('subject','')}（匹配度 {sim:.0%}）\n"

    if from_cache:
        answer += "\n\n---\n💾 该答案来自缓存（重复问题，零费用）"

    # 缓存统计
    stats = engine.cache.stats()
    answer += f"\n\n---\n📊 缓存命中: {stats['valid']} 条（帮省了 {stats['valid']} 次 API 调用）"

    yield answer


# ============================================================
# Gradio 界面
# ============================================================
TITLE = "🎓 考研数学全年规划 Q&A"
DESCRIPTION = """
基于「亚瑟爱数学」全年规划表（张宇 / 武忠祥 A1 / 武忠祥 A2）构建的智能问答系统。

**支持的问题类型：**
- 📅 各阶段该怎么复习？（基础 / 强化 / 真题 / 模拟卷）
- 👨‍🏫 线代/概率论该跟哪个老师？
- 📚 某本习题册/模拟卷怎么样？
- 🎯 不同目标分数（130+ / 110-130 / 110-）分别怎么安排？

**数据来源：** 6 张 Excel Sheet，涵盖全年规划 + 名师测评 + 书籍测评
"""

EXAMPLES = [
    "我基础比较差，线代应该跟哪个老师？",
    "7月份高数强化阶段应该怎么做？每天花多少时间？",
    "张宇1000题和880题哪个更好？怎么选？",
    "目标130分，概率论应该怎么复习？",
    "什么时候开始做真题？怎么做效率最高？",
    "武忠祥和张宇的规划有什么区别？",
]

css = """
footer {visibility: hidden}
.gradio-container {max-width: 800px !important}
"""

with gr.Blocks(css=css, title="考研数学 Q&A") as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)

    chatbot = gr.ChatInterface(
        fn=chat,
        title="",
        description="",
        examples=EXAMPLES,
        theme="soft",
        submit_btn="发送",
        retry_btn="重新生成",
        undo_btn="撤销",
        clear_btn="清空对话",
    )

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
    )
