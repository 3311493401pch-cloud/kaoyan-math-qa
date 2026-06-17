"""
考研数学 Q&A — 极简 Gradio 前端 + 知识库热更新
"""
import os
import gradio as gr
from rag_engine import RAGEngine

# ============================================================
# 全局引擎
# ============================================================
engine: RAGEngine = None


def init_engine():
    global engine
    engine = RAGEngine()
    return engine


def get_engine():
    global engine
    if engine is None:
        engine = init_engine()
    return engine


# ============================================================
# 对话处理
# ============================================================
def chat(message: str, history: list):
    """处理一轮对话"""
    if not message or not message.strip():
        yield ""
        return

    e = get_engine()
    result = e.ask(message.strip())

    answer = result["answer"]
    source_type = result["source_type"]
    hits = result["hits"]
    from_cache = result.get("from_cache", False)

    # AI 自主回答标注
    if source_type == "ai_generated":
        answer += "\n\n---\n🤖 *此回答由 AI 基于通用知识生成，规划表中未找到充分匹配内容*"

    # 参考来源
    if hits and not from_cache:
        seen = set()
        lines = []
        for h in hits[:5]:
            m = h["metadata"]
            k = (m.get("teacher", ""), m.get("stage", ""), m.get("subject", ""))
            if k not in seen:
                seen.add(k)
                sim = h["similarity"]
                lines.append(
                    f"> {m.get('teacher','')} · {m.get('stage','')} · "
                    f"{m.get('subject','')}（匹配 {sim:.0%}）"
                )
        if lines:
            answer += "\n\n**📋 参考来源**\n" + "\n".join(lines)

    if from_cache:
        answer += "\n\n---\n💾 命中缓存，零费用"

    yield answer


# ============================================================
# 知识库更新
# ============================================================
def rebuild_kb(excel_file):
    """上传新 Excel 后重建知识库"""
    if excel_file is None:
        return "⚠️ 请先上传 Excel 文件"

    try:
        import shutil
        # 备份原文件
        backup = "亚瑟爱数学考研数学规划.xlsx.bak"
        if os.path.exists("亚瑟爱数学考研数学规划.xlsx"):
            shutil.copy("亚瑟爱数学考研数学规划.xlsx", backup)

        # 覆盖
        shutil.copy(excel_file.name, "亚瑟爱数学考研数学规划.xlsx")

        # 重建
        import subprocess
        result = subprocess.run(
            ["python", "kb_builder.py"],
            capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            # 恢复备份
            if os.path.exists(backup):
                shutil.copy(backup, "亚瑟爱数学考研数学规划.xlsx")
            return f"❌ 构建失败:\n{result.stderr[-300:]}"

        # 重载引擎
        global engine
        engine = RAGEngine()

        # 清理旧备份
        if os.path.exists(backup):
            os.remove(backup)

        lines = [l for l in result.stdout.split("\n") if l.strip()]
        summary = "\n".join(lines[-5:])
        return f"✅ 知识库已更新！\n\n{summary}"

    except Exception as ex:
        return f"❌ 更新失败: {str(ex)}"


# ============================================================
# 界面
# ============================================================
CUSTOM_CSS = """
/* 全局字体 & 背景 */
* {
    font-family: "PingFang SC", "Microsoft YaHei", "Noto Sans SC", system-ui, sans-serif !important;
}
body, .gradio-container {
    background: #f8f9fa !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* 主容器居中紧凑 */
.gradio-container {
    max-width: 720px !important;
    margin: 0 auto !important;
    padding: 12px 16px !important;
}

/* 标题区 */
.app-header {
    text-align: center;
    padding: 18px 0 6px 0;
    border-bottom: 1px solid #e9ecef;
    margin-bottom: 10px;
}
.app-header h1 {
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    color: #1a1a2e !important;
    margin: 0 0 4px 0 !important;
    letter-spacing: 0.02em;
}
.app-header p {
    font-size: 0.8rem !important;
    color: #6c757d !important;
    margin: 0 !important;
}

/* 聊天框 */
.chatbot {
    border-radius: 12px !important;
    border: 1px solid #e9ecef !important;
    background: #fff !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}
.chatbot .message {
    font-size: 0.92rem !important;
    line-height: 1.65 !important;
}
.chatbot .message.bot {
    background: #fff !important;
    color: #212529 !important;
}
.chatbot .message.user {
    background: #e8f0fe !important;
    color: #1a1a2e !important;
}

/* 输入框 */
.input-row {
    margin-top: 6px !important;
}
.input-row textarea {
    border-radius: 10px !important;
    border: 1px solid #dee2e6 !important;
    font-size: 0.92rem !important;
    padding: 10px 14px !important;
    box-shadow: none !important;
}
.input-row textarea:focus {
    border-color: #4a6cf7 !important;
    box-shadow: 0 0 0 3px rgba(74,108,247,0.1) !important;
}

/* 按钮 */
button.primary {
    background: #4a6cf7 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 8px 20px !important;
    border: none !important;
}
button.primary:hover {
    background: #3b5de7 !important;
}

/* 折叠面板（知识库管理） */
.accordion {
    border: 1px solid #e9ecef !important;
    border-radius: 8px !important;
    margin-top: 10px !important;
}
.accordion > .label-wrap {
    font-size: 0.82rem !important;
    color: #6c757d !important;
}

/* 隐藏 Gradio 默认 footer */
footer { display: none !important; }

/* 响应式 */
@media (max-width: 600px) {
    .gradio-container { padding: 8px 10px !important; }
    .app-header h1 { font-size: 1.15rem !important; }
}
"""

HEADER_HTML = """
<div class="app-header">
    <h1>🎓 考研数学 · 全年规划问答</h1>
    <p>基于亚瑟爱数学规划表 · 张宇 / 武忠祥 · 随时更新知识库</p>
</div>
"""


def create_demo():
    with gr.Blocks(
        css=CUSTOM_CSS,
        title="考研数学 Q&A",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
    ) as demo:
        gr.HTML(HEADER_HTML)

        gr.ChatInterface(
            fn=chat,
            chatbot=gr.Chatbot(height=520, show_copy_button=True, bubble_full_width=False),
            textbox=gr.Textbox(
                placeholder="输入你的考研数学问题…",
                scale=4,
                container=False,
            ),
            submit_btn="发送",
            clear_btn="🗑️",
        )

        # 知识库管理（折叠）
        with gr.Accordion("📂 知识库管理（上传新规划表更新）", open=False):
            with gr.Row():
                file_input = gr.File(
                    label="上传新的 Excel 规划表（.xlsx）",
                    file_types=[".xlsx"],
                    scale=3,
                )
                upload_btn = gr.Button("🔄 更新知识库", variant="secondary", scale=1)
            rebuild_status = gr.Markdown("")

        upload_btn.click(
            fn=rebuild_kb,
            inputs=[file_input],
            outputs=[rebuild_status],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
    )
