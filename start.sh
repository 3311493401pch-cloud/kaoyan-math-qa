#!/bin/bash
# Render 启动脚本：先构建知识库，再启动 Gradio

echo "📂 构建知识库..."
python kb_builder.py

echo "🚀 启动 Gradio 服务..."
python app.py