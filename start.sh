#!/bin/bash
echo "📂 构建知识库..."
python kb_builder.py
echo "🚀 启动考研数学 Q&A..."
python app.py