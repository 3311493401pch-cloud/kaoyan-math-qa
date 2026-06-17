"""
共享工具函数
"""
import jieba


def jieba_tokenize(text):
    """jieba 分词，供 TfidfVectorizer 使用"""
    return list(jieba.cut(text))
