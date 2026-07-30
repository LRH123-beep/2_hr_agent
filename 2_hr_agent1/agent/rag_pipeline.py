from pathlib import Path

from langchain_core.tools import tool
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = PROJECT_ROOT / 'data' / 'company_handbook.md'
VECTOR_DIR = PROJECT_ROOT / 'db' / 'chroma.db'

# 【修改说明】原版本用本地磁盘上的模型文件夹（model_path 拼接 PROJECT_ROOT）。
# 这要求把几百MB的模型文件也传到 GitHub 上，既不现实也没必要。
# 改成直接用 HuggingFace Hub 上的模型名，部署平台第一次启动时会自动下载并缓存，
# 不需要把模型文件放进仓库。BAAI/bge-small-zh-v1.5 是一个较小的中文向量模型，适合 demo。
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-zh-v1.5')

print(f'正在加载嵌入模型：{EMBEDDING_MODEL_NAME} ......')
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    encode_kwargs={'normalize_embeddings': True},
)


def init_vector_store() -> Chroma:
    """初始化向量库，如果存在则读取，如果不存在则切分文档并生成"""
    if VECTOR_DIR.exists() and any(VECTOR_DIR.iterdir()):
        return Chroma(persist_directory=str(VECTOR_DIR), embedding_function=embeddings)

    print('未检测到本地向量库，开始构建朴素 RAG 索引')

    if not DOC_PATH.exists():
        raise FileNotFoundError(f'找不到知识库文件：{DOC_PATH}')

    with open(DOC_PATH, 'r', encoding='utf-8') as f:
        markdown_text = f.read()

    headers_to_split_on = [
        ('##', 'Chapter'),
        ('###', 'Section'),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_text)

    chunk_size = 500
    chunk_overlap = 50
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(md_header_splits)

    print(f'文档切分完毕，共生成 {len(splits)} 个语义文本块(chunks)。正在存入数据库')

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(VECTOR_DIR),
    )

    print(f'向量数据库已构建完成，已落盘在: {VECTOR_DIR}')
    return vectorstore


vector_store = init_vector_store()
retriever = vector_store.as_retriever(search_kwargs={'k': 5})


@tool
def search_hr_policy(query: str) -> str:
    """
    搜索公司规章制度、差旅费报销标准、假期政策、福利等相关详细的必备工具。
    输入参数 query 必须是你从员工问题中提炼出来的精准检索词
    """
    docs = retriever.invoke(query)
    if not docs:
        return '知识库中未检索到相关政策，请提示用户询问 HR 人工'

    context_parts = []
    for i, doc in enumerate(docs, 1):
        chapter = doc.metadata.get('Chapter', '未知章节')
        section = doc.metadata.get('Section', '未知段落')
        context_parts.append(f'【来源{i}】 {chapter} -> {section} \n {doc.page_content}')

    merged_context = '\n\n'.join(context_parts)

    return f'【知识库检索结果】\n{merged_context}'
