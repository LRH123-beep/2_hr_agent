from pathlib import Path

from langchain_core.tools import tool
# 导入Markdown标题切分器和递归字符切分器
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
# 导入HuggingFace嵌入模型和embeddings模块
from langchain_huggingface import HuggingFaceEmbeddings
# 导入Chroma向量数据库
from langchain_chroma import Chroma

import os
from dotenv import load_dotenv

load_dotenv('.env')

# 获取项目根目录（当前文件所在目录的父目录）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 定义公司手册markdown文件的路径
DOC_PATH = PROJECT_ROOT / 'data' / 'company_handbook.md'
# 定义向量数据库的存储路径
VECTOR_OIR = PROJECT_ROOT / 'db' / 'chroma.db'

# 1. 全局单列初始化 Embedding model (BGE)
print(f'正在加载 BGE 嵌入模型......')
model_name = PROJECT_ROOT / 'EMBEDDING_MODEL'
embeddings = HuggingFaceEmbeddings(
    model_name = os.getenv(model_name),
    # model_kwargs = {'device':'cpu'},
    encode_kwargs = {'normalize_embeddings': True},        # 如果是 英伟达显卡可以填写 cuda, 苹果M芯片填写 mps，其他填写 cpu 或这个参数都不写
)

def init_vector_store() -> Chroma:
    """初始化向量库，如果存在则读取，如果不存在则切分文档并生成"""
    if VECTOR_OIR.exists() and any(VECTOR_OIR.iterdir()):
        return Chroma(persist_directory=str(VECTOR_OIR),
                      embedding_function = embeddings)

    print('未检测到本地向量库，开始构建朴素 RAG 索引')

    if not DOC_PATH.exists():
        raise FileNotFoundError(f'找不到知识库文件：{DOC_PATH}')

    with open(DOC_PATH, 'r', encoding='utf-8') as f:
        markdown_text = f.read()

    # 基于 Markdown 层剂进行切分
    headers_to_split_on = [
        ('##', 'Chapter'),
        ('###', 'Section')
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_text)

    # 为了防止莫格章节依然过长，再叠加一个字符集滑动窗口切分
    chunk_size = 500
    chunk_overlap = 50
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(md_header_splits)

    print(f'文档切分完毕，共生成{len(splits)}个语义文本块(chunks)。正在存入数据库')

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding = embeddings,
        persist_directory=str(VECTOR_OIR),
    )

    print(f'向量数据库已构建完成，已落盘在:{VECTOR_OIR}')
    return vectorstore

vector_store = init_vector_store()                              #初始化全局向量库示例
retriever = vector_store.as_retriever(search_kwargs={'k':5})    #转换成检索器对象(retriever)，设置召回top-3的结果

# retriever = vector_store.as_retriever(          # 带阈值的做法， 只返回相似度 >= 0.5
#     search_type='similarity_score_threshold',
#     search_kwargs={
#         'score_threshold': 0.5,
#         'k':20                                  # 候选池大小（先取 Top 20, 再过滤相似度 >= 0.5）
#     }
# )

# 2. 封装成工具
@tool
def search_hr_policy(query:str) -> str:
    """
    搜索公司规章制度、差旅费报销标准、假期政策、福利等相关详细的必备工具。
    输入参数 query 必须是你从员工问题中提炼出来的精准检索词
    """
    docs = retriever.invoke(query)
    if not docs:
        return '知识库中未检索到相关政策，请提示用户询问 HR 人工'

    # 组装召回的上下文，附带 Matedata 让大模型知道出自那个章节，有效降低幻觉
    context_parts = []
    for i, doc in enumerate(docs, 1):
        chapter = doc.metadata.get('Chapter', '未知章节')
        section = doc.metadata.get('Section', '未知段落')
        context_parts.append(f'【来源{i}】 {chapter} -> {section} \n {doc.page_content}')

        merged_context = '\n\n'.join(context_parts)

        return f'【知识库检索结果】\n{merged_context}'