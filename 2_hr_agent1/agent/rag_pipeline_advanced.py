"""
【进阶版 RAG，默认不启用】

这是你原来的 rag_pipeline2.py，包含：
- BM25 + 向量检索的混合召回
- 手动 RRF (Reciprocal Rank Fusion) 融合排序
- Query 多角度扩写 + HyDE 假设文档生成
- Cross-Encoder 重排序 (bge-reranker-base)

技术含量比默认版本(rag_pipeline.py)更高，适合在简历/面试里讲述你对 RAG 优化的理解，
但依赖更重（sentence-transformers + 本地 reranker 模型），
部署到免费云平台上可能会更慢、更占内存。

如果想在 demo 里启用这个版本：
1. requirements.txt 里加上 langchain-community
2. RERANK_MODEL 换成 HuggingFace Hub 上的模型名（例如 BAAI/bge-reranker-base），
   而不是本地路径，原理和 rag_pipeline.py 里 embedding 模型的修改一致
3. graph_builder.py 里把 `from agent.rag_pipeline import search_hr_policy`
   改成 `from agent.rag_pipeline_advanced import search_hr_policy`
"""
from pathlib import Path

from langchain_core.tools import tool
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = PROJECT_ROOT / 'data' / 'company_handbook.md'
VECTOR_DIR = PROJECT_ROOT / 'db' / 'chroma.db'

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL')
DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL')

EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-zh-v1.5')
RERANK_MODEL_NAME = os.getenv('RERANK_MODEL', 'BAAI/bge-reranker-base')

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    encode_kwargs={'normalize_embeddings': True},
)

reranker = CrossEncoder(RERANK_MODEL_NAME, max_length=512, device='cpu')

llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.7,
)


def build_ensemble_retriever():
    """构建 BM25 + Vector 的混合检索器"""
    if not DOC_PATH.exists():
        raise FileNotFoundError(f'找不到知识库文件：{DOC_PATH}')

    with open(DOC_PATH, 'r', encoding='utf-8') as f:
        markdown_text = f.read()

    headers_to_split_on = [('##', 'Chapter'), ('###', 'Section')]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=['\n\n', '\n']
    )
    splits = text_splitter.split_documents(md_header_splits)

    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 5

    if VECTOR_DIR.exists() and any(VECTOR_DIR.iterdir()):
        vectorstore = Chroma(persist_directory=str(VECTOR_DIR), embedding_function=embeddings)
    else:
        vectorstore = Chroma.from_documents(
            documents=splits, embedding=embeddings, persist_directory=str(VECTOR_DIR)
        )
    vector_retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

    return bm25_retriever, vector_retriever


retriever = build_ensemble_retriever()


class QueryExpansion(BaseModel):
    expanded_queries: list[str] = Field(description='从不同维度扩写 3 个相关检索词或短语')
    hypothetical_document: str = Field(description='针对该问题的一段假设性、看似专业的官方制度回答片段')


expansion_parser = JsonOutputParser(pydantic_object=QueryExpansion)


def expand_and_hyde(original_query: str) -> list[str]:
    prompt = ChatPromptTemplate.from_template(
        "你是一名专业的企业 HR 专家。为了提高知识库检索命中率，请协助处理用户的原始提问。\n"
        "任务1（多维扩写）：站在不同视角扩写 3 个相关检索词或短语。\n"
        "任务2（HyDE假设）：用官方、严谨的 HR 规章制度口吻，写一段回答该问题的假设性文本。\n\n"
        "用户原始问题：{query}\n{format_instructions}"
    )
    chain = prompt | llm | expansion_parser
    try:
        result = chain.invoke({
            'query': original_query,
            'format_instructions': expansion_parser.get_format_instructions(),
        })
        return [original_query] + result['expanded_queries'] + [result['hypothetical_document']]
    except Exception as e:
        print(f'LLM 调用失败，降级使用基础检索。原因：{e}')
        return [original_query]


@tool
def search_hr_policy(query: str) -> str:
    """
    高级知识搜索引擎（自动改写 + 混合检索 + 重排）。
    当用户查询公司规章制度、差旅报销标准、假期政策、福利等相关信息时调用。
    """
    search_queries = expand_and_hyde(query)

    all_docs = []
    bm25_r, vector_r = retriever
    for q in search_queries:
        all_docs.extend(bm25_r.invoke(q))
        all_docs.extend(vector_r.invoke(q))

    unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
    if not unique_docs:
        return '知识库中未检索到相关政策，请提示用户询问 HR 人工'

    pairs = [[query, doc.page_content] for doc in unique_docs]
    scores = reranker.predict(pairs)
    scored = sorted(zip(unique_docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in scored[:3]]

    context_parts = []
    for i, doc in enumerate(top_docs, 1):
        chapter = doc.metadata.get('Chapter', '未知章节')
        section = doc.metadata.get('Section', '未知段落')
        context_parts.append(f'来源[{i}] {chapter} -> {section}\n{doc.page_content}')

    return f'【知识库检索结果】\n' + '\n\n'.join(context_parts)
