# 飞羽科技 HR 智能助理（RAG + LangGraph Demo）

一个基于 **LangGraph 多节点编排 + RAG 知识库检索 + Self-Reflection 事实审计** 的企业 HR Agent 演示项目。

## 功能

- 查询员工假期余额、员工档案
- 自助开具《在职证明》《薪资收入证明》（带职级权限校验）
- 基于公司规章制度手册（Markdown）做 RAG 检索，回答假期政策、差旅报销标准等问题
- **审计节点**：每次生成回答后，会对照知识库原文自动核查是否有编造数字/政策，如果发现幻觉会自动打回重写

## 项目结构

```
2_hr_agent/
├── agent/
│   ├── graph_builder.py   # LangGraph 状态图：chatbot / tools / fact_check 三节点编排
│   └── rag_pipeline.py    # RAG 检索工具（Markdown 切分 + Chroma 向量库）
├── tools/
│   └── hr_tools.py        # 员工档案 / 假期余额 / 开证明 三个工具函数
├── database/
│   └── mock_db.py         # 演示用 SQLite 数据库（自动初始化，无需手动跑脚本）
├── data/
│   └── company_handbook.md  # 知识库原文（公司规章制度）
├── app.py                 # Streamlit 网页入口
├── requirements.txt
├── .env.example
└── .gitignore
```

## 相比你原始代码做了哪些修改（重要，面试也可以讲）

1. **补上了遗漏的检索工具**：原 `graph_builder.py` 的工具列表里没有加入 `search_hr_policy`，导致大模型永远不会真正调用知识库检索，`fact_check_node` 里查找该工具调用记录的逻辑也永远不会命中。已修复。
2. **移除了不合理的工具绑定**：原工具列表里包含 `get_connection`（返回数据库连接对象），这不是一个能给大模型调用的工具（LLM 工具需要返回可读文本，而不是连接句柄）。已移除，`get_connection` 只作为内部辅助函数使用。
3. **Embedding 模型改为 HuggingFace Hub 远程加载**：原版本从本地磁盘路径加载模型文件，要求把几百 MB 的模型文件也提交到 GitHub，不现实。改为直接用模型名 `BAAI/bge-small-zh-v1.5`，云端第一次启动时自动下载。
4. **数据库自动初始化**：原版本数据库不存在时会直接报错退出，要求手动运行初始化脚本。云端部署没有人会帮你手动跑命令，已改成自动初始化演示数据。
5. **新增了 `app.py`**：这是唯一新增的功能文件，用 Streamlit 包了一层聊天网页界面，让"部署后生成的链接"能被任何人直接打开使用。
6. **重建了 `tools/hr_tools.py`**：这个文件当时没有提供给我，我根据 `graph_builder.py` 的引用和数据库表结构重新实现了一版。**如果你本地已有自己的实现，直接用你自己的文件覆盖即可**。

## 本地运行

```bash
pip install -r requirements.txt
cp .env.example .env   # 然后把 .env 里的 DEEPSEEK_API_KEY 换成你的真实 key
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`。

## 部署到公网（推荐：Streamlit Community Cloud，免费）

1. 把这个项目推到你的 **Public** GitHub 仓库（`.env` 不要传上去，`.gitignore` 已经排除了）
2. 打开 [share.streamlit.io](https://share.streamlit.io)，用 GitHub 账号登录
3. 点击 **New app**，选择你的仓库、分支 `main`、主文件填 `app.py`
4. 点击 **Advanced settings → Secrets**，填入：
   ```
   DEEPSEEK_API_KEY = "sk-你的真实key"
   DEEPSEEK_BASE_URL = "https://api.deepseek.com"
   DEEPSEEK_MODEL = "deepseek-chat"
   EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
   ```
5. 点击 **Deploy**，等待 1-3 分钟（首次启动要下载嵌入模型，会慢一点）
6. 拿到形如 `https://xxx.streamlit.app` 的链接，任何人打开都能直接使用

## 注意事项

- 知识库文档（`data/company_handbook.md`）和数据库数据（`database/mock_db.py`）**都是虚构的演示数据**，不涉及任何真实公司信息，可以放心放到 Public 仓库
- 云端平台的免费额度通常有请求频率限制，长时间没人访问会自动休眠，属正常现象
- 如果想接入你自己更完整的 `hr_tools.py`，直接替换 `tools/hr_tools.py` 即可，函数名和 `@tool` 装饰器保持一致就行
