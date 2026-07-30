import sys
from pathlib import Path

# 保证 agent/、tools/、database/ 这些包能被绝对导入（app.py 在仓库根目录）
sys.path.append(str(Path(__file__).resolve().parent))

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from database.mock_db import DB_PATH, init_db, get_connection, query_db
from agent.graph_builder import hr_agent_app

st.set_page_config(page_title='飞羽科技 HR 智能助理', page_icon='🤖')

# 首次启动自动初始化演示数据库（云端每次重启文件系统可能是空的，所以要能自愈）
if not DB_PATH.exists():
    init_db()

st.title('🤖 飞羽科技 HR 智能助理 Demo')
st.caption('LangGraph 多节点编排 + RAG 检索 + Self-Reflection 事实审计 的企业 HR Agent 演示')

# 读取演示员工列表，供左侧选择"模拟登录"
conn = get_connection()
employees = query_db(conn, 'select uid, name, rank, location from employees')
conn.close()

emp_options = {
    f"{e['name']}（工号{e['uid']}，{e['rank']}，{e['location']}）": e['uid']
    for e in employees
}

with st.sidebar:
    st.header('演示设置')
    selected_label = st.selectbox('选择模拟登录的员工', list(emp_options.keys()))
    current_uid = emp_options[selected_label]

    if st.button('🗑️ 清空对话'):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown(
        '**可以试着问：**\n'
        '- 我还有多少年假？\n'
        '- 帮我开一份在职证明\n'
        '- 帮我开一份薪资收入证明\n'
        '- 出差去上海，住宿报销标准是多少？\n'
        '- 请假10天需要谁审批？'
    )
    st.divider()
    st.caption('⚠️ 这是一个技术能力演示 Demo，数据均为虚构测试数据，非真实公司信息。')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
    with st.chat_message(role):
        st.markdown(msg.content)

user_input = st.chat_input('请输入你的问题，例如：我还有多少年假？')

if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message('user'):
        st.markdown(user_input)

    with st.chat_message('assistant'):
        with st.spinner('正在思考...'):
            try:
                result = hr_agent_app.invoke({
                    'messages': st.session_state.messages,
                    'current_uid': current_uid,
                    'loop_state': 0,
                })
                final_messages = result['messages']

                ai_reply = None
                for m in reversed(final_messages):
                    if isinstance(m, AIMessage) and m.content:
                        ai_reply = m.content
                        break

                if ai_reply is None:
                    ai_reply = '抱歉，我没能生成有效回复，请换个问法再试一次。'
            except Exception as e:
                ai_reply = f'⚠️ 出现异常：{e}'

            st.markdown(ai_reply)
            st.session_state.messages.append(AIMessage(content=ai_reply))
