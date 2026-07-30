from typing import Annotated, TypedDict
from pydantic import BaseModel, Field

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, START, END

from tools.hr_tools import get_leave_balance, get_employee_profile, generate_employment_certificate
from agent.rag_pipeline import search_hr_policy

# 【修改说明1】原版本 tools 列表里少了 search_hr_policy —— 这是整个 RAG 系统的检索工具。
# 没有它，大模型永远不会真正调用知识库检索，fact_check_node 里
# "查找 name == 'search_hr_policy' 的消息" 这段逻辑也永远不会命中，等于白写了。
# 已在此处补上。
#
# 【修改说明2】原版本 tools 列表里包含了 get_connection —— 这是一个返回
# sqlite3.Connection 对象的函数，不是给大模型调用的工具（LLM 工具需要返回可序列化的文本/数据，
# 而不是数据库连接句柄）。已从工具列表中移除，get_connection 只在 hr_tools.py 内部使用。


# 1.定义全局共享状态
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_uid: str
    loop_state: int

# 2.初始化 LLM 与工具绑定
llm = ChatOpenAI(
    model=os.getenv('DEEPSEEK_MODEL'),
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url=os.getenv('DEEPSEEK_BASE_URL'),
    temperature=0.0
)

tools = [get_leave_balance, get_employee_profile, generate_employment_certificate, search_hr_policy]
llm_with_tools = llm.bind_tools(tools)

tools_node = ToolNode(tools)

# 3.定义执行节点
def chatbot_node(state: AgentState):
    """ [执行者节点] 意图理解、工具调用与内容生成 """
    messages = state.get('messages', [])

    if len(messages) == 1:
        system_msg = SystemMessage(
            content=f' 你是飞羽科技的高级 HR 智能助理。\n'
                    f'当前提问员工 UID 为 {state.get("current_uid")}\n'
                    f'请务必先调用 get_employee_profile 获取该员工的工作属性，再回答具体问题\n'
                    f'如涉及公司规章制度、报销标准、假期政策等问题，必须调用 search_hr_policy 检索知识库\n'
                    f'必须基于工具返回的事实，绝对不能编造数字或条件')
        messages = [system_msg] + messages

    response = llm_with_tools.invoke(messages)

    return {'messages': [response], 'loop_state': state.get('loop_state', 0) + 1}

class FactCheckResult(BaseModel):
    is_pass: bool = Field(description='如果AI回答完全忠于知识库原文输出True,捏造了数字或者政策则输出False')
    feedback: str = Field(description='如果False,指出造假点:如果True 输出‘PASS')

def fact_check_node(state: AgentState):
    """ [审计节点]后置事实检验 (Self-Reflection) """
    messages = state['messages']
    last_message = messages[-1]

    rag_context = ""
    for msg in reversed(messages):
        if getattr(msg, 'name', '') == 'search_hr_policy':
            rag_context = msg.content
            break

    if not rag_context:
        return {'messages': []}

    print('\n [审计者介入] 正在核查生成内容是否包含幻觉......')

    checker_llm = ChatOpenAI(
        model=os.getenv('DEEPSEEK_MODEL'),
        api_key=os.getenv('DEEPSEEK_API_KEY'),
        base_url=os.getenv('DEEPSEEK_BASE_URL'),
        temperature=0.0
    )

    parser = JsonOutputParser(pydantic_object=FactCheckResult)

    check_prompt = (
        f'你是一个冷酷的合规审计员，对比以下[知识库原文]和[AI生成的回复]。\n'
        f'[知识库原文]:\n{rag_context}\n'
        f'[AI生成的回复]：\n{last_message.content}\n'
        f'严查金额、职级门槛、天数！发现捏造请判 False 并给出修改意见\n\n'
        f'{parser.get_format_instructions()}'
    )

    response = checker_llm.invoke(check_prompt)

    try:
        result = parser.invoke(response)
        is_pass = result.get('is_pass', True)
        feedback = result.get('feedback', 'PASS')
    except Exception as e:
        print(f'[审计异常]JSON 解析失败，默认放行，原因:{e}')
        is_pass = True
        feedback = 'PASS'

    if is_pass:
        print('[审计通过]回答安全,无幻觉。')
        return {'messages': []}
    else:
        print(f'[发现幻觉]拦截生成！ 审计意见：{feedback}')
        correction_msg = HumanMessage(
            content=f'[SYSTEM AUDIT FAILED] 事实错误反馈:{feedback},根据知识库原文重写，绝不可包含虚假数据'
        )
        return {'messages': [correction_msg]}

# 4.定义路由逻辑
def router_after_chatbot(state: AgentState):
    """ Chatbot输出后的路由判断 """
    last_message = state['messages'][-1]

    if last_message.tool_calls:
        return 'tools'
    else:
        return 'fact_check'

def router_after_fact_check(state: AgentState):
    """ 审计完成后的路由判断 """
    last_message = state['messages'][-1]
    if isinstance(last_message, HumanMessage):
        if state.get('loop_state', 0) > 4:
            print('[强制熔断]反思次数达到上限，放弃纠错')
            return 'end'
        print('[打回重写]图路由指针倒流回 chatbot 节点....')
        return 'chatbot'
    return 'end'

# 5.构建状态图
workflow = StateGraph(AgentState)

workflow.add_node('chatbot', chatbot_node)
workflow.add_node('fact_check', fact_check_node)
workflow.add_node('tools', tools_node)

workflow.add_edge(START, 'chatbot')
workflow.add_conditional_edges('chatbot', router_after_chatbot, {
    'tools': 'tools',
    'fact_check': 'fact_check',
})
workflow.add_edge('tools', 'chatbot')
workflow.add_conditional_edges('fact_check', router_after_fact_check, {
    'chatbot': 'chatbot',
    'end': END
})
hr_agent_app = workflow.compile()
