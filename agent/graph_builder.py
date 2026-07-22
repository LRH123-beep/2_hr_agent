from typing import Annotated,TypedDict
from pydantic import BaseModel,Field

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,SystemMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph,START,END

from tools.hr_tools import get_leave_balance,get_employee_profile,get_connection,generate_employment_certificate


# 1.定义全局共享状态
class AgentState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]
    current_uid:str
    loop_state:int

# 2.初始化 LLM　与工具绑定
llm = ChatOpenAI(
    model=os.getenv('DEEPSEEK_MODEL'),
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url=os.getenv('DEEPSEEK_BASE_URL'),
    temperature=0.0
)

tools = [get_leave_balance,get_employee_profile,get_connection,generate_employment_certificate]
llm_with_tools = llm.bind_tools(tools)   ################ 给大模型工具的权力 这里没有agent是手动写循环

tools_node = ToolNode(tools)             # 真正的使用工具

# 3.定义执行节点
def chatbot_node(state:AgentState):
    """ [执行者节点] 意图理解、工具调用与内容生成 """
    messages = state.get('messages',[])

    # 首轮对话注入 System Prompt
    if len(messages) == 1:
        system_msg = SystemMessage(
            content=f' 你是飞羽科技的高级 HR　智能助理。\n'
                    f'当前提问员工 UID 为 {state.get("current_uid")}\n'
                    f'请务必先调用 get_employee_profiles 获取该员工的工作属性，再回答具体问题\n'
                    f'必须基于工具返回的事实，绝对不能编造数字或条件')
        messages = [system_msg] + messages

    response = llm_with_tools.invoke(messages)

    return {'messages':[response],'loop_state':state.get('loop_state',0)+1} # 返回的内容加一防止死循环

class FactCheckResult(BaseModel):  # 后面会控制大模型的行为要是不知道就不要瞎说
    is_pass: bool = Field(description='如果AI回答完全忠于知识库原文输出True,捏造了数字或者政策则输出False')     # 描述
    feedback:str = Field(description='如果False,指出造假点:如果True 输出‘PASS')

def fact_check_node(state:AgentState):
    """ [审计节点]后置事实检验 (Self-Reflection) """
    messages = state['messages']
    last_message = messages[-1]

    # 逆向查找 RAG　召回的原文
    rag_context = ""
    for msg in reversed(messages):
        if getattr(msg,'name','') == 'search_hr_policy':
            rag_context = msg.content
            break

    # 不用rag 若未调用知识库,直接放行
    if not rag_context:
        return {'messages':[]}

    print('\n [审计者介入] 正在核查生成内容是否包含幻觉......')

    checker_llm = ChatOpenAI(
        model=os.getenv('DEEPSEEK_MODEL'),
        api_key=os.getenv('DEEPSEEK_API_KEY'),
        base_url=os.getenv('DEEPSEEK_BASE_URL'),
        temperature=0.0
    )

    parser =JsonOutputParser(pydantic_object=FactCheckResult)

    # 检查提示词
    check_prompt = (
        f'你是一个冷酷的合规审计员，对比以下[知识库原文]和[AI生成的回复]。\n'
        f'[知识库原文]:\n{rag_context}\n'
        f'[AI生成的回复]：\n{last_message.content}\n'
        f'严查金额、职级门槛、天数！发现捏造请判 False 并给出修改意见\n\n'
        f'{parser.get_format_instructions()}'
    )

    response = checker_llm.invoke(check_prompt)

    # 手动解析 JSON   增加容错机制
    try:
        result = parser.invoke(response)
        is_pass = result.get('is_pass',True)  # 这里会用到上面写好的
        feedback = result.get('feedback','PASS')
    except Exception as e:
        print(f'[审计异常]JSON  解析失败，默认放行，原因:{e}')
        is_pass = True
        feedback = 'PASS'

    if is_pass:
        print('[审计通过]回答安全,无幻觉。')
        return {'messages':[]}
    else:
        print(f'[发现幻觉]拦截生成！ 审计意见：{feedback}')
        correction_msg = HumanMessage(
            content=f'[SYSTEM AUDIT FAILED] 事实错误反馈:{feedback},根据知识库原文重写，绝不可包含虚假数据'
        )
        return {'messages':[correction_msg]}

# 4.定义路由逻辑
def router_after_chatbot(state:AgentState):
    """ Chatbot输出后的路由判断 """
    last_message = state['messages'][-1]

    if last_message.tool_calls:
        return 'tools'
    else:
        return 'fact_check'

def router_after_fact_check(state:AgentState):
    """ 审计完成后的路由判断 """
    last_message = state['messages'][-1]
    if isinstance(last_message,HumanMessage):
        if state.get('loop_state',0)>4:
            print('[强制熔断]反思次数达到上限，放弃纠错')
            return 'end'
        print('[打回重写]图路由指针倒流回 chatbot 节点....')
        return 'chatbot'
    return 'end'

#5.构建状态图
workflow = StateGraph(AgentState)

workflow.add_node('chatbot',chatbot_node)
workflow.add_node('fact_check',fact_check_node)
workflow.add_node('tools',tools_node)

workflow.add_edge(START,'chatbot')
workflow.add_conditional_edges('chatbot',router_after_chatbot,{
    'tools':'tools',
    'fact_check':'fact_check',
})# 第二个参数是 条件分边器 第三个 是映射关系  #######
workflow.add_edge('tools','chatbot')
workflow.add_conditional_edges('fact_check',router_after_fact_check,{
    'chatbot':'chatbot',
    'end':END
})
hr_agent_app=workflow.compile()