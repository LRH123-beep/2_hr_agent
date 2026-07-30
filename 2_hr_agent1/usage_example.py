"""
使用示例：怎么调用这个 RAG / HR Agent 系统

运行方式：
    python usage_example.py

前提：
    1. 已经 pip install -r requirements.txt
    2. 已经配置好 .env（DEEPSEEK_API_KEY 等）
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from langchain_core.messages import HumanMessage, AIMessage


# ============================================================
# 用法一：只测试"知识库检索"本身，不经过大模型对话
# 适合你想单独调试 RAG 检索效果好不好的时候用
# ============================================================
def demo_search_only():
    print('\n========== 用法一：单独测试知识库检索 ==========')
    from agent.rag_pipeline import search_hr_policy

    query = '出差去上海住宿报销标准是多少？'
    # search_hr_policy 被 @tool 装饰过，直接调用要用 .invoke()
    result = search_hr_policy.invoke(query)
    print(f'检索问题：{query}\n')
    print(result)


# ============================================================
# 用法二：调用完整的 HR Agent（意图理解 -> 工具调用 -> 事实审计）
# 这是真正对外提供服务时用的方式，app.py 内部也是这样调用的
# ============================================================
def demo_full_agent():
    print('\n========== 用法二：调用完整 HR Agent ==========')
    from agent.graph_builder import hr_agent_app

    current_uid = '1002'  # 模拟"李四"（P4 职级）登录提问
    question = '我还有多少年假？帮我看看能不能开薪资证明'

    result = hr_agent_app.invoke({
        'messages': [HumanMessage(content=question)],
        'current_uid': current_uid,
        'loop_state': 0,
    })

    print(f'\n员工 UID：{current_uid}')
    print(f'提问：{question}\n')

    # 打印完整消息链（能看到中间调用了哪些工具、审计节点有没有介入）
    for msg in result['messages']:
        role = type(msg).__name__
        print(f'[{role}] {msg.content}')

    # 只取最终给用户看的那句回答
    final_reply = next(
        (m.content for m in reversed(result['messages']) if isinstance(m, AIMessage) and m.content),
        None
    )
    print(f'\n>>> 最终回复：{final_reply}')


# ============================================================
# 用法三：模拟多轮对话（和网页版一样，需要自己维护 messages 历史）
# ============================================================
def demo_multi_turn():
    print('\n========== 用法三：模拟多轮对话 ==========')
    from agent.graph_builder import hr_agent_app

    current_uid = '1001'
    messages = []  # 手动维护的对话历史

    turns = [
        '帮我开一份在职证明',
        '那薪资证明呢？',
    ]

    for user_text in turns:
        messages.append(HumanMessage(content=user_text))

        result = hr_agent_app.invoke({
            'messages': messages,
            'current_uid': current_uid,
            'loop_state': 0,
        })

        final_reply = next(
            (m.content for m in reversed(result['messages']) if isinstance(m, AIMessage) and m.content),
            '（无有效回复）'
        )

        print(f'\n用户：{user_text}')
        print(f'助理：{final_reply}')

        # 只把"用户提问 + 最终回复"存进历史，不保留中间的工具调用消息
        # （这样历史更干净，也是 app.py 里网页版的做法）
        messages.append(AIMessage(content=final_reply))


if __name__ == '__main__':
    demo_search_only()
    demo_full_agent()
    demo_multi_turn()
