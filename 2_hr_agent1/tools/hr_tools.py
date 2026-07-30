"""
【说明】这个文件在你上传给我的资料里没有包含（只看到 graph_builder.py 引用了它）。
我根据 graph_builder.py 里用到的函数名，以及 database/mock_db.py 的表结构，
重新实现了一版功能等价的 hr_tools.py，让整个 demo 能跑通。

如果你本地仓库里已经有一份不同实现的 tools/hr_tools.py，
请用你自己的版本替换这个文件（只要函数名、参数、@tool 装饰保持一致即可）。
"""
from langchain_core.tools import tool
from database.mock_db import get_connection, query_db

RANK_ORDER = ['P3', 'P4', 'P5', 'P6', 'P7', 'P8']


def _rank_level(rank: str) -> int:
    try:
        return RANK_ORDER.index(rank.strip().upper())
    except ValueError:
        return 0


@tool
def get_employee_profile(uid: str) -> str:
    """
    查询员工的基本档案信息，包括姓名、职级、工作地点、入职年限。
    输入参数 uid 为员工唯一标识（工号）。
    """
    conn = get_connection()
    try:
        rows = query_db(
            conn,
            "select uid, name, rank, location, seniority, base_salary from employees where uid = ?",
            (uid,)
        )
    finally:
        conn.close()

    if not rows:
        return f'未找到工号为 {uid} 的员工档案。'

    e = rows[0]
    return (f"员工档案：姓名={e['name']}，工号={e['uid']}，职级={e['rank']}，"
            f"工作地点={e['location']}，入职年限={e['seniority']}年。")


@tool
def get_leave_balance(uid: str) -> str:
    """
    查询员工当前剩余的年假与病假天数。
    输入参数 uid 为员工唯一标识（工号）。
    """
    conn = get_connection()
    try:
        rows = query_db(
            conn,
            "select annual_leave_remaining, sick_leave_remaining from leave_balances where uid = ?",
            (uid,)
        )
    finally:
        conn.close()

    if not rows:
        return f'未找到工号为 {uid} 的假期余额记录。'

    b = rows[0]
    return f"剩余年假：{b['annual_leave_remaining']} 天；剩余病假：{b['sick_leave_remaining']} 天。"


@tool
def generate_employment_certificate(uid: str, cert_type: str) -> str:
    """
    为员工生成《在职证明》或《薪资收入证明》。
    输入参数：
    - uid：员工工号
    - cert_type：证明类型，取值为 "在职证明" 或 "薪资收入证明"
    薪资收入证明仅限 P5 及以上职级员工可直接自助生成，P4 及以下员工需线下办理。
    """
    conn = get_connection()
    try:
        rows = query_db(conn, "select name, rank, base_salary from employees where uid = ?", (uid,))
    finally:
        conn.close()

    if not rows:
        return f'未找到工号为 {uid} 的员工档案，无法生成证明。'

    e = rows[0]

    if '在职' in cert_type:
        return (f"【在职证明】\n兹证明 {e['name']}（工号 {uid}）为我司在职员工，职级 {e['rank']}。\n"
                f"（本证明由系统自动生成，加盖飞羽科技人力资源部电子章）")

    if '薪资' in cert_type or '收入' in cert_type:
        if _rank_level(e['rank']) < _rank_level('P5'):
            return ('根据《飞羽科技员工共享服务手册》规定，薪资收入证明仅限 P5 及以上职级员工自助生成。'
                    f"该员工当前职级为 {e['rank']}，请提示其在线提交工单，由 HR 线下核实后手工开具。")
        return (f"【薪资收入证明】\n兹证明 {e['name']}（工号 {uid}，职级 {e['rank']}）"
                f"在我司的月基本工资为人民币 {e['base_salary']} 元。\n"
                f"（本证明由系统自动生成，加盖飞羽科技人力资源部电子章）")

    return '未识别的证明类型，请指定为「在职证明」或「薪资收入证明」。'
