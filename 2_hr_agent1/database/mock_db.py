import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / 'db' / 'employees.db'

def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """
    业务运行时连接函数。
    【修改说明】原版本在数据库不存在时会直接抛出 FileNotFoundError，
    要求用户手动运行 `python database/mock_db.py` 初始化。
    这在云端部署时会导致服务直接崩溃（没有人能帮你手动跑一遍脚本），
    所以这里改成：数据库不存在就自动初始化一份演示数据。
    """
    if not db_path.exists():
        print('[提示] 未检测到员工数据库，正在自动初始化演示数据 ...')
        return init_db(db_path)

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute('PRAGMA foreign_keys = ON')
    return conn

def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """
    数据库初始化并数据落盘。
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute('PRAGMA foreign_keys = ON')
    cursor = conn.cursor()

    cursor.execute("""
    create table if not exists employees (
        uid text primary key,
        name text not null,
        rank text not null,
        location text not null,
        seniority integer not null,
        base_salary integer not null
    )
    """)

    cursor.execute('''
    create table if not exists leave_balances (
        uid text primary key,
        annual_leave_remaining integer not null,
        sick_leave_remaining integer not null,
        foreign key (uid) references employees (uid)
    )
    ''')

    # 清空旧数据（确保幂等性）
    cursor.execute('''delete from employees''')
    cursor.execute('''delete from leave_balances where 1=1''')

    test_employees = [
        ('1001', '张三', 'P5', '北京', 2, 18000),
        ('1002', '李四', 'P4', '成都', 4, 9000),
        ('1003', '王五', 'P7', '上海', 5, 35000),
        ('1004', '赵六', 'P3', '深圳', 0, 7500),
    ]

    test_balances = [
        ('1001', 6, 10),
        ('1002', 7, 12),
        ('1003', 14, 15),
        ('1004', 2, 5),
    ]

    cursor.executemany("insert into employees values (?, ?, ?, ?, ?, ?)", test_employees)
    cursor.executemany("insert into leave_balances values (?, ?, ?)", test_balances)

    conn.commit()

    print('[成功] 实体数据库已成功落盘')
    print(f'数据库路径: {db_path}')
    return conn

def query_db(conn: sqlite3.Connection, sql: str, params: tuple = ()):
    """通用查询函数"""
    cursor = conn.cursor()
    cursor.execute(sql, params)
    columns = [col[0] for col in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]

def close_db(conn: sqlite3.Connection):
    """安全关闭数据库"""
    if conn:
        conn.close()
        print('数据库连接已安全关闭。')

if __name__ == '__main__':
    print('正在执行数据库手动初始化操作')
    standalone_conn = init_db()
    close_db(standalone_conn)
