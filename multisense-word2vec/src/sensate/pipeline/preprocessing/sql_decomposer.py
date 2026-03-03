import re
import hashlib
from concurrent.futures import ProcessPoolExecutor
import os


JOIN_PATTERN = re.compile(
    r"JOIN\s+\w+(?:\s+\w+)?\s+ON\s+(.*?)(?=JOIN|WHERE|$)",
    re.DOTALL,
)

WHERE_PATTERN = re.compile(
    r"WHERE\s+(.*?);?$",
    re.DOTALL,
)

AND_SPLIT = re.compile(r"\s+AND\s+")

TABLE_ALIAS_PATTERN = re.compile(
    r"(FROM|JOIN)\s+(\w+)(?:\s+(\w+))?",
    re.IGNORECASE,
)


def extract_alias_mapping(sql: str):
    mapping = {}
    matches = TABLE_ALIAS_PATTERN.findall(sql)

    for _, table, alias in matches:
        table = table.upper()
        if alias:
            mapping[alias.upper()] = table
        mapping[table] = table  # self map

    return mapping


def replace_alias_with_table(condition: str, alias_map: dict):
    for alias, table in alias_map.items():
        condition = re.sub(rf"\b{alias}\.", f"{table}.", condition)
    return condition


def decompose_operators_fast(sql: str):
    sql = sql.upper()

    alias_map = extract_alias_mapping(sql)

    # JOIN operators (root first)
    join_ops = JOIN_PATTERN.findall(sql)
    join_ops = [j.strip() for j in reversed(join_ops)]

    join_ops = [
        replace_alias_with_table(op, alias_map)
        for op in join_ops
    ]

    # WHERE operators
    where_ops = []
    m = WHERE_PATTERN.search(sql)
    if m:
        conditions = m.group(1)
        parts = AND_SPLIT.split(conditions)

        per_table = {}
        for cond in parts:
            cond = cond.strip()
            cond = replace_alias_with_table(cond, alias_map)

            table = cond.split(".")[0]
            per_table.setdefault(table, []).append(cond)

        for v in per_table.values():
            where_ops.append(" AND ".join(v))

    return join_ops + where_ops



def operator_hash(op: str):
    return hashlib.md5(op.encode()).hexdigest()



def process_single_sql(sql: str):
    ops = decompose_operators_fast(sql)
    hashed = [operator_hash(op) for op in ops]

    return {
        "operators": ops,      # giữ nguyên số
        "hashes": hashed
    }



def process_sql_batch_parallel(sql_list, workers=None):
    if workers is None:
        workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(process_single_sql, sql_list))

    return results


if __name__ == "__main__":

    query_original = """
    SELECT *
    FROM users
    JOIN messages
      ON users.id = messages.sender_id
    JOIN friends
      ON users.id = friends.first_id
     AND messages.receiver_id = friends.second_id
    """

    query_extended = """
    SELECT *
    FROM users
    JOIN messages
      ON users.id = messages.sender_id
    JOIN friends
      ON users.id = friends.first_id
     AND messages.receiver_id = friends.second_id
    JOIN groups
      ON friends.group_id = groups.id
    JOIN posts
      ON users.id = posts.user_id
    JOIN comments
      ON posts.id = comments.post_id
    WHERE users.age > 25
      AND users.id > 1000
      AND messages.created_at >= '2024-01-01'
      AND friends.status = 'ACTIVE'
      AND groups.type = 'PRIVATE'
      AND posts.created_at >= '2023-01-01'
      AND comments.is_deleted = false
    """

    batch = [query_original, query_extended]

    results = process_sql_batch_parallel(batch)

    for idx, r in enumerate(results):
        print(f"\n===== Query {idx} =====")
        for i, op in enumerate(r["operators"], 1):
            print(f"Node {i}: {op}")