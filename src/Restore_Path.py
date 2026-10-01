import os
import pandas as pd
import networkx as nx

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "L2_downstream_trace")

CONNECTION_FILE = os.path.join(
    BASE, "L2_all_downstream_connections.csv"
)

NODES_FILE = os.path.join(
    BASE, "L2_all_downstream_nodes.csv"
)

L2_FILE = os.path.join(BASE, "level_0_L2_neurons.csv")

DN_FILE = os.path.join(
    BASE, "L2_DN_candidates.csv"
)

OUTPUT_DIR = os.path.join(BASE, "path_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# 1. 데이터 로드
# =========================================================

print("연결 데이터 로드 중...")

edges = pd.read_csv(CONNECTION_FILE)
nodes = pd.read_csv(NODES_FILE)
dn = pd.read_csv(DN_FILE)

print()
print("=== 파일 확인 ===")
print("연결:", edges.shape)
print("노드:", nodes.shape)
print("DN:", dn.shape)

print()
print("연결 컬럼:")
print(edges.columns.tolist())

print()
print("노드 컬럼:")
print(nodes.columns.tolist())


# =========================================================
# 2. bodyId 컬럼 확인
# =========================================================

possible_pre = [
    "bodyId_pre",
    "bodyId_source",
    "source",
]

possible_post = [
    "bodyId_post",
    "bodyId_target",
    "target",
]


def find_column(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


PRE = find_column(edges.columns, possible_pre)
POST = find_column(edges.columns, possible_post)

if PRE is None or POST is None:
    raise RuntimeError(
        f"연결 파일에서 source/target 컬럼을 찾지 못했습니다.\n"
        f"현재 컬럼: {edges.columns.tolist()}"
    )


print()
print("Source:", PRE)
print("Target:", POST)


# =========================================================
# 3. synapse weight 확인
# =========================================================

if "weight" in edges.columns:
    WEIGHT = "weight"
elif "synapses" in edges.columns:
    WEIGHT = "synapses"
else:
    WEIGHT = None

print("Weight:", WEIGHT)


# =========================================================
# 4. Directed graph 생성
# =========================================================

G = nx.DiGraph()

for _, row in edges.iterrows():

    source = int(row[PRE])
    target = int(row[POST])

    weight = 1

    if WEIGHT is not None:
        try:
            weight = int(row[WEIGHT])
        except:
            weight = 1

    # 같은 연결이 여러 번 존재할 경우
    # 가장 큰 weight 유지
    if G.has_edge(source, target):

        if weight > G[source][target]["weight"]:
            G[source][target]["weight"] = weight

    else:

        G.add_edge(
            source,
            target,
            weight=weight
        )


print()
print("=== Graph ===")
print("노드:", G.number_of_nodes())
print("연결:", G.number_of_edges())


# =========================================================
# 5. L2 시작점 찾기
# =========================================================

# The node index contains only bodyId and depth. Use the original L2 annotations.
l2_annotations = pd.read_csv(L2_FILE, usecols=["bodyId", "type"])
l2_nodes = l2_annotations.loc[
    l2_annotations["type"].eq("L2"), "bodyId"
].astype(int).tolist()

print()
print("L2 시작 뉴런:", len(l2_nodes))


# =========================================================
# 6. DN 찾기
# =========================================================

dn_ids = dn["bodyId"].astype(int).tolist()

print("DN 후보:", len(dn_ids))


# =========================================================
# 7. 실제 L2 → DN 경로 탐색
# =========================================================

paths = []

print()
print("L2 → DN 경로 탐색 시작...")
print()


for dn_id in dn_ids:

    best_paths = []

    for l2_id in l2_nodes:

        if l2_id not in G:
            continue

        if dn_id not in G:
            continue

        try:
            path = nx.shortest_path(
                G,
                source=l2_id,
                target=dn_id
            )

            best_paths.append(path)

        except nx.NetworkXNoPath:
            pass

    if not best_paths:
        continue

    # 가장 짧은 경로
    best_path = min(
        best_paths,
        key=len
    )

    paths.append({
        "dn_bodyId": dn_id,
        "l2_bodyId": best_path[0],
        "hops": len(best_path) - 1,
        "path": " -> ".join(
            str(x) for x in best_path
        )
    })


# =========================================================
# 8. 결과 저장
# =========================================================

path_df = pd.DataFrame(paths, columns=["dn_bodyId", "l2_bodyId", "hops", "path"])

path_df = path_df.sort_values(
    ["hops", "dn_bodyId"]
)

OUTPUT = os.path.join(
    OUTPUT_DIR,
    "L2_to_DN_paths.csv"
)

path_df.to_csv(
    OUTPUT,
    index=False
)


# =========================================================
# 9. 출력
# =========================================================

print()
print("================================")
print("L2 → DN 경로 탐색 완료")
print("================================")

print()
print("실제로 연결된 DN:", len(path_df))

print()

if len(path_df) > 0:

    print(path_df.to_string(index=False))

else:

    print("L2에서 DN까지 직접적인 경로를 찾지 못했습니다.")

print()
print("결과:")
print(OUTPUT)
