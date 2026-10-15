import os
import sys
import time
import pandas as pd

from neuprint import (
    Client,
    NeuronCriteria as NC,
    fetch_neurons,
    fetch_adjacencies,
)


# ============================================================
# 설정
# ============================================================

SERVER = "https://neuprint.janelia.org"
DATASET = "male-cns:v1.0"

TOKEN = os.environ.get("NEUPRINT_TOKEN", "")
# 몇 단계까지 downstream을 추적할 것인가
MAX_DEPTH = 8

# 연결 weight가 이 값보다 작은 연결은 제외
#
# 1 = 가능한 모든 연결
# 2 = 아주 약한 1-synapse 연결 일부 제거
# 3 = 더 강한 연결 위주
#
# 처음에는 2~3 권장
MIN_WEIGHT = 2

# 한 번에 처리할 source neuron 수
BATCH_SIZE = 100

# 병렬 요청 수
THREADS = 4

# 결과 폴더
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "L2_downstream_trace")


# ============================================================
# 환경 확인
# ============================================================

if not TOKEN:
    print("NEUPRINT_TOKEN이 없습니다.")
    sys.exit(1)


os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# NeuPrint 연결
# ============================================================

print("=" * 70)
print("MaleCNS L2 Downstream 자동 추적")
print("=" * 70)

print()
print("[1] NeuPrint 연결")

client = Client(
    SERVER,
    dataset=DATASET,
    token=TOKEN
)

print("연결 성공")
print("Dataset:", client.dataset)
print()


# ============================================================
# L2 전체 검색
# ============================================================

print("=" * 70)
print("[2] L2 전체 neuron 검색")
print("=" * 70)

l2_neurons, _ = fetch_neurons(
    NC(type="L2"),
    client=client
)

if l2_neurons.empty:
    print()
    print("L2 neuron을 찾지 못했습니다.")
    sys.exit(1)


l2_ids = (
    l2_neurons["bodyId"]
    .astype("int64")
    .tolist()
)

print()
print("L2 neuron 수:", len(l2_ids))
print()

print(
    l2_neurons[
        [
            c for c in
            ["bodyId", "type", "instance", "superclass",
             "class", "subclass", "somaSide"]
            if c in l2_neurons.columns
        ]
    ].to_string(index=False)
)

print()


# ============================================================
# L2 저장
# ============================================================

l2_neurons.to_csv(
    os.path.join(OUTPUT_DIR, "level_0_L2_neurons.csv"),
    index=False
)


# ============================================================
# BFS 준비
# ============================================================

# 현재 단계에서 탐색할 neuron
frontier = set(l2_ids)

# 지금까지 한 번이라도 방문한 neuron
visited = set(l2_ids)

# 모든 neuron의 최소 depth
node_depth = {
    body_id: 0
    for body_id in l2_ids
}

# 전체 연결
all_connections = []


# ============================================================
# Downstream BFS
# ============================================================

for depth in range(1, MAX_DEPTH + 1):

    print()
    print("=" * 70)
    print(f"[3-{depth}] DOWNSTREAM DEPTH {depth}")
    print("=" * 70)

    print()
    print("현재 frontier neuron:", len(frontier))
    print("현재까지 발견한 neuron:", len(visited))
    print()

    if not frontier:
        print("더 이상 downstream neuron이 없습니다.")
        break


    source_ids = list(frontier)


    # --------------------------------------------------------
    # downstream 연결 조회
    # --------------------------------------------------------

    print("NeuPrint에서 downstream 연결 검색 중...")

    start_time = time.time()

    try:

        neuron_info, conn_df = fetch_adjacencies(
            sources=source_ids,
            targets=None,
            rois=None,

            # 약한 연결 제거
            min_total_weight=MIN_WEIGHT,

            # ROI별로 나누지 않고 총 weight만 사용
            omit_rois=True,

            # 요청 크기
            batch_size=BATCH_SIZE,

            # 병렬 처리
            threads=THREADS,

            # 기본 weight만 사용
            weight_props=["weight"],

            client=client
        )

    except Exception as e:

        print()
        print("DOWNSTREAM 검색 실패")
        print(type(e).__name__)
        print(e)

        sys.exit(1)


    elapsed = time.time() - start_time

    print(
        f"검색 완료 ({elapsed:.1f}초)"
    )


    # --------------------------------------------------------
    # 연결이 없는 경우
    # --------------------------------------------------------

    if conn_df.empty:

        print()
        print("이 단계에서 downstream 연결이 없습니다.")
        break


    # --------------------------------------------------------
    # self-loop 제거
    # --------------------------------------------------------

    conn_df = conn_df[
        conn_df["bodyId_pre"] != conn_df["bodyId_post"]
    ].copy()


    if conn_df.empty:

        print()
        print("유효한 downstream 연결이 없습니다.")
        break


    # --------------------------------------------------------
    # 현재 단계 정보 추가
    # --------------------------------------------------------

    conn_df["depth"] = depth


    # source가 실제 현재 frontier인지 확인
    conn_df = conn_df[
        conn_df["bodyId_pre"].isin(frontier)
    ].copy()


    if conn_df.empty:

        print()
        print("현재 frontier에서 연결된 neuron이 없습니다.")
        break


    # --------------------------------------------------------
    # 새로운 neuron 찾기
    # --------------------------------------------------------

    downstream_ids = set(
        conn_df["bodyId_post"]
        .astype("int64")
        .tolist()
    )

    new_ids = downstream_ids - visited


    print()
    print("현재 단계 연결 수:", len(conn_df))
    print("현재 단계 downstream neuron:", len(downstream_ids))
    print("새롭게 발견한 neuron:", len(new_ids))


    # --------------------------------------------------------
    # neuron depth 기록
    # --------------------------------------------------------

    for body_id in new_ids:

        node_depth[body_id] = depth


    # --------------------------------------------------------
    # 전체 연결에 추가
    # --------------------------------------------------------

    all_connections.append(
        conn_df[
            [
                "bodyId_pre",
                "bodyId_post",
                "weight",
                "depth"
            ]
        ].copy()
    )


    # --------------------------------------------------------
    # 현재 단계 연결 저장
    # --------------------------------------------------------

    conn_output = os.path.join(
        OUTPUT_DIR,
        f"level_{depth}_connections.csv"
    )

    conn_df.to_csv(
        conn_output,
        index=False
    )


    # --------------------------------------------------------
    # 새 neuron 정보 저장
    # --------------------------------------------------------

    if new_ids:

        print()
        print("새 neuron annotation 조회 중...")

        try:

            new_neurons, _ = fetch_neurons(
                list(new_ids),
                client=client
            )

            new_neurons["depth"] = depth

            neuron_output = os.path.join(
                OUTPUT_DIR,
                f"level_{depth}_neurons.csv"
            )

            new_neurons.to_csv(
                neuron_output,
                index=False
            )

            print(
                "새 neuron annotation 저장:",
                neuron_output
            )

        except Exception as e:

            print(
                "annotation 조회 실패:",
                e
            )


    # --------------------------------------------------------
    # visited 업데이트
    # --------------------------------------------------------

    visited.update(new_ids)


    # --------------------------------------------------------
    # 다음 frontier
    # --------------------------------------------------------

    frontier = new_ids


    # --------------------------------------------------------
    # 현재 단계 상위 연결 출력
    # --------------------------------------------------------

    print()
    print("상위 downstream 연결:")

    display_cols = [
        "bodyId_pre",
        "bodyId_post",
        "weight"
    ]

    print(
        conn_df
        .sort_values(
            "weight",
            ascending=False
        )
        [display_cols]
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# 전체 연결 통합
# ============================================================

print()
print("=" * 70)
print("[4] 전체 결과 통합")
print("=" * 70)


if all_connections:

    all_conn_df = pd.concat(
        all_connections,
        ignore_index=True
    )

else:

    all_conn_df = pd.DataFrame(
        columns=[
            "bodyId_pre",
            "bodyId_post",
            "weight",
            "depth"
        ]
    )


# ============================================================
# 전체 connection 저장
# ============================================================

all_conn_output = os.path.join(
    OUTPUT_DIR,
    "L2_all_downstream_connections.csv"
)

all_conn_df.to_csv(
    all_conn_output,
    index=False
)


# ============================================================
# 전체 neuron 목록 생성
# ============================================================

all_nodes = set(
    l2_ids
)

if not all_conn_df.empty:

    all_nodes.update(
        all_conn_df["bodyId_pre"]
        .astype("int64")
        .tolist()
    )

    all_nodes.update(
        all_conn_df["bodyId_post"]
        .astype("int64")
        .tolist()
    )


nodes_df = pd.DataFrame({
    "bodyId": list(all_nodes)
})

nodes_df["depth"] = (
    nodes_df["bodyId"]
    .map(node_depth)
)


nodes_output = os.path.join(
    OUTPUT_DIR,
    "L2_all_downstream_nodes.csv"
)

nodes_df.to_csv(
    nodes_output,
    index=False
)


# ============================================================
# 최종 결과
# ============================================================

print()
print("=" * 70)
print("추적 완료")
print("=" * 70)

print()
print("시작 L2 neuron:", len(l2_ids))
print("전체 발견 neuron:", len(all_nodes))
print("전체 connection:", len(all_conn_df))
print("최종 depth:", MAX_DEPTH)

print()
print("결과 폴더:")
print(OUTPUT_DIR)

print()
print("주요 파일:")
print()
print("L2:")
print(
    os.path.join(
        OUTPUT_DIR,
        "level_0_L2_neurons.csv"
    )
)

print()
print("전체 연결:")
print(all_conn_output)

print()
print("전체 neuron:")
print(nodes_output)

print()
print("=" * 70)