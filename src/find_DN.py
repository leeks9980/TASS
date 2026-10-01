import os
import glob
import pandas as pd


# ============================================================
# 설정
# ============================================================

TRACE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "L2_downstream_trace")

OUTPUT_FILE = os.path.join(
    TRACE_DIR,
    "L2_downstream_analysis.csv"
)


# ============================================================
# 파일 검색
# ============================================================

files = glob.glob(
    os.path.join(
        TRACE_DIR,
        "level_*_neurons.csv"
    )
)

if not files:
    print("level_*_neurons.csv 파일을 찾지 못했습니다.")
    print("현재 폴더:", os.getcwd())
    print("검색 경로:", TRACE_DIR)
    exit()


print("=" * 70)
print("L2 Downstream 결과 분석")
print("=" * 70)

print()
print("발견한 neuron 파일:", len(files))
print()


# ============================================================
# 전체 neuron annotation 합치기
# ============================================================

all_neurons = []

for file in files:

    print("읽는 중:", file)

    df = pd.read_csv(file)

    all_neurons.append(df)


neurons = pd.concat(
    all_neurons,
    ignore_index=True
)


# bodyId 중복 제거
neurons = neurons.drop_duplicates(
    subset=["bodyId"],
    keep="first"
)


print()
print("전체 발견 neuron:", len(neurons))


# ============================================================
# Depth별 통계
# ============================================================

print()
print("=" * 70)
print("Depth별 neuron 수")
print("=" * 70)

if "depth" in neurons.columns:

    depth_count = (
        neurons
        .groupby("depth")
        .size()
        .reset_index(name="neuron_count")
        .sort_values("depth")
    )

    print(
        depth_count.to_string(
            index=False
        )
    )


# ============================================================
# superclass 분포
# ============================================================

if "superclass" in neurons.columns:

    print()
    print("=" * 70)
    print("Superclass 분포")
    print("=" * 70)

    superclass_count = (
        neurons["superclass"]
        .fillna("None")
        .value_counts()
    )

    print(
        superclass_count.to_string()
    )


# ============================================================
# class 분포
# ============================================================

if "class" in neurons.columns:

    print()
    print("=" * 70)
    print("Class 분포")
    print("=" * 70)

    class_count = (
        neurons["class"]
        .fillna("None")
        .value_counts()
    )

    print(
        class_count.head(100).to_string()
    )


# ============================================================
# Type 분포
# ============================================================

if "type" in neurons.columns:

    print()
    print("=" * 70)
    print("Type 분포")
    print("=" * 70)

    type_count = (
        neurons["type"]
        .fillna("None")
        .value_counts()
    )

    print(
        type_count.head(200).to_string()
    )


# ============================================================
# DN 검색
# ============================================================

print()
print("=" * 70)
print("Descending Neuron 검색")
print("=" * 70)


dn_mask = pd.Series(
    False,
    index=neurons.index
)


# superclass에서 검색
if "superclass" in neurons.columns:

    dn_mask |= (
        neurons["superclass"]
        .astype(str)
        .str.contains(
            "descending",
            case=False,
            na=False
        )
    )


# type에서 DN 검색
if "type" in neurons.columns:

    dn_mask |= (
        neurons["type"]
        .astype(str)
        .str.match(
            r"^DN",
            case=False,
            na=False
        )
    )


# instance에서 DN 검색
if "instance" in neurons.columns:

    dn_mask |= (
        neurons["instance"]
        .astype(str)
        .str.match(
            r"^DN",
            case=False,
            na=False
        )
    )


dn_neurons = neurons[dn_mask].copy()


print()
print("DN 후보:", len(dn_neurons))


if not dn_neurons.empty:

    columns = [
        c for c in [
            "bodyId",
            "type",
            "instance",
            "superclass",
            "class",
            "subclass",
            "somaSide",
            "somaNeuromere",
            "depth"
        ]
        if c in dn_neurons.columns
    ]

    print(
        dn_neurons[columns]
        .sort_values("depth")
        .to_string(index=False)
    )


# ============================================================
# VNC 관련 annotation 검색
# ============================================================

print()
print("=" * 70)
print("VNC 관련 annotation 검색")
print("=" * 70)


vnc_mask = pd.Series(
    False,
    index=neurons.index
)


for column in [
    "type",
    "instance",
    "superclass",
    "class",
    "subclass",
    "somaNeuromere"
]:

    if column not in neurons.columns:
        continue

    vnc_mask |= (
        neurons[column]
        .astype(str)
        .str.contains(
            "VNC|Ventral|thoracic|T1|T2|T3",
            case=False,
            na=False,
            regex=True
        )
    )


vnc_neurons = neurons[vnc_mask].copy()


print()
print("VNC 관련 후보:", len(vnc_neurons))


if not vnc_neurons.empty:

    columns = [
        c for c in [
            "bodyId",
            "type",
            "instance",
            "superclass",
            "class",
            "subclass",
            "somaSide",
            "somaNeuromere",
            "depth"
        ]
        if c in vnc_neurons.columns
    ]

    print(
        vnc_neurons[columns]
        .sort_values("depth")
        .to_string(index=False)
    )


# ============================================================
# DN 결과 저장
# ============================================================

dn_output = os.path.join(
    TRACE_DIR,
    "L2_DN_candidates.csv"
)

dn_neurons.to_csv(
    dn_output,
    index=False
)


# ============================================================
# VNC 결과 저장
# ============================================================

vnc_output = os.path.join(
    TRACE_DIR,
    "L2_VNC_candidates.csv"
)

vnc_neurons.to_csv(
    vnc_output,
    index=False
)


# ============================================================
# 전체 분석 결과 저장
# ============================================================

analysis_output = os.path.join(
    TRACE_DIR,
    "L2_downstream_all_neurons.csv"
)

neurons.to_csv(
    analysis_output,
    index=False
)


# ============================================================
# 완료
# ============================================================

print()
print("=" * 70)
print("분석 완료")
print("=" * 70)

print()
print("전체 neuron:")
print(analysis_output)

print()
print("DN 후보:")
print(dn_output)

print()
print("VNC 후보:")
print(vnc_output)