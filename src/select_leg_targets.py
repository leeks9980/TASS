"""Select anatomically annotated leg motor targets from the existing L2 trace.

These are reachable candidates in the collected graph, not proven visual behavior
controllers. Original body IDs, sides and neurotransmitter annotations are retained.
"""

import csv
from collections import Counter
from pathlib import Path


BASE = Path(__file__).resolve().parent.parent / "L2_downstream_trace"
LEG_SUBCLASSES = {"fl": "front", "ml": "middle", "hl": "hind"}
FIELDS = [
    "bodyId", "type", "instance", "superclass", "subclass", "somaSide",
    "rootSide", "somaNeuromere", "exitNerve", "mancType", "mancBodyid",
    "consensusNt", "predictedNt", "predictedNtConfidence", "depth",
]


def select_targets(source, destination):
    csv.field_size_limit(100_000_000)
    counts = Counter()
    seen = set()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        required = {"bodyId", "superclass", "subclass"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing required annotation columns: {required}")
        with destination.open("w", encoding="utf-8", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=FIELDS + ["leg_group", "selection_basis"])
            writer.writeheader()
            for row in reader:
                if row["superclass"] != "vnc_motor" or row["subclass"] not in LEG_SUBCLASSES:
                    continue
                if row["bodyId"] in seen:
                    continue
                seen.add(row["bodyId"])
                leg = LEG_SUBCLASSES[row["subclass"]]
                selected = {key: row.get(key, "") for key in FIELDS}
                selected.update(leg_group=leg, selection_basis="vnc_motor + fl/ml/hl annotation")
                writer.writerow(selected)
                counts[leg] += 1
    return counts


if __name__ == "__main__":
    output = BASE / "leg_motor_targets.csv"
    result = select_targets(BASE / "L2_downstream_all_neurons.csv", output)
    print("Leg motor candidates:", dict(result), "total:", sum(result.values()))
    print("Saved:", output)
