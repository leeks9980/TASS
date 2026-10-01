"""Extract L2 -> leg motor directed-walk corridors from the full MaleCNS graph.

Unannotated segments are retained. Reachability is structural, not proof of a
functional visual motor circuit. Both unrestricted and bounded results are saved.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import breadth_first_order

DATA = ROOT / "data" / "male-cns-v1.0"
ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
NT = "body-neurotransmitters-male-cns-v1.0.feather"
LEG_GROUPS = {"fl": "front", "ml": "middle", "hl": "hind"}


def log(message):
    print(message, flush=True)


def batches():
    with pa.memory_map(str(DATA / WEIGHTS), "r") as source:
        reader = ipc.open_file(source)
        for i in range(reader.num_record_batches):
            yield reader.get_batch(i)


def arrays(batch):
    return tuple(batch.column(name).to_numpy() for name in ("body_pre", "body_post", "weight"))


def valid_edges(pre, post, weight, minimum):
    # Preserve self connections too; BFS naturally ignores already visited nodes.
    return weight >= minimum


def distances(graph, seeds):
    """One BFS with a virtual root gives nearest-seed distances in O(V+E)."""
    size = graph.shape[0]
    seeds = np.unique(np.asarray(seeds, dtype=np.int32))
    indptr = np.concatenate([graph.indptr, [graph.nnz + len(seeds)]])
    augmented = csr_matrix((np.concatenate([graph.data, np.ones(len(seeds), dtype=bool)]),
                            np.concatenate([graph.indices, seeds]), indptr),
                           shape=(size + 1, size + 1))
    order, predecessors = breadth_first_order(augmented, size, directed=True)
    distance = np.full(size + 1, -1, dtype=np.int32)
    # The virtual root has distance -1 so real seeds have distance zero.
    for node in order[1:]:
        distance[node] = distance[predecessors[node]] + 1
    return distance[:-1]


def corridor_masks(forward, backward, pre, post, max_hops=None):
    keep_nodes = (forward >= 0) & (backward >= 0)
    keep_edges = keep_nodes[pre] & keep_nodes[post]
    if max_hops is not None:
        keep_nodes &= forward.astype(np.int64) + backward >= 0
        keep_nodes &= forward.astype(np.int64) + backward <= max_hops
        keep_edges &= forward[pre].astype(np.int64) + 1 + backward[post] <= max_hops
    return keep_nodes, keep_edges


def build_graph(minimum, work):
    chunks = []
    edge_count = raw_count = 0
    for batch in batches():
        pre, post, weight = arrays(batch)
        raw_count += len(pre)
        mask = valid_edges(pre, post, weight, minimum)
        edge_count += int(mask.sum())
        chunks.append(np.unique(np.concatenate([pre[mask], post[mask]])))
    log(f"Raw rows: {raw_count:,}; eligible edges: {edge_count:,}; merging body IDs")
    ids = np.unique(np.concatenate(chunks))
    del chunks
    gc.collect()
    if len(ids) >= np.iinfo(np.int32).max:
        raise ValueError("Graph exceeds the supported int32 node-index range")
    log(f"Full graph nodes: {len(ids):,}; indexing edges")
    rows = np.lib.format.open_memmap(work / "source_indices.npy", mode="w+", dtype=np.int32, shape=(edge_count,))
    cols = np.lib.format.open_memmap(work / "target_indices.npy", mode="w+", dtype=np.int32, shape=(edge_count,))
    offset = 0
    for batch in batches():
        pre, post, weight = arrays(batch)
        mask = valid_edges(pre, post, weight, minimum)
        length = int(mask.sum())
        rows[offset:offset + length] = np.searchsorted(ids, pre[mask])
        cols[offset:offset + length] = np.searchsorted(ids, post[mask])
        offset += length
    rows.flush()
    cols.flush()
    assert offset == edge_count
    log("Building sparse directed graph")
    graph = coo_matrix((np.ones(edge_count, dtype=bool), (rows, cols)), shape=(len(ids), len(ids))).tocsr()
    del rows, cols
    gc.collect()
    return ids, graph, raw_count, edge_count


def export_nodes(path, ids, mask, annotations, nt, forward, backward, seeds, targets):
    frame = pd.DataFrame({"bodyId": ids[mask], "from_L2_hops": forward[mask], "to_leg_motor_hops": backward[mask]})
    frame["is_L2_source"] = frame.bodyId.isin(seeds)
    frame["is_leg_motor_target"] = frame.bodyId.isin(targets)
    frame = frame.merge(annotations, on="bodyId", how="left", validate="one_to_one", indicator=True)
    frame["has_neuron_annotation"] = frame.pop("_merge").eq("both")
    frame = frame.merge(nt.rename(columns={"body": "bodyId"}), on="bodyId", how="left", validate="one_to_one")
    # Feather retains original lists and body IDs without CSV coercion.
    feather.write_feather(frame, path / "neurons.feather", compression="lz4")
    known = frame.loc[frame.has_neuron_annotation].copy()
    known.to_csv(path / "annotated_neurons.csv", index=False)
    return {"nodes": len(frame), "annotated_nodes": len(known),
            "unannotated_segments": int((~frame.has_neuron_annotation).sum()),
            "L2_sources_in_corridor": int(frame.is_L2_source.sum()),
            "leg_motor_targets_in_corridor": int(frame.is_leg_motor_target.sum()),
            "superclass_counts": {str(k): int(v) for k, v in frame.superclass.fillna("unannotated").value_counts().items()}}


def run(minimum, bound, output):
    started = time.time()
    manifest_path = DATA / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run download_connectome.py successfully first")
    output.mkdir(parents=True, exist_ok=True)
    work = output / "work"
    work.mkdir(exist_ok=True)
    annotations = feather.read_feather(DATA / ANNOTATIONS)
    nt = feather.read_feather(DATA / NT)
    sources = annotations.loc[annotations.type.eq("L2"), "bodyId"].to_numpy(dtype=np.int64)
    targets = annotations.loc[annotations.superclass.eq("vnc_motor") & annotations.subclass.isin(LEG_GROUPS)].copy()
    if len(sources) == 0 or len(targets) == 0:
        raise ValueError("No L2 sources or leg motor targets found in official annotations")
    targets["leg_group"] = targets.subclass.map(LEG_GROUPS)
    log(f"Official annotations: {len(annotations):,}; L2: {len(sources):,}; leg motors: {len(targets):,}")
    ids, graph, raw_count, eligible_count = build_graph(minimum, work)
    source_present = np.isin(sources, ids)
    target_present = targets.bodyId.isin(ids).to_numpy()
    log("Forward BFS from all L2 sources")
    forward = distances(graph, np.searchsorted(ids, sources[source_present]))
    log("Reverse BFS from all leg motor targets")
    backward = distances(graph.transpose().tocsr(), np.searchsorted(ids, targets.loc[target_present, "bodyId"]))
    targets["in_graph"] = target_present
    targets["from_L2_hops"] = -1
    targets.loc[target_present, "from_L2_hops"] = forward[np.searchsorted(ids, targets.loc[target_present, "bodyId"])]
    targets["reachable_from_L2"] = targets.from_L2_hops.ge(0)
    targets.to_csv(output / "leg_motor_targets.csv", index=False)
    source_report = annotations.loc[annotations.bodyId.isin(sources)].copy()
    source_report["in_graph"] = source_report.bodyId.isin(ids)
    source_report["to_leg_motor_hops"] = -1
    valid = source_report.in_graph
    source_report.loc[valid, "to_leg_motor_hops"] = backward[np.searchsorted(ids, source_report.loc[valid, "bodyId"])]
    source_report.to_csv(output / "L2_sources.csv", index=False)
    variants = {"unbounded": None, f"max_{bound}_hops": bound}
    reports = {}
    writers = {}
    sinks = {}
    node_masks = {}
    schema = pa.schema([("body_pre", pa.int64()), ("body_post", pa.int64()), ("weight", pa.int64())])
    for name, max_hops in variants.items():
        path = output / name
        path.mkdir(exist_ok=True)
        mask, _ = corridor_masks(forward, backward, np.array([], dtype=np.int32), np.array([], dtype=np.int32), max_hops)
        node_masks[name] = mask
        log(f"Exporting {name}: {int(mask.sum()):,} nodes")
        reports[name] = export_nodes(path, ids, mask, annotations, nt, forward, backward, sources, targets.bodyId)
        reports[name]["connection_rows"] = 0
        sinks[name] = pa.OSFile(str(path / "connections.feather"), "wb")
        writers[name] = ipc.new_file(sinks[name], schema, options=ipc.IpcWriteOptions(compression="lz4"))
    log("Exporting original weighted edges for both corridors")
    try:
        for batch in batches():
            pre, post, weight = arrays(batch)
            valid = valid_edges(pre, post, weight, minimum)
            if not valid.any():
                continue
            pre, post, weight = pre[valid], post[valid], weight[valid]
            row, col = np.searchsorted(ids, pre), np.searchsorted(ids, post)
            for name, max_hops in variants.items():
                keep = node_masks[name][row] & node_masks[name][col]
                if max_hops is not None:
                    keep &= forward[row].astype(np.int64) + 1 + backward[col] <= max_hops
                if keep.any():
                    writers[name].write_batch(pa.record_batch([pre[keep], post[keep], weight[keep]], schema=schema))
                    reports[name]["connection_rows"] += int(keep.sum())
    finally:
        for writer in writers.values():
            writer.close()
        for sink in sinks.values():
            sink.close()
    report = {"dataset": "male-cns:v1.0", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
              "selection": {"source_type": "L2", "target_superclass": "vnc_motor", "target_subclasses": LEG_GROUPS,
                            "min_weight": minimum, "max_hops_variant": bound,
                            "self_loops_included": True, "unannotated_segments_included": True,
                            "sensory_neurons_excluded": False},
              "raw_connection_rows": raw_count, "eligible_connection_rows": eligible_count,
              "unique_directed_edges": int(graph.nnz), "graph_nodes": len(ids),
              "official_annotation_rows": len(annotations), "L2_source_count": len(sources),
              "leg_motor_target_count": len(targets), "targets_reachable_from_L2": int(targets.reachable_from_L2.sum()),
              "leg_group_counts": {k: int(v) for k, v in targets.leg_group.value_counts().items()},
              "variants": reports, "elapsed_seconds": round(time.time() - started, 1),
              "limitations": ["Structural directed-walk corridor; not a proven functional or minimal circuit.",
                              "Walks can revisit nodes; recurrent connectivity can make corridors large.",
                              "All IDs in the published weight table participate, including unannotated segments.",
                              "Motor and sensory feedback outside the L2-to-target corridor is not guaranteed to be retained.",
                              "Synapse confidence filtering comes from the official minconf-0.5 release.",
                              "No motor-to-muscle mapping or executable neuron dynamics is inferred."]}
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log(json.dumps({"targets_reachable": report["targets_reachable_from_L2"], "variants": reports,
                    "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-weight", type=int, default=1)
    parser.add_argument("--max-hops", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "circuit_extract" / "weight_1")
    args = parser.parse_args()
    if args.min_weight < 1 or args.max_hops < 1:
        parser.error("min-weight and max-hops must be positive")
    run(args.min_weight, args.max_hops, args.output)
