"""Trainable connectome-constrained RNN using the FLYNN recurrence."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pyarrow.feather as feather
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neurotransmitter_model import neurotransmitter_effects, resolved_neurotransmitters


def inverse_sigmoid(value):
    value = min(max(float(value), 1.0e-5), 1.0-1.0e-5)
    return math.log(value/(1.0-value))


def prepare_base_weights(folder, glutamate="negative"):
    folder = Path(folder)
    ids = np.load(folder / "body_ids.npy", mmap_mode="r")
    indptr = np.load(folder / "indptr.npy", mmap_mode="r")
    indices = np.load(folder / "indices.npy", mmap_mode="r")
    counts = np.load(folder / "synapse_counts.npy", mmap_mode="r")
    totals = np.load(folder / "incoming_sum.npy", mmap_mode="r")
    values = np.asarray(counts, dtype=np.float32).copy()
    for first in range(0, len(ids), 250_000):
        last = min(first+250_000, len(ids))
        start, stop = int(indptr[first]), int(indptr[last])
        inv = np.divide(1.0, totals[first:last], out=np.zeros(last-first, dtype=np.float32),
                        where=totals[first:last] > 0)
        values[start:stop] *= np.repeat(inv, np.diff(indptr[first:last+1]))

    nt = feather.read_feather(folder / "neurotransmitters.feather")
    names, _sources = resolved_neurotransmitters(nt)
    lookup = neurotransmitter_effects(glutamate)
    signs = np.ones(len(ids), dtype=np.float32)
    nt_positions = np.searchsorted(ids, nt.body.to_numpy(dtype=np.int64))
    signs[nt_positions] = names.map(lookup).fillna(1).to_numpy(dtype=np.float32)
    for first in range(0, len(values), 10_000_000):
        last = min(first+10_000_000, len(values))
        values[first:last] *= signs[indices[first:last]]
    return ids, indptr, indices, values


class FlynnPolicy(nn.Module):
    """One hidden state vector with connectome-masked recurrence and PPO heads."""

    def __init__(self, circuit_folder, receptor_indices, config, device):
        super().__init__()
        folder = Path(circuit_folder)
        ids, indptr, indices, base_values = prepare_base_weights(folder)
        self.node_count = len(ids)
        self.edge_count = len(indices)
        self.weight_training = config["weight_training"]
        self.preserve_sign = bool(config["preserve_transmitter_sign"])
        self.state_clip = float(config.get("state_clip", 1.0))
        self.register_buffer("receptor_indices", torch.as_tensor(receptor_indices, dtype=torch.int64), persistent=False)

        if self.weight_training == "per_edge":
            # PyTorch's sparse CSR backward may materialize an N x N dense
            # gradient for trainable sparse values.  Keep the edge list and
            # aggregate contributions with index_add instead.
            edge_rows = np.repeat(
                np.arange(self.node_count, dtype=np.int64), np.diff(indptr)
            )
            self.register_buffer(
                "edge_rows", torch.from_numpy(edge_rows), persistent=False
            )
            self.register_buffer(
                "edge_sources",
                torch.from_numpy(np.array(indices, dtype=np.int64, copy=True)),
                persistent=False,
            )
            initial_edges = torch.from_numpy(np.array(base_values, dtype=np.float32, copy=True))
            self.edge_values = nn.Parameter(initial_edges)
            self.register_buffer("base_signs", torch.sign(initial_edges), persistent=False)
            self.fixed_matrix = None
            self.source_log_gain = None
        elif self.weight_training in ("source_gain", "frozen"):
            # Make writable owning copies before handing the arrays to torch.
            # This also removes the read-only NumPy memmap warning.
            crow = torch.from_numpy(np.array(indptr, dtype=np.int64, copy=True))
            columns = torch.from_numpy(np.array(indices, dtype=np.int64, copy=True))
            fixed_values = torch.from_numpy(np.array(base_values, dtype=np.float32, copy=True))
            fixed_matrix = torch.sparse_csr_tensor(
                crow, columns, fixed_values,
                size=(self.node_count, self.node_count),
                check_invariants=False,
            )
            self.register_buffer("fixed_matrix", fixed_matrix, persistent=False)
            self.edge_values = None
            self.source_log_gain = (
                nn.Parameter(torch.zeros(self.node_count, dtype=torch.float32))
                if self.weight_training == "source_gain" else None
            )
        else:
            raise ValueError("weight_training must be per_edge, source_gain or frozen")

        frame = feather.read_feather(folder / "annotations.feather")
        classes = frame.superclass.fillna("unannotated").astype(str)
        class_names = ["unannotated"] + sorted(set(classes)-{"unannotated"})
        class_lookup = {name: index for index, name in enumerate(class_names)}
        node_classes = np.zeros(self.node_count, dtype=np.int64)
        node_classes[frame.graph_index.to_numpy(dtype=np.int64)] = classes.map(class_lookup).to_numpy(dtype=np.int64)
        self.register_buffer("node_classes", torch.as_tensor(node_classes, dtype=torch.int64), persistent=False)
        initial_logit = inverse_sigmoid(config["initial_leak"])
        self.leak_logits = nn.Parameter(torch.full((len(class_names),), initial_logit, dtype=torch.float32))
        self.bias = nn.Parameter(torch.full((self.node_count,), float(config["initial_bias"]), dtype=torch.float32))

        descending = frame.superclass.fillna("").eq("descending_neuron")
        descending_indices = frame.loc[descending, "graph_index"].to_numpy(dtype=np.int64)
        if not len(descending_indices):
            raise ValueError("no descending_neuron annotations in the selected circuit")
        self.register_buffer("descending_indices", torch.as_tensor(descending_indices, dtype=torch.int64), persistent=False)
        hidden = int(config["readout_hidden"])
        self.actor = nn.Sequential(
            nn.LayerNorm(len(descending_indices)),
            nn.Linear(len(descending_indices), hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2), nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.full((2,), -0.7, dtype=torch.float32))
        critic_hidden = int(config["critic_hidden"])
        self.critic = nn.Sequential(
            nn.Linear(7, critic_hidden), nn.Tanh(),
            nn.Linear(critic_hidden, critic_hidden), nn.Tanh(),
            nn.Linear(critic_hidden, 1),
        )
        self.to(device)

    def initial_state(self):
        return torch.zeros(self.node_count, dtype=torch.float32, device=self.bias.device)

    def recurrent(self, state):
        if self.edge_values is not None:
            # Each contribution is W_ij * h_j. index_add accumulates it into
            # target row i while keeping gradients sparse over the edge vector.
            contributions = self.edge_values * state[self.edge_sources]
            recurrent = torch.zeros_like(state)
            recurrent.index_add_(0, self.edge_rows, contributions)
            return recurrent
        effective_state = state
        if self.source_log_gain is not None:
            # W_ij*g_j*h_j == W @ (g*h).  Applying the trainable source gain
            # to the dense state keeps W fixed and prevents an N x N gradient.
            gains = torch.exp(torch.clamp(self.source_log_gain, -3.0, 3.0))
            effective_state = gains * state
        return torch.sparse.mm(self.fixed_matrix, effective_state[:, None]).squeeze(1)

    def step(self, state, receptor_values, privileged):
        sensory = torch.zeros_like(state)
        sensory.index_copy_(0, self.receptor_indices, receptor_values)
        recurrent = self.recurrent(state)
        leak = torch.sigmoid(self.leak_logits[self.node_classes])
        # Paper equation: h(t+1)=alpha*h(t)+(1-alpha)*tanh(W*h(t)+x(t)+b)
        next_state = leak*state + (1.0-leak)*torch.tanh(recurrent + sensory + self.bias)
        next_state = torch.clamp(next_state, -self.state_clip, self.state_clip)
        descending = next_state[self.descending_indices]
        action_mean = self.actor(descending)
        value = self.critic(privileged).squeeze(-1)
        return next_state, action_mean, value

    def project_parameters(self):
        if self.edge_values is not None and self.preserve_sign:
            with torch.no_grad():
                self.edge_values.copy_(self.base_signs * torch.abs(self.edge_values))
        with torch.no_grad():
            self.leak_logits.clamp_(-8.0, 8.0)
            self.bias.clamp_(-3.0, 3.0)

    @property
    def report(self):
        return {
            "nodes": self.node_count,
            "edges": int(self.edge_count),
            "receptors": int(len(self.receptor_indices)),
            "descending_neurons": int(len(self.descending_indices)),
            "weight_training": self.weight_training,
            "equation": "alpha*h + (1-alpha)*tanh(W*h + sensory + bias)",
        }
