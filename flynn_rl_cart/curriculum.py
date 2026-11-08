"""Three-stage target-motion curriculum with prior-stage replay."""

from __future__ import annotations

from collections import deque


STAGE_NAMES = {1: "stationary", 2: "horizontal", 3: "free_motion"}


class Curriculum:
    def __init__(self, config, rng):
        self.config = config
        self.rng = rng
        self.current_stage = 1
        self.total_episodes = {1: 0, 2: 0, 3: 0}
        window = int(config["promotion_window"])
        self.outcomes = {stage: deque(maxlen=window) for stage in (1, 2, 3)}

    def sample_stage(self):
        replay = float(self.config["previous_stage_replay_probability"])
        if self.current_stage > 1 and self.rng.random() < replay:
            return int(self.rng.integers(1, self.current_stage))
        return self.current_stage

    def record(self, stage, success):
        self.total_episodes[stage] += 1
        self.outcomes[stage].append(bool(success))
        if stage != self.current_stage or self.current_stage == 3:
            return False
        minimum = int(self.config["minimum_episodes_per_stage"])
        window = int(self.config["promotion_window"])
        if self.total_episodes[stage] < minimum or len(self.outcomes[stage]) < window:
            return False
        success_rate = sum(self.outcomes[stage]) / len(self.outcomes[stage])
        if success_rate >= float(self.config["promotion_success_rate"]):
            self.current_stage += 1
            return True
        return False

    def success_rate(self, stage=None):
        stage = stage or self.current_stage
        values = self.outcomes[stage]
        return float(sum(values)/len(values)) if values else 0.0

    def state_dict(self):
        return {
            "current_stage": self.current_stage,
            "total_episodes": self.total_episodes,
            "outcomes": {key: list(value) for key, value in self.outcomes.items()},
        }

    def load_state_dict(self, state):
        self.current_stage = int(state["current_stage"])
        self.total_episodes = {int(k): int(v) for k, v in state["total_episodes"].items()}
        for key, values in state["outcomes"].items():
            self.outcomes[int(key)].extend(bool(value) for value in values)
