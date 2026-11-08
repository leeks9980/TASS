"""Headless PPO training for the connectome-constrained light-following cart."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from curriculum import Curriculum, STAGE_NAMES
from environment import LightTargetCartEnv
from flynn_model import FlynnPolicy
from ppo import generalized_advantage, ppo_sequence_update, sample_action


HERE = Path(__file__).resolve().parent


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else (HERE/path).resolve()


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU를 사용하도록 설정했지만 현재 PyTorch에서 CUDA를 사용할 수 없습니다. "
            "NVIDIA 드라이버와 CUDA 지원 PyTorch 설치를 확인하세요. "
            f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}, "
            f"cuda_available={torch.cuda.is_available()}"
        )
    # torch.cuda.set_device() requires an explicit CUDA index.  Treat the
    # conventional config value "cuda" as the first GPU.
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    return device


def save_checkpoint(path, policy, optimizer, curriculum, total_steps, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "curriculum": curriculum.state_dict(),
        "total_steps": total_steps,
        "config": config,
        "model_report": policy.report,
    }, path)


def load_checkpoint(path, policy, optimizer, curriculum, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    policy.load_state_dict(checkpoint["policy"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    curriculum.load_state_dict(checkpoint["curriculum"])
    return int(checkpoint["total_steps"])


@torch.no_grad()
def evaluate(policy, env, stage, device, episodes=10):
    successes, returns, lengths = [], [], []
    for _ in range(episodes):
        receptors, privileged = env.reset(stage)
        state = policy.initial_state()
        total_reward = 0.0
        for step in range(int(env.config["max_episode_steps"])):
            receptor_tensor = torch.as_tensor(receptors, device=device)
            privileged_tensor = torch.as_tensor(privileged, device=device)
            state, mean, _value = policy.step(state, receptor_tensor, privileged_tensor)
            receptors, privileged, reward, done, info = env.step(torch.tanh(mean).cpu().numpy())
            total_reward += reward
            if done:
                successes.append(bool(info["success"]))
                returns.append(total_reward)
                lengths.append(step+1)
                break
    return {
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
    }


def train(config_path, resume=None):
    config = load_json(config_path)
    ppo_config = config["ppo"]
    seed = int(ppo_config["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = choose_device(ppo_config["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(seed)
    curriculum = Curriculum(config["curriculum"], rng)
    circuit = resolve(config["paths"]["circuit"])
    env = LightTargetCartEnv(
        resolve(config["paths"]["mujoco_model"]), circuit,
        config["vision"], config["environment"], config["reward"],
        config["curriculum"], seed,
    )
    evaluation_env = LightTargetCartEnv(
        resolve(config["paths"]["mujoco_model"]), circuit,
        config["vision"], config["environment"], config["reward"],
        config["curriculum"], seed+10_000,
    )
    policy = FlynnPolicy(circuit, env.vision.indices, config["network"], device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=float(ppo_config["learning_rate"]))
    checkpoint_dir = resolve(config["paths"]["checkpoint_dir"])
    log_dir = resolve(config["paths"]["log_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    total_steps = 0
    if resume:
        total_steps = load_checkpoint(Path(resume), policy, optimizer, curriculum, device)

    stage = curriculum.sample_stage()
    receptors, privileged = env.reset(stage)
    state = policy.initial_state()
    next_checkpoint = total_steps+int(ppo_config["checkpoint_every_steps"])
    next_evaluation = total_steps+int(ppo_config["evaluation_every_steps"])
    log_path = log_dir/"training.jsonl"
    device_report = {"device": str(device)}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        device_report.update({
            "gpu": properties.name,
            "gpu_memory_gb": round(properties.total_memory/(1024**3), 2),
            "torch_cuda": torch.version.cuda,
        })
    print(json.dumps({**device_report, "model": policy.report,
                      "photoreceptors": env.vision.report}, ensure_ascii=False), flush=True)
    try:
        with log_path.open("a", encoding="utf-8") as log:
            while total_steps < int(ppo_config["total_environment_steps"]):
                started = time.perf_counter()
                rollout = {key: [] for key in (
                    "receptors", "privileged", "actions", "log_probabilities", "values",
                    "rewards", "dones")}
                rollout["hidden_starts"] = {}
                sequence_length = int(ppo_config["sequence_length"])
                for index in range(int(ppo_config["rollout_steps"])):
                    if index % sequence_length == 0:
                        rollout["hidden_starts"][index] = state.detach().to("cpu", dtype=torch.float16)
                    receptor_tensor = torch.as_tensor(receptors, device=device)
                    privileged_tensor = torch.as_tensor(privileged, device=device)
                    with torch.no_grad():
                        next_state, mean, value = policy.step(state, receptor_tensor, privileged_tensor)
                        action, log_probability = sample_action(policy, mean)
                    next_receptors, next_privileged, reward, done, info = env.step(action.cpu().numpy())
                    rollout["receptors"].append(np.asarray(receptors, dtype=np.float32))
                    rollout["privileged"].append(np.asarray(privileged, dtype=np.float32))
                    rollout["actions"].append(action.cpu().numpy().astype(np.float32))
                    rollout["log_probabilities"].append(float(log_probability))
                    rollout["values"].append(float(value))
                    rollout["rewards"].append(float(reward))
                    rollout["dones"].append(bool(done))
                    total_steps += 1
                    if done:
                        promoted = curriculum.record(stage, info["success"])
                        if promoted:
                            print(f"curriculum promoted to {STAGE_NAMES[curriculum.current_stage]}", flush=True)
                        stage = curriculum.sample_stage()
                        receptors, privileged = env.reset(stage)
                        state = policy.initial_state()
                    else:
                        receptors, privileged, state = next_receptors, next_privileged, next_state.detach()
                with torch.no_grad():
                    bootstrap = float(policy.critic(torch.as_tensor(privileged, device=device)).squeeze())
                advantages, returns = generalized_advantage(
                    rollout["rewards"], rollout["values"], rollout["dones"], bootstrap,
                    float(ppo_config["gamma"]), float(ppo_config["gae_lambda"]),
                )
                rollout["advantages"], rollout["returns"] = advantages, returns
                metrics = ppo_sequence_update(policy, optimizer, rollout, ppo_config, device)
                record = {
                    "steps": total_steps,
                    "stage": curriculum.current_stage,
                    "stage_name": STAGE_NAMES[curriculum.current_stage],
                    "rolling_success": curriculum.success_rate(),
                    "mean_rollout_reward": float(np.mean(rollout["rewards"])),
                    "seconds": time.perf_counter()-started,
                    **metrics,
                }
                print(json.dumps(record, ensure_ascii=False), flush=True)
                log.write(json.dumps(record, ensure_ascii=False)+"\n"); log.flush()

                if total_steps >= next_evaluation:
                    evaluation = evaluate(policy, evaluation_env, curriculum.current_stage, device)
                    evaluation_record = {"steps": total_steps, "evaluation": evaluation,
                                         "stage": curriculum.current_stage}
                    print(json.dumps(evaluation_record, ensure_ascii=False), flush=True)
                    log.write(json.dumps(evaluation_record, ensure_ascii=False)+"\n"); log.flush()
                    next_evaluation += int(ppo_config["evaluation_every_steps"])
                if total_steps >= next_checkpoint:
                    save_checkpoint(checkpoint_dir/f"step_{total_steps:09d}.pt", policy, optimizer,
                                    curriculum, total_steps, config)
                    save_checkpoint(checkpoint_dir/"latest.pt", policy, optimizer,
                                    curriculum, total_steps, config)
                    next_checkpoint += int(ppo_config["checkpoint_every_steps"])
    finally:
        env.close(); evaluation_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE/"config.json")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    train(args.config.resolve(), args.resume)
