"""Minimal sequence PPO utilities for the large connectome recurrent policy."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.distributions import Normal


def distribution(policy, mean):
    std = torch.exp(torch.clamp(policy.log_std, -5.0, 1.0))
    return Normal(mean, std)


def sample_action(policy, mean):
    normal = distribution(policy, mean)
    raw = normal.rsample()
    action = torch.tanh(raw)
    log_probability = (normal.log_prob(raw)-torch.log(1.0-action.square()+1.0e-6)).sum()
    return action, log_probability


def evaluate_action(policy, mean, action):
    action = torch.clamp(action, -0.999999, 0.999999)
    raw = torch.atanh(action)
    normal = distribution(policy, mean)
    log_probability = (normal.log_prob(raw)-torch.log(1.0-action.square()+1.0e-6)).sum()
    entropy = normal.entropy().sum()
    return log_probability, entropy


def generalized_advantage(rewards, values, dones, bootstrap_value, gamma, gae_lambda):
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    advantages = np.zeros_like(rewards)
    gae = 0.0
    next_value = float(bootstrap_value)
    for index in range(len(rewards)-1, -1, -1):
        continuation = 1.0-dones[index]
        delta = rewards[index] + gamma*next_value*continuation-values[index]
        gae = delta + gamma*gae_lambda*continuation*gae
        advantages[index] = gae
        next_value = values[index]
    returns = advantages+values
    return advantages, returns


def ppo_sequence_update(policy, optimizer, rollout, config, device):
    advantages = rollout["advantages"]
    advantages = (advantages-advantages.mean())/(advantages.std()+1.0e-8)
    sequence_length = int(config["sequence_length"])
    starts = list(range(0, len(rollout["rewards"]), sequence_length))
    rng = np.random.default_rng()
    metrics = []
    for _epoch in range(int(config["update_epochs"])):
        rng.shuffle(starts)
        for start in starts:
            stop = min(start+sequence_length, len(rollout["rewards"]))
            state = rollout["hidden_starts"][start].to(device=device, dtype=torch.float32)
            policy_losses, value_losses, entropies = [], [], []
            for index in range(start, stop):
                receptors = torch.as_tensor(rollout["receptors"][index], device=device)
                privileged = torch.as_tensor(rollout["privileged"][index], device=device)
                state, mean, value = policy.step(state, receptors, privileged)
                action = torch.as_tensor(rollout["actions"][index], device=device)
                new_log_probability, entropy = evaluate_action(policy, mean, action)
                old_log_probability = torch.as_tensor(rollout["log_probabilities"][index], device=device)
                ratio = torch.exp(new_log_probability-old_log_probability)
                advantage = torch.as_tensor(advantages[index], device=device)
                clipped_ratio = torch.clamp(
                    ratio, 1.0-float(config["clip_ratio"]), 1.0+float(config["clip_ratio"])
                )
                policy_losses.append(-torch.minimum(ratio*advantage, clipped_ratio*advantage))
                target_return = torch.as_tensor(rollout["returns"][index], device=device)
                value_losses.append((value-target_return).square())
                entropies.append(entropy)
                if rollout["dones"][index]:
                    state = policy.initial_state()
            policy_loss = torch.stack(policy_losses).mean()
            value_loss = torch.stack(value_losses).mean()
            entropy = torch.stack(entropies).mean()
            loss = (policy_loss + float(config["value_loss_coefficient"])*value_loss -
                    float(config["entropy_coefficient"])*entropy)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), float(config["max_gradient_norm"])
            )
            optimizer.step()
            policy.project_parameters()
            metrics.append((float(policy_loss.detach()), float(value_loss.detach()),
                            float(entropy.detach()), float(gradient_norm)))
    values = np.asarray(metrics, dtype=np.float64)
    return {
        "policy_loss": float(values[:, 0].mean()),
        "value_loss": float(values[:, 1].mean()),
        "entropy": float(values[:, 2].mean()),
        "gradient_norm": float(values[:, 3].mean()),
    }
