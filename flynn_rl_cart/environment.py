"""MuJoCo differential-drive curriculum environment for light-target pursuit."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from vision import PhotoreceptorInput


def wrap_angle(value):
    return (value+math.pi) % (2.0*math.pi) - math.pi


class LightTargetCartEnv:
    def __init__(self, model_path, circuit_folder, vision_config, env_config,
                 reward_config, curriculum_config, seed=0):
        self.model = mujoco.MjModel.from_xml_path(str(Path(model_path)))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(
            self.model, height=int(vision_config["eye_height"]),
            width=int(vision_config["eye_width"]),
        )
        self.vision = PhotoreceptorInput(
            circuit_folder, retention=float(vision_config["visual_retention"])
        )
        self.config = env_config
        self.reward_config = reward_config
        self.curriculum_config = curriculum_config
        self.rng = np.random.default_rng(seed)
        self.physics_steps = int(env_config["physics_steps_per_action"])
        self.dt = float(self.model.opt.timestep*self.physics_steps)
        self.max_wheel_speed = float(env_config["max_wheel_speed_m_s"])
        self.wheel_radius = float(env_config["wheel_radius_m"])
        self.cart_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cart")
        self.wall_geoms = {
            index for index in range(self.model.ngeom)
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, index) or "").startswith("wall_")
        }
        self.stage = 1
        self.target_position = np.zeros(3)
        self.target_velocity = np.zeros(2)
        self.horizontal_origin = np.zeros(2)
        self.horizontal_direction = 1.0
        self.step_count = 0
        self.idle_count = 0
        self.previous_distance = 0.0
        self.previous_xy = np.zeros(2)
        self.episode_return = 0.0

    def _sample_target(self):
        low, high = self.config["initial_target_distance_m"]
        angle_low, angle_high = self.config["initial_target_angle_degrees"]
        distance = self.rng.uniform(low, high)
        angle = math.radians(self.rng.uniform(angle_low, angle_high))
        heading = float(self.data.qpos[2])
        world_angle = heading + angle
        origin = self.data.qpos[:2]
        target = origin + distance*np.array([math.cos(world_angle), math.sin(world_angle)])
        half = float(self.config["arena_half_size_m"])-0.4
        target = np.clip(target, -half, half)
        return np.array([target[0], target[1], float(self.config["target_height_m"])])

    def reset(self, stage=1):
        mujoco.mj_resetData(self.model, self.data)
        self.stage = int(stage)
        self.data.qpos[0:2] = self.rng.uniform(-0.45, 0.45, size=2)
        self.data.qpos[2] = self.rng.uniform(-math.pi, math.pi)
        self.target_position = self._sample_target()
        self.data.mocap_pos[0] = self.target_position
        self.horizontal_origin = self.target_position[:2].copy()
        self.horizontal_direction = 1.0 if self.rng.random() < 0.5 else -1.0
        speed_range = (self.curriculum_config["stage2_target_speed_m_s"]
                       if self.stage == 2 else self.curriculum_config["stage3_target_speed_m_s"])
        speed = self.rng.uniform(*speed_range)
        direction = self.rng.uniform(-math.pi, math.pi)
        self.target_velocity = speed*np.array([math.cos(direction), math.sin(direction)])
        self.step_count = self.idle_count = 0
        self.episode_return = 0.0
        self.vision.reset()
        self.previous_xy = self.data.qpos[:2].copy()
        self.previous_distance = self.distance_to_target()
        mujoco.mj_forward(self.model, self.data)
        receptors = self.render_receptors()
        return receptors, self.privileged_observation()

    def _move_target(self):
        half = float(self.config["arena_half_size_m"])-0.35
        if self.stage == 2:
            speed = float(np.linalg.norm(self.target_velocity))
            self.target_position[1] += self.horizontal_direction*speed*self.dt
            if abs(self.target_position[1]-self.horizontal_origin[1]) > 1.4:
                self.horizontal_direction *= -1.0
                self.target_position[1] = np.clip(self.target_position[1], -half, half)
        elif self.stage == 3:
            jitter = self.rng.normal(0.0, 0.18, size=2)*self.dt
            self.target_velocity += jitter
            low, high = self.curriculum_config["stage3_target_speed_m_s"]
            speed = float(np.linalg.norm(self.target_velocity))
            if speed > 0:
                self.target_velocity *= np.clip(speed, low, high)/speed
            self.target_position[:2] += self.target_velocity*self.dt
            for axis in (0, 1):
                if abs(self.target_position[axis]) > half:
                    self.target_position[axis] = np.clip(self.target_position[axis], -half, half)
                    self.target_velocity[axis] *= -1.0
        self.data.mocap_pos[0] = self.target_position

    def render_receptors(self):
        self.renderer.update_scene(self.data, camera="left_eye")
        left = self.renderer.render().copy()
        self.renderer.update_scene(self.data, camera="right_eye")
        right = self.renderer.render().copy()
        return self.vision.sample(left, right)

    def distance_to_target(self):
        return float(np.linalg.norm(self.target_position[:2]-self.data.qpos[:2]))

    def privileged_observation(self):
        difference = self.target_position[:2]-self.data.qpos[:2]
        heading = float(self.data.qpos[2])
        rotation = np.array([[math.cos(heading), math.sin(heading)],
                             [-math.sin(heading), math.cos(heading)]])
        relative = rotation @ difference
        return np.asarray([
            relative[0], relative[1], np.linalg.norm(difference),
            self.data.qvel[0], self.data.qvel[1], self.data.qvel[2],
            self.stage/3.0,
        ], dtype=np.float32)

    def _wall_contact(self):
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if int(contact.geom1) in self.wall_geoms or int(contact.geom2) in self.wall_geoms:
                return True
        return False

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        wheel_speed = action*self.max_wheel_speed
        self.data.ctrl[0] = wheel_speed[0]/self.wheel_radius
        self.data.ctrl[1] = wheel_speed[1]/self.wheel_radius
        self._move_target()
        for _ in range(self.physics_steps):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        distance = self.distance_to_target()
        progress = self.previous_distance-distance
        actual_speed = float(np.linalg.norm(self.data.qpos[:2]-self.previous_xy)/max(self.dt, 1.0e-6))
        self.idle_count = self.idle_count+1 if actual_speed < self.reward_config["idle_speed_threshold_m_s"] else 0
        reached = distance <= float(self.config["reach_distance_m"])
        wall_contact = self._wall_contact()
        reward = float(self.reward_config["progress_scale"])*progress
        reward -= float(self.reward_config["step_penalty"])
        reward -= float(self.reward_config["energy_penalty_scale"])*float(np.square(action).sum())
        reward -= float(self.reward_config["spin_penalty_scale"])*abs(float(self.data.qvel[2]))
        if progress < 0:
            reward += float(self.reward_config["lost_distance_penalty_scale"])*progress
        if self.idle_count >= int(self.reward_config["idle_after_steps"]):
            reward -= float(self.reward_config["idle_penalty"])
        if wall_contact:
            reward -= float(self.reward_config["wall_collision_penalty"])
        if reached:
            reward += float(self.reward_config["reach_bonus"])

        half = float(self.config["arena_half_size_m"])
        out_of_bounds = bool(np.any(np.abs(self.data.qpos[:2]) > half))
        timeout = self.step_count >= int(self.config["max_episode_steps"])
        done = reached or out_of_bounds or timeout
        self.previous_distance = distance
        self.previous_xy = self.data.qpos[:2].copy()
        self.episode_return += reward
        receptors = self.render_receptors()
        info = {
            "success": reached,
            "distance": distance,
            "progress": progress,
            "wall_contact": wall_contact,
            "timeout": timeout,
            "stage": self.stage,
            "episode_return": self.episode_return,
            "actual_speed": actual_speed,
        }
        return receptors, self.privileged_observation(), reward, done, info

    def close(self):
        self.renderer.close()
