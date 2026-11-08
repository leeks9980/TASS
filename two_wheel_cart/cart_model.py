"""Differential-drive cart and synthetic camera used by the TASS cart demo."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class WheelCommand:
    left_signal: float
    right_signal: float
    left_speed: float
    right_speed: float


class SignalToWheels:
    """Map the two leg-motor population means directly to wheel speeds."""

    def __init__(self, motor_gain: float, max_wheel_speed: float):
        if motor_gain < 0 or max_wheel_speed <= 0:
            raise ValueError("motor_gain must be non-negative and max_wheel_speed positive")
        self.motor_gain = float(motor_gain)
        self.max_wheel_speed = float(max_wheel_speed)

    def command(self, scores: dict[str, float]) -> WheelCommand:
        # FullCNS reports the signed mean for every annotated leg motor neuron
        # on each anatomical side.  Preserve the sign all the way to the wheel:
        # positive is forward and negative is reverse/braking motion.
        left = float(scores.get("L", 0.0))
        right = float(scores.get("R", 0.0))
        return WheelCommand(
            left_signal=left,
            right_signal=right,
            left_speed=clamp(left * self.motor_gain, -self.max_wheel_speed, self.max_wheel_speed),
            right_speed=clamp(right * self.motor_gain, -self.max_wheel_speed, self.max_wheel_speed),
        )


@dataclass
class DifferentialDriveCart:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    wheel_base: float = 0.34

    def step(self, left_speed: float, right_speed: float, dt: float) -> None:
        if dt <= 0:
            return
        linear = 0.5 * (left_speed + right_speed)
        angular = (right_speed - left_speed) / self.wheel_base
        if abs(angular) < 1e-9:
            self.x += linear * math.cos(self.heading) * dt
            self.y += linear * math.sin(self.heading) * dt
        else:
            next_heading = self.heading + angular * dt
            radius = linear / angular
            self.x += radius * (math.sin(next_heading) - math.sin(self.heading))
            self.y -= radius * (math.cos(next_heading) - math.cos(self.heading))
            self.heading = wrap_angle(next_heading)


def render_target_camera(
    cart: DifferentialDriveCart,
    target: tuple[float, float],
    width: int = 128,
    height: int = 64,
    field_of_view_degrees: float = 120.0,
    background: int = 8,
) -> np.ndarray:
    """Render a small RGB image with one bright target in the cart's view.

    Positive relative bearing is to the cart's left, so it is rendered in the
    left half of the camera image.  The FullCNS input adapter then samples that
    half with the left-eye photoreceptor map.
    """

    image = np.full((height, width, 3), background, dtype=np.uint8)
    dx, dy = target[0] - cart.x, target[1] - cart.y
    distance = max(math.hypot(dx, dy), 1e-6)
    bearing = wrap_angle(math.atan2(dy, dx) - cart.heading)
    half_fov = math.radians(field_of_view_degrees) * 0.5
    if abs(bearing) > half_fov:
        return image

    center_x = (0.5 - 0.5 * bearing / half_fov) * (width - 1)
    center_y = 0.5 * (height - 1)
    sigma = clamp(16.0 / math.sqrt(distance + 0.15), 2.0, 18.0)
    yy, xx = np.mgrid[0:height, 0:width]
    spot = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma**2))
    intensity = 247.0 * clamp(2.5 / distance, 0.2, 1.0)
    value = np.clip(background + spot * intensity, 0, 255).astype(np.uint8)
    image[..., 0] = value
    image[..., 1] = value
    image[..., 2] = value
    return image
