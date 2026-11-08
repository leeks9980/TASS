"""Display photoreceptor activations without exposing raw eye-camera RGB."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np


class CompoundEyePanel:
    WIDTH = 780
    HEIGHT = 390

    def __init__(self, enqueue, input_mapping, motor_report, target_step):
        self.enqueue = enqueue
        self.mapping = input_mapping
        self.motor_report = motor_report
        self.closed = False
        self.photo = None
        self.root = tk.Tk()
        self.root.title("TASS — 초파리 광수용체 시야")
        self.root.geometry("820x540")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        controls = ttk.Frame(self.root, padding=8)
        controls.pack(fill="x")
        ttk.Button(controls, text="시작 / 일시정지", command=lambda: enqueue(32)).pack(side="left")
        ttk.Button(controls, text="초기화", command=lambda: enqueue(ord("R"))).pack(side="left", padx=6)
        ttk.Label(
            controls,
            text="MuJoCo: 카트 외부 관찰  |  아래: 카메라 원본이 아닌 광수용체 활성",
        ).pack(side="left", padx=12)

        target = ttk.LabelFrame(self.root, text="빛 구체 직접 이동", padding=6)
        target.pack(fill="x", padx=8, pady=(0, 4))
        commands = [
            ("앞 W", (target_step, 0.0, 0.0)),
            ("뒤 S", (-target_step, 0.0, 0.0)),
            ("왼쪽 A", (0.0, target_step, 0.0)),
            ("오른쪽 D", (0.0, -target_step, 0.0)),
            ("위 Q", (0.0, 0.0, target_step)),
            ("아래 E", (0.0, 0.0, -target_step)),
        ]
        for text, delta in commands:
            ttk.Button(
                target, text=text,
                command=lambda d=delta: enqueue(("target", *d)),
            ).pack(side="left", padx=3)

        self.image = ttk.Label(self.root, anchor="center")
        self.image.pack(fill="both", expand=True, padx=10, pady=5)
        self.status = tk.StringVar(value="일시정지 — Space 또는 시작 버튼")
        ttk.Label(self.root, textvariable=self.status, font=("Consolas", 10), padding=8).pack(fill="x")
        ttk.Label(
            self.root,
            text=("각 점은 MaleCNS 광수용체 한 개의 실제 입력값입니다. hex 위상은 보존하지만 "
                  "복안 시야각·회전·분광 감도는 아직 생리학적으로 보정되지 않았습니다."),
            wraplength=790,
            padding=(10, 0, 10, 10),
        ).pack(fill="x")
        self.root.bind("<space>", lambda _event: enqueue(32))
        self.root.bind("<KeyPress-r>", lambda _event: enqueue(ord("R")))
        self.root.bind("<KeyPress-R>", lambda _event: enqueue(ord("R")))
        for key in "wWsSaAdDqQeE":
            self.root.bind(f"<KeyPress-{key}>", lambda _event, k=key: enqueue(ord(k)))

        self._positions = []
        for side, _indices, x, y in self.mapping:
            origin_x = 25 if side == "L" else 405
            px = np.rint(origin_x + np.asarray(x) * 350).astype(np.int32)
            py = np.rint(55 + np.asarray(y) * 285).astype(np.int32)
            self._positions.append((side, px, py))
        count = sum(len(item[1]) for item in self.mapping)
        self.refresh(np.zeros(count, dtype=np.float32), {"L": 0.0, "R": 0.0},
                     {"L": 0.0, "R": 0.0}, True, 0.0,
                     np.array([1.8, 0.0, 0.25]), "not loaded")

    @staticmethod
    def _colors(values):
        level = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
        red = np.clip(255.0 * level * 1.8, 0, 255)
        green = np.clip(255.0 * np.maximum(level - 0.18, 0.0) * 1.25, 0, 255)
        blue = np.clip(90.0 + 210.0 * level, 0, 255)
        return np.column_stack((red, green, blue)).astype(np.uint8)

    def _render_mosaic(self, values):
        expected = sum(len(item[1]) for item in self.mapping)
        supplied = np.asarray(values, dtype=np.float32).reshape(-1)
        if len(supplied) != expected:
            values = np.zeros(expected, dtype=np.float32)
            values[:min(expected, len(supplied))] = supplied[:expected]
        else:
            values = supplied
        canvas = np.empty((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        canvas[:] = (8, 12, 18)
        canvas[35:355, 10:390] = (13, 19, 27)
        canvas[35:355, 390:770] = (13, 19, 27)
        offset = 0
        for (_side, px, py), (_name, indices, _x, _y) in zip(self._positions, self.mapping):
            count = len(indices)
            colors = self._colors(values[offset:offset + count])
            offset += count
            for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                xx = np.clip(px + dx, 0, self.WIDTH - 1)
                yy = np.clip(py + dy, 0, self.HEIGHT - 1)
                canvas[yy, xx] = colors
        canvas[30:360, 388:392] = 210
        return canvas

    def refresh(self, receptor_values, scores, wheel_speeds, paused, sim_time,
                target_position, control_mode):
        if self.closed:
            return
        image = self._render_mosaic(receptor_values)
        height, width = image.shape[:2]
        self.photo = tk.PhotoImage(
            data=f"P6\n{width} {height}\n255\n".encode() + image.tobytes(), format="PPM"
        )
        self.image.configure(image=self.photo)
        minimum = float(np.min(receptor_values)) if len(receptor_values) else 0.0
        maximum = float(np.max(receptor_values)) if len(receptor_values) else 0.0
        self.status.set(
            f"{'PAUSED' if paused else 'RUN'} [{control_mode}] t={sim_time:7.2f}s  "
            f"광수용체 L={len(self.mapping[0][1]):,} R={len(self.mapping[1][1]):,} "
            f"입력={minimum:.4f}..{maximum:.4f}  "
            f"운동평균 L={scores['L']:+.8f} R={scores['R']:+.8f}  "
            f"바퀴 L={wheel_speeds['L']:+.4f} R={wheel_speeds['R']:+.4f}  "
            f"빛=({target_position[0]:+.2f}, {target_position[1]:+.2f}, {target_position[2]:+.2f})"
        )

    def pump(self):
        if not self.closed:
            self.root.update_idletasks()
            self.root.update()

    def close(self):
        if not self.closed:
            self.closed = True
            self.root.destroy()
