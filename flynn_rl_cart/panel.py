"""Evaluation panel showing the actual photoreceptor values used by the policy."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

from curriculum import STAGE_NAMES


class ReceptorPanel:
    def __init__(self, enqueue, vision, width=780, height=390):
        self.enqueue = enqueue
        self.vision = vision
        self.width, self.height = int(width), int(height)
        self.closed = False
        self.photo = None
        self.root = tk.Tk()
        self.root.title("TASS FLYNN RL — 광수용체 입력과 카트 평가")
        self.root.geometry("840x570")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        controls = ttk.Frame(self.root, padding=8); controls.pack(fill="x")
        for text, key in (("시작 / 정지", 32), ("초기화", ord("R")),
                          ("1 정지 구체", ord("1")), ("2 좌우 이동", ord("2")),
                          ("3 자유 이동", ord("3"))):
            ttk.Button(controls, text=text, command=lambda value=key: enqueue(value)).pack(side="left", padx=3)
        labels = ttk.Frame(self.root); labels.pack(fill="x", padx=12)
        ttk.Label(labels, text="왼쪽 눈 광수용체 (L)", anchor="center").pack(side="left", expand=True, fill="x")
        ttk.Label(labels, text="오른쪽 눈 광수용체 (R)", anchor="center").pack(side="left", expand=True, fill="x")
        self.image = ttk.Label(self.root, anchor="center"); self.image.pack(fill="both", expand=True)
        self.status = tk.StringVar(value="checkpoint 대기")
        ttk.Label(self.root, textvariable=self.status, font=("Consolas", 10), padding=8,
                  wraplength=810).pack(fill="x")
        self.root.bind("<space>", lambda _event: enqueue(32))
        for key in "rR123":
            self.root.bind(f"<KeyPress-{key}>", lambda _event, k=key: enqueue(ord(k)))
        self.positions = []
        for item in vision.sides:
            origin = 20 if item["side"] == "L" else self.width//2+10
            span = self.width//2-35
            px = np.rint(origin+item["x"]*span).astype(np.int32)
            py = np.rint(20+item["y"]*(self.height-40)).astype(np.int32)
            self.positions.append((px, py, len(item["indices"])))

    @staticmethod
    def colors(values):
        level = np.clip(np.asarray(values, dtype=np.float32), 0, 1)
        red = np.clip(255*level*1.8, 0, 255)
        green = np.clip(255*np.maximum(level-0.18, 0)*1.25, 0, 255)
        blue = np.clip(60+210*level, 0, 255)
        return np.column_stack((red, green, blue)).astype(np.uint8)

    def mosaic(self, values):
        image = np.empty((self.height, self.width, 3), dtype=np.uint8); image[:] = (7, 10, 15)
        image[:, self.width//2-2:self.width//2+2] = 190
        offset = 0
        for px, py, count in self.positions:
            colors = self.colors(values[offset:offset+count]); offset += count
            for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                x = np.clip(px+dx, 0, self.width-1); y = np.clip(py+dy, 0, self.height-1)
                image[y, x] = colors
        return image

    def refresh(self, values, action, reward, info, paused, total_return):
        if self.closed:
            return
        image = self.mosaic(values)
        h, w = image.shape[:2]
        self.photo = tk.PhotoImage(data=f"P6\n{w} {h}\n255\n".encode()+image.tobytes(), format="PPM")
        self.image.configure(image=self.photo)
        self.status.set(
            f"{'PAUSED' if paused else 'RUN'} | stage={info.get('stage', 1)} "
            f"{STAGE_NAMES.get(info.get('stage', 1), '')} | distance={info.get('distance', 0):.3f}m | "
            f"receptors={float(np.min(values)):.3f}..{float(np.max(values)):.3f} | "
            f"wheel L={action[0]:+.3f} R={action[1]:+.3f} | reward={reward:+.4f} "
            f"return={total_return:+.3f} | success={info.get('success', False)}"
        )

    def pump(self):
        if not self.closed:
            self.root.update_idletasks(); self.root.update()

    def close(self):
        if not self.closed:
            self.closed = True; self.root.destroy()
