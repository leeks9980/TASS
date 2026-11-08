"""Monitoring panel for annotation-derived front-leg function groups."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np


FUNCTION_LABELS = {
    "promotor": "앞으로",
    "remotor": "뒤로",
    "abductor": "벌림",
    "adductor": "모음",
    "levator": "올림",
    "depressor": "내림",
    "flexor": "굽힘",
    "extensor": "폄",
}


class FrontLegPanel:
    def __init__(self, enqueue, report, target_step):
        self.enqueue = enqueue
        self.closed = False
        self.root = tk.Tk()
        self.root.title("TASS — 주석 기반 앞다리 운동뉴런 MuJoCo 테스트")
        self.root.geometry("940x670")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        controls = ttk.Frame(self.root, padding=8)
        controls.pack(fill="x")
        for text, command in (
            ("시작 / 일시정지", 32),
            ("초기화", ord("R")),
            ("현재값을 기준선으로", ord("B")),
        ):
            ttk.Button(controls, text=text, command=lambda value=command: enqueue(value)).pack(side="left", padx=3)

        target = ttk.LabelFrame(self.root, text="빛 구체", padding=6)
        target.pack(fill="x", padx=8)
        for text, delta in (
            ("앞 W", (target_step, 0.0, 0.0)),
            ("뒤 S", (-target_step, 0.0, 0.0)),
            ("왼쪽 A", (0.0, target_step, 0.0)),
            ("오른쪽 D", (0.0, -target_step, 0.0)),
            ("위 Q", (0.0, 0.0, target_step)),
            ("아래 E", (0.0, 0.0, -target_step)),
        ):
            ttk.Button(target, text=text,
                       command=lambda d=delta: enqueue(("target", *d))).pack(side="left", padx=3)

        self.status = tk.StringVar(value="일시정지")
        ttk.Label(self.root, textvariable=self.status, font=("Malgun Gothic", 11, "bold"),
                  padding=8).pack(fill="x")

        columns = ("side", "function", "count", "value", "baseline", "delta")
        self.groups = ttk.Treeview(self.root, columns=columns, show="headings", height=16)
        for column, title, width in zip(
            columns,
            ("좌우", "주석 기능", "뉴런 수", "부호 포함 평균", "기준값", "변화량"),
            (80, 150, 90, 170, 170, 170),
        ):
            self.groups.heading(column, text=title)
            self.groups.column(column, width=width, anchor="center")
        self.groups.pack(fill="both", expand=True, padx=8, pady=5)
        for side in ("L", "R"):
            for function in FUNCTION_LABELS:
                key = f"{side}_{function}"
                count = report["group_counts"][key]
                self.groups.insert("", "end", iid=key,
                                   values=(side, FUNCTION_LABELS[function], count, "—", "—", "—"))

        self.joints = tk.StringVar(value="관절 명령: 계산 전")
        ttk.Label(self.root, textvariable=self.joints, padding=8, wraplength=900).pack(fill="x")
        ttk.Label(
            self.root,
            text=(f"앞다리 운동뉴런 {report['front_motor_rows']}개 중 이름으로 기능을 분류하지 못한 "
                  f"행 {report['unclassified_front_motor_rows']}개. 여러 기능이 함께 적힌 뉴런은 "
                  "복수 그룹에 포함됩니다. 이 리그의 관절 대응은 검증된 근육역학이 아닙니다."),
            wraplength=900,
            padding=8,
        ).pack(fill="x")
        self.root.bind("<space>", lambda _event: enqueue(32))
        for key in "rRbBwWsSaAdDqQeE":
            self.root.bind(f"<KeyPress-{key}>", lambda _event, k=key: enqueue(ord(k)))

    def refresh(self, rows, baseline, targets, paused, sim_time, target_position):
        if self.closed:
            return
        for row in rows:
            key = f"{row.side}_{row.function}"
            base = baseline[(row.side, row.function)]
            self.groups.item(key, values=(
                row.side, FUNCTION_LABELS[row.function], row.count,
                f"{row.mean:+.8e}", f"{base:+.8e}", f"{row.mean-base:+.8e}",
            ))
        self.status.set(
            f"{'PAUSED' if paused else 'RUN'} | t={sim_time:.2f}s | "
            f"빛=({target_position[0]:+.2f}, {target_position[1]:+.2f}, {target_position[2]:+.2f})"
        )
        self.joints.set("관절 명령 rad | " + "  ".join(
            f"{name}={value:+.4f}" for name, value in targets.items()
        ))

    def pump(self):
        if not self.closed:
            self.root.update_idletasks()
            self.root.update()

    def close(self):
        if not self.closed:
            self.closed = True
            self.root.destroy()
