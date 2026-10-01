"""Live display of recorded neural readout, antagonist forces and joint response."""
import tkinter as tk
from tkinter import ttk
import numpy as np
from full_cns_muscles import LEGS, LEG_LABELS


CASE_LABELS = {'neutral': '물체 없음', 'bright': '밝기 증가', 'dim': '밝기 감소',
               'move_left': '왼쪽 이동', 'move_right': '오른쪽 이동', 'approach': '정면 접근', 'free': '자유 관찰'}


class MusclePanel:
    def __init__(self, enqueue, decoder):
        self.root = tk.Tk()
        self.root.title('TASS — 신경 · 가상 근육 · 관절')
        self.root.geometry('1050x850')
        self.closed = False
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.status = tk.StringVar(value='일시정지 — Space로 시작')
        self.reference = tk.StringVar(value='무물체 기준 없음: 1번 조건을 먼저 실행하면 같은 신경 업데이트 시점과 비교합니다.')
        self.steering = tk.StringVar(value='하행신경: 아직 계산 전')
        ttk.Label(self.root, textvariable=self.status, font=('Malgun Gothic', 12, 'bold')).pack(anchor='w', padx=12, pady=8)
        controls = ttk.Frame(self.root)
        controls.pack(fill='x', padx=12)
        def make_button(parent, title, key):
            button = ttk.Button(parent, text=title, command=lambda: enqueue(key))
            # Prevent the focused button's default Space action from also firing.
            def space(event):
                enqueue(32)
                return 'break'
            button.bind('<KeyPress-space>', space)
            button.bind('<KeyRelease-space>', lambda event: 'break')
            button.pack(side='left', padx=3)
        for title, key in [('시작 / 정지', 32), ('초기화', ord('R')), ('카메라 전환', ord('C'))]:
            make_button(controls, title, key)
        conditions = ttk.Frame(self.root)
        conditions.pack(fill='x', padx=12, pady=8)
        for i, (case, title) in enumerate(CASE_LABELS.items(), 1):
            make_button(conditions, f'{i} {title}', ord(str(i)))
        def keypress(event):
            if event.keysym == 'space':
                enqueue(32)
                return 'break'
            key = event.char.upper()
            if key in ('R', 'C', '1', '2', '3', '4', '5', '6', '7'):
                enqueue(ord(key))
                return 'break'
        self.root.bind('<KeyPress>', keypress)
        ttk.Label(self.root, text='고정 몸체 · 여섯 경골 관절 · 중력 없음 | 양의 토크: 굽힘 | 관절 반응 관찰용').pack(anchor='w', padx=12)
        ttk.Label(self.root, text=f'고정 활동 기준 {decoder.activity_scale:.3e} | 최대 힘 {decoder.force_max:g} N | 모멘트암 {decoder.moment_arm:g} m').pack(anchor='w', padx=12)
        ttk.Label(self.root, textvariable=self.reference, wraplength=1000).pack(anchor='w', padx=12, pady=5)
        grid = ttk.Frame(self.root)
        grid.pack(fill='both', expand=True, padx=12)
        self.cells = {}
        for i, leg in enumerate(LEGS):
            frame = ttk.LabelFrame(grid, text=LEG_LABELS[leg])
            frame.grid(row=i//3, column=i%3, sticky='nsew', padx=4, pady=4)
            grid.columnconfigure(i%3, weight=1)
            grid.rowconfigure(i//3, weight=1)
            values = [tk.StringVar(value='계산 전') for _ in range(4)]
            bars = []
            for j, function in enumerate(('flexor', 'extensor')):
                count = len(decoder.groups[leg][function])
                title = '굽힘' if j == 0 else '폄'
                ttk.Label(frame, text=f'{title} 운동신경 {count}개' + (' — 주석 없음' if count == 0 else '')).pack(anchor='w', padx=8)
                ttk.Label(frame, textvariable=values[j]).pack(anchor='w', padx=8)
                bar = ttk.Progressbar(frame, maximum=1., length=265)
                bar.pack(fill='x', padx=8, pady=3)
                bars.append(bar)
            ttk.Label(frame, textvariable=values[2]).pack(anchor='w', padx=8, pady=4)
            ttk.Label(frame, textvariable=values[3], wraplength=300).pack(anchor='w', padx=8, pady=4)
            self.cells[leg] = values, bars
        bottom = ttk.Frame(self.root)
        bottom.pack(fill='x', padx=12, pady=8)
        self.camera = ttk.Label(bottom, text='카메라 입력: 계산 전')
        self.camera.pack(side='right')
        ttk.Label(bottom, textvariable=self.steering, wraplength=700).pack(anchor='w')
        ttk.Label(bottom, text='활동값은 무차원 모델 값입니다. 기준 비교와 하행신경 표시는 모터 제어에 사용하지 않습니다.', wraplength=700).pack(anchor='w', pady=4)
        self.photo = None

    def pump(self):
        if not self.closed:
            try:
                self.root.update_idletasks()
                self.root.update()
            except tk.TclError:
                if not self.closed:
                    raise

    def set_status(self, case, paused, frame, seconds, phase=''):
        if not self.closed:
            self.status.set(f'{CASE_LABELS[case]} | {"일시정지" if paused else "진행"} | 기록 {frame} | 가상 {seconds:.2f}s | {phase}')

    def refresh(self, legs, steering, reference, reference_episode, rgb=None):
        if self.closed:
            return
        self.reference.set(f'무물체 에피소드 {reference_episode}의 같은 신경 업데이트 시점과 비교' if reference is not None else '같은 시점의 무물체 기준 기록이 없습니다.')
        for leg, item in legs.items():
            values, bars = self.cells[leg]
            for j, function in enumerate(('flexor', 'extensor')):
                values[j].set(f'활동 {item[function+"_activity"]:.3e} | 힘 {item[function+"_force_N"]:.3f} N')
                bars[j]['value'] = item[function+'_activation']
            values[2].set(f'토크 {item["torque_Nm"]:+.4f} Nm | 각도 {np.degrees(item["angle_rad"]):+.2f}°')
            if reference is None:
                values[3].set('기준 대비 변화: 기록 없음')
            else:
                diff = reference[leg]
                values[3].set(f'기준 대비 Δ굽힘 {diff["flexor_activity"]:+.2e}, Δ폄 {diff["extensor_activity"]:+.2e}\nΔ토크 {diff["torque_Nm"]:+.4f} Nm, Δ각도 {np.degrees(diff["angle_rad"]):+.2f}°')
        parts = []
        for name, values in steering.items():
            def fmt(value):
                return '--' if value is None else f'{value:.3e}'
            parts.append(f'{name}: L {fmt(values["L"])}, R {fmt(values["R"])}, R−L {fmt(values["R_minus_L"])}')
        self.steering.set('하행신경 관찰(세포체 쪽 기준)\n' + '\n'.join(parts))
        if rgb is not None:
            h, w = rgb.shape[:2]
            ppm = f'P6\n{w} {h}\n255\n'.encode('ascii') + np.asarray(rgb, dtype=np.uint8).tobytes()
            self.photo = tk.PhotoImage(data=ppm, format='PPM')
            self.camera.configure(image=self.photo, text='')

    def close(self):
        if not self.closed:
            self.closed = True
            self.root.destroy()
