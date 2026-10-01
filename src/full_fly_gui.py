"""Korean monitoring panel for the free public fly body; no simulation logic."""
import tkinter as tk
from tkinter import ttk
import numpy as np

CASE_LABELS = {'neutral': '물체 없음', 'bright': '밝게', 'dim': '어둡게',
               'move_left': '왼쪽 이동', 'move_right': '오른쪽 이동', 'approach': '접근', 'free': '이동 물체'}
LEG_LABELS = {'LF': '왼쪽 앞', 'LM': '왼쪽 중간', 'LH': '왼쪽 뒤',
              'RF': '오른쪽 앞', 'RM': '오른쪽 중간', 'RH': '오른쪽 뒤'}
JOINT_LABELS = {'Coxa': '몸통–기절 1', 'Coxa_yaw': '몸통–기절 2', 'Coxa_roll': '몸통–기절 3',
                'Femur': '대퇴 굽힘', 'Femur_roll': '대퇴 회전', 'Tibia': '경골 굽힘', 'Tarsus1': '발목'}


class FlyPanel:
    def __init__(self, enqueue, decoder):
        self.root = tk.Tk()
        self.root.title('TASS — 전체 CNS · 공개 초파리 몸체')
        self.root.geometry('1120x870')
        self.closed, self.photo = False, None
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.status = tk.StringVar(value='일시정지 — Space 또는 시작 버튼')
        self.body = tk.StringVar(value='몸체와 발 상태: 계산 전')
        ttk.Label(self.root, textvariable=self.status, font=('Malgun Gothic', 12, 'bold')).pack(anchor='w', padx=12, pady=7)
        row = ttk.Frame(self.root)
        row.pack(fill='x', padx=10)
        def button(parent, title, key):
            b = ttk.Button(parent, text=title, command=lambda: enqueue(key))
            def space(event):
                enqueue(32)
                return 'break'
            b.bind('<KeyPress-space>', space)
            b.bind('<KeyRelease-space>', lambda e: 'break')
            b.pack(side='left', padx=3, pady=3)
        for title, key in [('시작 / 정지', 32), ('초기화', ord('R')), ('시점 전환', ord('C'))]:
            button(row, title, key)
        row = ttk.Frame(self.root)
        row.pack(fill='x', padx=10)
        for i, (case, label) in enumerate(CASE_LABELS.items(), 1):
            button(row, f'{i} {label}', ord(str(i)))
        def keypress(event):
            if event.keysym == 'space':
                enqueue(32)
                return 'break'
            key = event.char.upper()
            if key in tuple('RC1234567'):
                enqueue(ord(key))
                return 'break'
        self.root.bind('<KeyPress>', keypress)
        ttk.Label(self.root, text='자유 몸체 · 중력 · 지면 접촉 | 모델 길이 mm, 힘 µN, 토크 nN·m').pack(anchor='w', padx=12)
        ttk.Label(self.root, text=f'다리 운동신경 {decoder.report["leg_mn_count"]}개 중 출력 대응 {decoder.report["mapped_mn_count"]}개, '
                  f'미대응 {decoder.report["unmapped_mn_count"]}개 — 모든 신경 상태는 저장됩니다.').pack(anchor='w', padx=12)
        ttk.Label(self.root, text='관절 축·모멘트암은 가정입니다. 대퇴 회전은 수동 관절이며, 보행 패턴 생성기는 없습니다.').pack(anchor='w', padx=12)
        ttk.Label(self.root, textvariable=self.body, wraplength=1080).pack(anchor='w', padx=12, pady=6)
        tabs = ttk.Notebook(self.root)
        tabs.pack(fill='both', expand=True, padx=12)
        joint_tab, channel_tab = ttk.Frame(tabs), ttk.Frame(tabs)
        tabs.add(joint_tab, text='42개 구동 관절')
        tabs.add(channel_tab, text='신경 → 힘 채널')
        def table(parent, columns, titles):
            frame = ttk.Frame(parent)
            frame.pack(fill='both', expand=True)
            tree = ttk.Treeview(frame, columns=columns, show='headings', height=21)
            bar = ttk.Scrollbar(frame, orient='vertical', command=tree.yview)
            tree.configure(yscrollcommand=bar.set)
            tree.pack(side='left', fill='both', expand=True)
            bar.pack(side='right', fill='y')
            for col, title in zip(columns, titles):
                tree.heading(col, text=title)
                tree.column(col, width=160, anchor='e' if col != columns[0] else 'w')
            return tree
        self.joints = table(joint_tab, ('joint', 'angle', 'speed', 'torque', 'delta'),
                            ('관절', '각도 °', '속도 °/s', '신경 토크 nN·m', '무물체 대비 Δ각도 °'))
        self.channels = table(channel_tab, ('channel', 'count', 'raw', 'activation', 'force'),
                              ('채널', '운동신경 수', '원래 활동값', '활성화 0–1', '힘 µN'))
        for name in decoder.joint_names:
            leg, suffix = name[6:8], name[8:]
            self.joints.insert('', 'end', iid=name, values=(LEG_LABELS[leg] + ' ' + JOINT_LABELS[suffix], '—', '—', '—', '—'))
        for name, group in zip(decoder.names, decoder.groups):
            self.channels.insert('', 'end', iid=name, values=(name, len(group), '—', '—', '—'))
        bottom = ttk.Frame(self.root)
        bottom.pack(fill='x', padx=12, pady=8)
        self.camera = ttk.Label(bottom, text='카메라 입력: 계산 전')
        self.camera.pack(side='right')
        self.note = tk.StringVar(value='1번 무물체 실행 후 같은 업데이트 시점의 결과와 비교할 수 있습니다.')
        ttk.Label(bottom, textvariable=self.note, wraplength=850).pack(anchor='w')
        ttk.Label(bottom, text='몸체가 쓰러지거나 관절이 움직이지 않는 경우도 기록됩니다. 그것만으로 신경 회로의 생물학적 기능을 확정할 수 없습니다.', wraplength=850).pack(anchor='w', pady=5)

    def pump(self):
        if not self.closed:
            self.root.update_idletasks()
            self.root.update()

    def set_status(self, case, paused, count, seconds, phase=''):
        if not self.closed:
            self.status.set(f'{CASE_LABELS[case]} | {"일시정지" if paused else "진행"} | 기록 {count} | 가상 {seconds:.3f}s | {phase}')

    def refresh(self, outputs, body, feet, reference=None, rgb=None):
        if self.closed:
            return
        self.body.set(f'몸체 위치 mm: ({body["xyz_mm"][0]:+.3f}, {body["xyz_mm"][1]:+.3f}, {body["xyz_mm"][2]:+.3f}) | '
                      f'몸체 앞 방향 속도 {body["forward_velocity_mm_s"]:+.4f} mm/s | 경로 {body["path_mm"]:.4f} mm\n'
                      + '발 접촉: ' + ', '.join(f'{LEG_LABELS[k]} {"접촉" if v["ground_contact"] else "공중"}' for k, v in feet.items()))
        for i, name in enumerate(outputs['joint_names']):
            current = list(self.joints.item(name, 'values'))
            current[1:] = [f'{np.degrees(outputs["angle_rad"][i]):+.3f}',
                           f'{np.degrees(outputs["velocity_rad_s"][i]):+.3f}', f'{outputs["torque_nNm"][i]:+.5f}',
                           f'{np.degrees(outputs["angle_rad"][i]-reference["angle_rad"][i]):+.3f}' if reference else '기준 없음']
            self.joints.item(name, values=current)
        for i, name in enumerate(outputs['channel_names']):
            current = list(self.channels.item(name, 'values'))
            current[2:] = [f'{outputs["raw_activity"][i]:.6e}', f'{outputs["activation"][i]:.5f}', f'{outputs["force_uN"][i]:.5f}']
            self.channels.item(name, values=current)
        if rgb is not None:
            height, width = rgb.shape[:2]
            self.photo = tk.PhotoImage(data=f'P6\n{width} {height}\n255\n'.encode() + rgb.tobytes(), format='PPM')
            self.camera.configure(image=self.photo, text='')
        self.note.set('무물체의 같은 신경 업데이트 시점과 비교 중입니다. 비교값은 제어에 사용하지 않습니다.' if reference else
                      '같은 시점의 무물체 기준이 없습니다. 1번 실행 후 다른 조건을 선택하고 Space로 시작하세요.')

    def close(self):
        if not self.closed:
            self.closed = True
            self.root.destroy()
