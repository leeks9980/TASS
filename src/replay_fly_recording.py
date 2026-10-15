"""GUI playback of recorded physical states; never integrates physics or steps CNS."""
import argparse
from pathlib import Path
import json
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '.deps'))
import numpy as np
import mujoco
import mujoco.viewer
from inspect_recorded_run import events


def replay(run, speed):
    metadata = json.loads((run / 'metadata.json').read_text(encoding='utf-8'))
    if metadata.get('output_mode') != 'public_fly_body':
        raise ValueError('새 full_fly_* 기록을 선택하세요.')
    log = events(run)
    if not log:
        raise ValueError('완료된 기록 프레임이 없습니다.')
    model = mujoco.MjModel.from_xml_path(str(run / 'body_assets' / 'arena.xml'))
    data = mujoco.MjData(model)
    commands, lock = [], threading.Lock()
    def enqueue(key):
        with lock:
            commands.append(key)
    frame_index, sample_index, paused = 0, 0, True
    def load_frame(index):
        with np.load(run / log[index]['snapshot']) as sample:
            return {key: sample[key] for key in ('physics_time_s', 'physics_qpos', 'physics_qvel', 'physics_ctrl',
                                                'physics_mocap_pos_mm', 'physics_mocap_quat', 'physics_light_diffuse',
                                                'physics_headlight_diffuse', 'physics_headlight_ambient')}
    sample = load_frame(0)
    print('기록 재생: Space 시작/정지, N 다음 물리 단계, P 이전 물리 단계, R 첫 기록, C 카메라', flush=True)
    print('신경망 계산과 물리 적분은 수행하지 않습니다. 저장된 위치·속도·출력만 표시합니다.', flush=True)
    root = metadata['body_root_id']
    first_person = False
    with mujoco.viewer.launch_passive(model, data, key_callback=enqueue) as viewer:
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = root
            viewer.cam.distance = 7.
            viewer.cam.azimuth = 140
            viewer.cam.elevation = -30
        while viewer.is_running():
            started = time.perf_counter()
            with lock:
                keys = commands[:]
                commands.clear()
            advance = 0 if paused else max(1, int(round(.02*speed / metadata['physics_step_seconds'])))
            for key in keys:
                if key == 32:
                    paused = not paused
                    advance = 0
                elif key == ord('N'):
                    paused, advance = True, 1
                elif key == ord('P'):
                    paused, advance = True, -1
                elif key == ord('R'):
                    frame_index, sample_index, paused, advance = 0, 0, True, 0
                    sample = load_frame(0)
                elif key == ord('C'):
                    first_person = not first_person
                    with viewer.lock():
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED if first_person else mujoco.mjtCamera.mjCAMERA_TRACKING
                        viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'eye')
                        viewer.cam.trackbodyid = root
            sample_index += advance
            while sample_index >= len(sample['physics_time_s']) and frame_index < len(log)-1:
                sample_index -= len(sample['physics_time_s'])
                frame_index += 1
                sample = load_frame(frame_index)
            while sample_index < 0 and frame_index > 0:
                frame_index -= 1
                sample = load_frame(frame_index)
                sample_index += len(sample['physics_time_s'])
            if sample_index >= len(sample['physics_time_s']):
                paused = True
            sample_index = int(np.clip(sample_index, 0, len(sample['physics_time_s'])-1))
            event = log[frame_index]
            data.qpos[:] = sample['physics_qpos'][sample_index]
            data.qvel[:] = sample['physics_qvel'][sample_index]
            data.ctrl[:] = sample['physics_ctrl'][sample_index]
            data.time = float(sample['physics_time_s'][sample_index])
            data.mocap_pos[:] = sample['physics_mocap_pos_mm'][sample_index]
            data.mocap_quat[:] = sample['physics_mocap_quat'][sample_index]
            model.light_diffuse[:] = sample['physics_light_diffuse'][sample_index]
            model.vis.headlight.diffuse[:] = sample['physics_headlight_diffuse'][sample_index]
            model.vis.headlight.ambient[:] = sample['physics_headlight_ambient'][sample_index]
            mujoco.mj_forward(model, data)
            with viewer.lock():
                viewer.user_scn.ngeom = 1
                geom = viewer.user_scn.geoms[0]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([.001]*3),
                                   data.xpos[root]+[0., 0., .5], np.eye(3).ravel(), np.array([1., 1., 1., 1.], dtype=np.float32))
                geom.label = f'REPLAY {event["case"]} episode={event["episode"]} t={data.time:.4f}s {"PAUSED" if paused else "PLAY"}'
            viewer.sync()
            time.sleep(max(0., .02-(time.perf_counter()-started)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--speed', type=float, default=1.)
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error('speed는 양수여야 합니다.')
    replay(args.run, args.speed)
