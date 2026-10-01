"""Download pinned public body assets and compose MJCF, without running MuJoCo."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
COMMIT = 'c7affce924cb1c6add16619adf83be5c6b223e89'
BASE = f'https://raw.githubusercontent.com/NeLy-EPFL/flygym/{COMMIT}/'
DEST = ROOT / 'data' / 'fly_body'
XML_SOURCE = 'flygym/data/mjcf/neuromechfly_deepfly3d_kinorder_ryp.xml'
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
DRIVEN_SUFFIXES = ('Coxa', 'Coxa_yaw', 'Coxa_roll', 'Femur', 'Femur_roll', 'Tibia', 'Tarsus1')
LEG_SUFFIXES = ('Coxa', 'Coxa_yaw', 'Coxa_roll', 'Femur', 'Femur_roll', 'Tibia',
                'Tarsus1', 'Tarsus2', 'Tarsus3', 'Tarsus4', 'Tarsus5')
LEG_JOINTS = {'joint_' + leg + suffix for leg in LEGS for suffix in LEG_SUFFIXES}


def quaternion_product(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return [w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
            w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v]


def neutral_quaternion(joints, neutral):
    result = [1., 0., 0., 0.]
    for joint in joints:
        axis = [float(x) for x in joint.get('axis', '0 0 1').split()]
        length = math.sqrt(sum(x*x for x in axis))
        angle = neutral.get(joint.attrib['name'], 0.)
        rotation = [math.cos(angle/2)] + [math.sin(angle/2)*x/length for x in axis]
        result = quaternion_product(result, rotation)
    return result


def get_file(remote, local, offline):
    local.parent.mkdir(parents=True, exist_ok=True)
    if not local.exists():
        if offline:
            raise FileNotFoundError(f'공개 몸체 원본이 없습니다: {local}')
        print(f'몸체 자산 다운로드: {remote}', flush=True)
        request = urllib.request.Request(BASE + remote, headers={'User-Agent': 'TASS-body-assets'})
        temporary = local.with_name(local.name + '.part')
        with urllib.request.urlopen(request, timeout=90) as source, temporary.open('wb') as target:
            while block := source.read(1024 * 1024):
                target.write(block)
        temporary.replace(local)
    content = local.read_bytes()
    return dict(source=BASE + remote, path=str(local.relative_to(DEST)),
                bytes=len(content), sha256=hashlib.sha256(content).hexdigest(),
                git_blob_sha1=hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest())


def prepare(offline=False):
    upstream = DEST / 'upstream'
    inventory = DEST / 'v1_source_tree.json'
    if not inventory.exists() and not offline:
        inventory.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            f'https://api.github.com/repos/NeLy-EPFL/flygym/git/trees/{COMMIT}?recursive=1',
            headers={'User-Agent': 'TASS-body-assets'})
        with urllib.request.urlopen(request, timeout=90) as response:
            listing = json.loads(response.read())
        if listing.get('sha') != COMMIT or listing.get('truncated'):
            raise ValueError('공개 몸체 파일 목록이 고정 commit과 일치하지 않습니다.')
        inventory.write_text(json.dumps(listing), encoding='utf-8')
    files = [get_file(XML_SOURCE, upstream / 'model.xml', offline),
             get_file('LICENSE', upstream / 'LICENSE', offline),
             get_file('flygym/data/pose/pose_tripod.yaml', upstream / 'pose_tripod.yaml', offline)]
    tree = ET.parse(upstream / 'model.xml')
    root = tree.getroot()
    mesh_files = sorted({Path(m.attrib['file']).name for m in root.findall('./asset/mesh')})
    for filename in mesh_files:
        files.append(get_file('flygym/data/mesh/' + filename, upstream / 'mesh' / filename, offline))
    # If the GitHub inventory was fetched, verify every original against its pinned Git blob.
    if inventory.exists():
        listing = json.loads(inventory.read_text(encoding='utf-8-sig'))
        if listing.get('sha') != COMMIT or listing.get('truncated'):
            raise ValueError('몸체 자산 목록의 commit이 다릅니다.')
        entries = {x['path']: x['sha'] for x in listing['tree']}
        for item in files:
            remote = item['source'][len(BASE):]
            if entries.get(remote) != item['git_blob_sha1']:
                raise ValueError(f'공개 원본 해시 불일치: {remote}')
    elif offline:
        raise FileNotFoundError('원본 해시 확인을 위한 v1_source_tree.json이 필요합니다.')
    # These upstream pose values are degrees, including values for joints absent in this XML.
    degrees = {name: float(value) for name, value in re.findall(
        r'^\s*(joint_\w+)\s*:\s*([-+\deE.]+)\s*$',
        (upstream / 'pose_tripod.yaml').read_text(), re.MULTILINE)}
    neutral = {name: math.radians(value) for name, value in degrees.items()}
    compiler = root.find('compiler')
    compiler.set('fusestatic', 'false')  # Preserve body names in recordings.
    for mesh in root.findall('./asset/mesh'):
        mesh.set('file', 'upstream/mesh/' + Path(mesh.attrib['file']).name)
    root.set('model', 'TASS_public_NeuroMechFly_CNS')
    # Keep public mm / g / s units and masses, articulation axes, mesh scales and positions.
    option = root.find('option')
    option.set('integrator', 'implicit')
    world = root.find('worldbody')
    fly = world.find("body[@name='FlyBody']")
    thorax = fly.find("body[@name='Thorax']")
    fly.remove(thorax)
    world.remove(fly)
    world.append(thorax)  # Free joints must be on a top-level massive body.
    ET.SubElement(thorax, 'freejoint', name='fly_free')
    for geom in world.findall('.//geom'):
        geom.set('contype', '1')
        geom.set('conaffinity', '2')
        geom.set('friction', '1 0.005 0.0001')
        geom.set('solref', '0.002 1')
        geom.set('solimp', '0.98 0.99 0.001')
        geom.set('rgba', '0.38 0.25 0.12 1')
    original_joint_names = [j.attrib['name'] for j in world.findall('.//joint')]
    converted, ball_initial = [], {}
    # Preserve the three rotational DoFs of unactuated head/antenna bodies, but
    # replace the Euler hinge chain that becomes singular near +/-90 degree pitch.
    # Exact names matter: LFuniculus starts with LF but is not a front-leg joint.
    for body in world.findall('.//body'):
        joints = body.findall('joint')
        if len(joints) != 3 or any(j.attrib['name'] in LEG_JOINTS for j in joints):
            continue
        if any(j.get('type', 'hinge') != 'hinge' or
               any(float(x) != 0 for x in j.get('pos', '0 0 0').split()) for j in joints):
            raise ValueError(f'볼 관절 변환을 지원하지 않는 연결: {body.attrib["name"]}')
        pose_rotation = neutral_quaternion(joints, neutral)
        previous_quat = [float(x) for x in body.get('quat', '1 0 0 0').split()]
        baked_quat = quaternion_product(previous_quat, pose_rotation)
        body.set('quat', ' '.join(str(x) for x in baked_quat))
        ball_name = 'joint_' + body.attrib['name'] + '_ball'
        converted.append({'body': body.attrib['name'], 'original_hinge_names': [j.attrib['name'] for j in joints],
                          'original_axes': [j.get('axis', '0 0 1') for j in joints],
                          'original_neutral_angles_rad': [neutral.get(j.attrib['name'], 0.) for j in joints],
                          'ball_joint': ball_name, 'neutral_body_quaternion_wxyz': baked_quat,
                          'reason': 'Avoid Euler hinge singularity; keeps 3 rotational DoFs, not a biological ball-joint claim.'})
        for joint in joints:
            body.remove(joint)
        ET.SubElement(body, 'joint', name=ball_name, type='ball', pos='0 0 0')
        ball_initial[ball_name] = [1., 0., 0., 0.]
    joint_names, rest = [], {}
    for joint in world.findall('.//joint'):
        name = joint.attrib['name']
        joint_names.append(name)
        if joint.get('type', 'hinge') != 'ball':
            rest[name] = neutral.get(name, 0.)
            joint.set('springref', str(rest[name]))
        # Equal absolute stiffness is unsafe for antenna/foot joints with tiny inertia.
        # MuJoCo derives per-joint k and damping from inertia at its reference pose.
        joint.set('springdamper', '0.02 1')
        joint.attrib.pop('stiffness', None)
        joint.attrib.pop('damping', None)
        joint.set('limited', 'false')  # No fabricated joint stops; record all rotations.
    actuators = root.find('actuator')
    for element in list(actuators):
        actuators.remove(element)
    driven = ['joint_' + leg + suffix for leg in LEGS for suffix in DRIVEN_SUFFIXES]
    for name in driven:
        if name not in joint_names:
            raise ValueError(f'공개 모델에 관절이 없습니다: {name}')
        ET.SubElement(actuators, 'motor', name=name + '_CNS', joint=name, gear='1',
                      ctrllimited='false', forcelimited='false')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='50 50 0.1',
                  contype='2', conaffinity='1', friction='1 0.005 0.0001',
                  solref='0.002 1', rgba='0.65 0.68 0.72 1')
    ET.SubElement(world, 'light', name='scene_light', pos='0 0 15', dir='0 0 -1', diffuse='0.7 0.7 0.7')
    target = ET.SubElement(world, 'body', name='stimulus_target', mocap='true', pos='8 0 1.5')
    ET.SubElement(target, 'geom', name='target_visual', type='box', size='0.5 0.8 1.5',
                  contype='2', conaffinity='1', rgba='0.08 0.08 0.08 1', mass='0.001')
    head = world.find(".//body[@name='Head']")
    # One forward camera, preserving the existing L1/L2/L3 input adapter.
    ET.SubElement(head, 'camera', name='eye', pos='0.6 0 0.15', xyaxes='0 -1 0 0 0 1', fovy='90')
    root.find('visual/map').set('zfar', '100')
    ET.indent(tree, space='  ')
    xml_path = DEST / 'arena.xml'
    tree.write(xml_path, encoding='utf-8', xml_declaration=True)
    report = {
        'composition_schema': 3,
        'repository': 'https://github.com/NeLy-EPFL/flygym', 'tag': 'v1.2.1', 'commit': COMMIT,
        'license': 'Apache-2.0; upstream/LICENSE', 'files': files,
        'derived_xml_sha256': hashlib.sha256(xml_path.read_bytes()).hexdigest(),
        'initial_joint_angles_rad': rest, 'initial_ball_quaternions_wxyz': ball_initial,
        'joint_names': joint_names, 'driven_joint_names': driven,
        'original_joint_names': original_joint_names, 'ball_joint_conversions': converted,
        'rotational_degrees_of_freedom': len(rest) + 3*len(ball_initial),
        'leg_joint_count': sum(n in LEG_JOINTS for n in joint_names),
        'units': {'length': 'mm', 'mass': 'g', 'time': 's', 'force': 'uN', 'torque': 'nNm',
                  'force_N_per_model_unit': 1e-6, 'torque_Nm_per_model_unit': 1e-9},
        'changes': ['free Thorax root; removed empty fixed FlyBody wrapper', '42 direct torque actuators; no trained controller',
                    'initial pose from published static tripod pose; never replayed as gait',
                    'inertia-scaled passive springs/damping: springdamper 0.02s 1; uncalibrated',
                    'head/antenna Euler hinge triples converted to balls; 3 rotational DoFs preserved',
                    'original leg articulation preserved; no fabricated joint stops',
                    'floor and stimulus contact enabled; inter-body self collisions disabled',
                    'one forward camera and controlled target/light', 'full implicit integrator including velocity derivatives'],
        'boundary': 'Public geometry/body physics plus engineering motor adapter; not a recovered biological muscle model.',
    }
    (DEST / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'초파리 몸체 원본과 XML 준비 완료: {xml_path}; 관절 {len(joint_names)}, 구동 {len(driven)}', flush=True)
    print('MuJoCo와 신경망은 실행하지 않았습니다.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='이미 내려받은 원본만 사용')
    prepare(parser.parse_args().offline)
