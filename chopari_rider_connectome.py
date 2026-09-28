# -*- coding: utf-8 -*-
# chopari_rider_connectome.py
#
# 초파리라이더 - 실제 Drosophila connectome 기반 제어 버전
#
# 입력 : KartRider 화면
#   -> 도로/벽/정지 특징
#   -> 실제 connectome에 존재하는 sensory/visual 뉴런에 주입
#   -> 실제 connectome 연결(syn_count)을 따라 sparse LIF 전파
#   -> 실제 descending neuron을 운동 출력으로 사용
#   -> vJoy
#
# 중요:
#   neurons.csv / connections_princeton.csv는 '연결 구조'를 제공한다.
#   CSV만으로 생물학적으로 완전한 초파리 뇌를 재현할 수 있는 것은 아니므로,
#   막전위/시간상수/보상학습(STDP)은 계산 모델로 추가한다.
#   그래프 자체의 neuron ID와 연결은 CSV에서 그대로 가져온다.
#
# 최초 실행 시 9M개 연결을 여러 번 읽어 실제 connectome subgraph cache를 만든다.
# 이후 실행은 cache를 사용한다.
#
# 실행:
#   관리자 권한 터미널에서
#       python chopari_rider_connectome.py
#
# 영역 재선택:
#       python chopari_rider_connectome.py --select
#
# 디버그/학습:
#   F8 = 주행 ON/OFF
#   F9 = 즉시 정지
#   F10 = 현재 connectome cache 재생성
#   F11 = 보상/학습 통계 출력
#   q = 종료
#   s = 스크린샷
#   r = 화면 녹화 ON/OFF
#   c = 게임 영역 재선택

import os
import sys
import json
import time
import ctypes
import threading
from collections import defaultdict, Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pandas as pd
import mss

try:
    from scipy.sparse import csr_matrix
except ImportError:
    csr_matrix = None

try:
    import pyvjoy
except ImportError:
    pyvjoy = None

# Wall AI: 실제 게임 프레임에서 학습된 벽/도로 분류기를 connectome 감각 입력 앞단에 연결
try:
    from wall_ai import WallAI
except Exception as e:
    WallAI = None
    print(f"[WallAI] 모듈 로드 실패: {e}")

# ============================================================
# 경로 / 기본 설정
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NEURON_CSV = os.path.join(BASE_DIR, "neurons.csv")
CONNECTION_CSV = os.path.join(BASE_DIR, "connections_princeton.csv")
CONFIG_PATH = os.path.join(BASE_DIR, "chopari_config.json")
LEARN_PATH = os.path.join(BASE_DIR, "chopari_connectome_learning.json")
CACHE_PATH = os.path.join(BASE_DIR, "chopari_connectome_cache.npz")
SAVE_DIR = os.path.join(BASE_DIR, "captures")

MONITOR_INDEX = 1
PREVIEW_WIDTH = 760
PROC_WIDTH = 320
RECORD_FPS = 10

# 최초 connectome 추출 설정.
# 실제 뉴런/연결을 사용하되, 166k 뉴런/9m edge 전체를 매 5ms Python에서 돌리는 대신
# 실제 그래프에서 모터 출력까지 연결되는 부분을 sparse subgraph로 추출한다.
CACHE_VERSION = 3
MOTOR_DESCENDING = 64
UPSTREAM_PER_TARGET = 32
MAX_UPSTREAM_TOTAL = 14000
MAX_EDGES = 450000
EXPANSION_HOPS = 3

# 실시간 뇌 계산
BRAIN_DT = 0.010       # 10 ms
MAX_STEPS_PER_UPDATE = 8
TAU_M = 0.025
TAU_SYN = 0.012
TAU_RATE = 0.12
V_THRESHOLD = 1.0
V_RESET = 0.0
REFRACTORY_STEPS = 1

# 시냅스 정규화
SYN_GAIN = 0.075
INHIBITORY_SCALE = 1.0
EXCITATORY_SCALE = 1.0

# 입력 채널
N_COL = 8
N_INPUT = 11            # road 8 + stall + wall L/R

# 출력
STEER_SMOOTH = 0.18
STEER_DEADZONE = 0.08
FULL_TURN = 0.62
STEER_INVERT = True
STEER_MODE = "pwm"
PWM_PERIOD = 0.12

VJOY_DEVICE_ID = 1
AXIS_MIN = 0x0001
AXIS_CENTER = 0x4000
AXIS_MAX = 0x8000
FORWARD_VALUE = AXIS_MIN
REVERSE_VALUE = AXIS_MAX if FORWARD_VALUE == AXIS_MIN else AXIS_MIN

# 화면 영역
LOOKAHEAD_BAND = (0.0, 0.33, 1.0, 0.46)
MINIMAP_RECT = (0.82, 0.36, 1.0, 0.88)
HUD_RECT = (0.78, 0.0, 1.0, 0.30)
SPEEDO_RECT = (0.38, 0.85, 0.62, 1.0)
LEFT_SIDE_RECT = (0.10, 0.55, 0.40, 0.85)
RIGHT_SIDE_RECT = (0.60, 0.55, 0.80, 0.85)
STUCK_INTERVAL = 0.25
STUCK_DIFF_THRESHOLD = 1.5
STALL_FULL_SEC = 1.0
STUCK_GRACE = 3.0
RECOVERY_GRACE = 1.5
WALL_BASE = 0.4

# 보상. 학습은 connectome의 선택된 실제 edge에만 적용된다.
REWARD_NORMAL = 0.015
REWARD_STEADY = 0.025
REWARD_CORNER = 0.08
REWARD_DRIFT = 0.25
REWARD_LAP = 1.5
REWARD_RECORD = 4.0
PENALTY_WALL = -0.45
PENALTY_COLLISION = -1.0
PENALTY_STUCK = -0.35
PENALTY_RESET = -2.5

LEARNING_RATE = 0.0025
ELIGIBILITY_DECAY = 0.96
MAX_PLASTIC_EDGES = 120000
REWARD_CLIP = 2.5

# 뇌 상태 API
API_HOST = "127.0.0.1"
API_PORT = 8765
API_VERSION = 1

# Wall AI는 매 프레임마다 비싼 patch 특징 계산을 하지 않고 이 주기로 갱신한다.
# 마지막 결과는 connectome이 다음 갱신까지 유지한다.
WALL_AI_INTERVAL = 0.10
WALL_AI_LEFT_GAIN = 1.00
WALL_AI_RIGHT_GAIN = 1.00
WALL_AI_FRONT_GAIN = 0.90
WALL_AI_STALL_MIX = 0.85

# 회복/출력 보정
RECOVERY_FRONT = 0.78
RECOVERY_STALL = 0.82
RECOVERY_HOLD_SEC = 0.20
RECOVERY_MIN_SEC = 0.90
RECOVERY_MAX_SEC = 1.20
WALL_ESCAPE_THRESHOLD = 0.60
WALL_REVERSE_THRESHOLD = 0.82
WALL_ESCAPE_HOLD_SEC = 0.08
OUTPUT_BALANCE_ALPHA = 0.012
LEARN_SAVE_INTERVAL = 20.0

DEFAULT_CFG = {"region": None, "s_max": 50, "v_min": 70, "v_max": 210}

user32 = ctypes.windll.user32


# ============================================================
# 설정 / 영역 선택
# ============================================================
def load_config():
    cfg = dict(DEFAULT_CFG)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception as e:
            print("[설정] 읽기 실패 -> 기본값:", e)
    return cfg


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def select_region(sct):
    mon = sct.monitors[MONITOR_INDEX]
    img = cv2.cvtColor(np.array(sct.grab(mon)), cv2.COLOR_BGRA2BGR)
    scale = 0.5 if img.shape[1] > 1600 else 1.0
    small = cv2.resize(img, None, fx=scale, fy=scale)
    print("게임 화면을 드래그한 뒤 Enter를 누르세요.")
    x, y, w, h = cv2.selectROI("select game area", small, False)
    cv2.destroyWindow("select game area")
    if w == 0 or h == 0:
        return None
    return {
        "left": mon["left"] + int(x / scale),
        "top": mon["top"] + int(y / scale),
        "width": int(w / scale),
        "height": int(h / scale),
    }


def rel_to_px(rect, w, h):
    return int(rect[0] * w), int(rect[1] * h), int(rect[2] * w), int(rect[3] * h)


# ============================================================
# 화면 -> 감각 입력
# ============================================================
def extract_features(frame, s_max, v_min, v_max):
    h0, w0 = frame.shape[:2]
    scale = PROC_WIDTH / max(1, w0)
    small = cv2.resize(frame, (PROC_WIDTH, max(1, int(h0 * scale))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]

    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    road = ((s <= s_max) & (v >= v_min) & (v <= v_max)).astype(np.uint8) * 255

    bx1, by1, bx2, by2 = rel_to_px(LOOKAHEAD_BAND, w, h)
    bx1, bx2 = np.clip([bx1, bx2], 0, w)
    by1, by2 = np.clip([by1, by2], 0, h)
    band = road[by1:by2, bx1:bx2]
    if band.size == 0:
        band = road

    colw = max(1, band.shape[1] // N_COL)
    road_cols = []
    for i in range(N_COL):
        x1 = i * colw
        x2 = band.shape[1] if i == N_COL - 1 else min(band.shape[1], (i + 1) * colw)
        road_cols.append(float(np.mean(band[:, x1:x2] > 0)) if x2 > x1 else 0.0)
    road_cols = np.asarray(road_cols, dtype=np.float32)

    def wall_score(rect):
        x1, y1, x2, y2 = rel_to_px(rect, w, h)
        x1, x2 = np.clip([x1, x2], 0, w)
        y1, y2 = np.clip([y1, y2], 0, h)
        q = road[y1:y2, x1:x2]
        if q.size == 0:
            return 0.0
        nonroad = 1.0 - float(np.mean(q > 0))
        return float(np.clip((nonroad - WALL_BASE) / max(1e-6, 1.0 - WALL_BASE), 0, 1))

    wall_l = wall_score(LEFT_SIDE_RECT)
    wall_r = wall_score(RIGHT_SIDE_RECT)

    vis = small.copy()
    cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 255, 255), 1)
    colw = max(1, (bx2 - bx1) // N_COL)
    for i, val in enumerate(road_cols):
        x0 = bx1 + i * colw
        cv2.rectangle(vis, (x0, by1 - int(val * 22)),
                       (x0 + colw - 2, by1), (0, 255, 0), -1)
    cv2.line(vis, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)

    return {
        "road": road_cols,
        "wall_l": wall_l,
        "wall_r": wall_r,
        "mask": road,
        "vis": vis,
        "coverage": float(band.mean() / 255.0),
    }


def motion_gray(frame):
    h0, w0 = frame.shape[:2]
    g = cv2.cvtColor(cv2.resize(frame, (160, max(1, int(160 * h0 / w0)))), cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    h, w = g.shape
    for rect in (MINIMAP_RECT, HUD_RECT, SPEEDO_RECT):
        x1, y1, x2, y2 = rel_to_px(rect, w, h)
        g[y1:y2, x1:x2] = 0
    return g


class StuckDetector:
    def __init__(self):
        self.reset()
        self.motion = 0.0

    def reset(self):
        self.ref = None
        self.ref_time = 0.0
        self.still_since = None

    def update(self, gray, now):
        if self.ref is None:
            self.ref = gray
            self.ref_time = now
            return
        if now - self.ref_time >= STUCK_INTERVAL:
            self.motion = float(np.mean(cv2.absdiff(gray, self.ref)))
            if self.motion < STUCK_DIFF_THRESHOLD:
                if self.still_since is None:
                    self.still_since = self.ref_time
            else:
                self.still_since = None
            self.ref = gray
            self.ref_time = now

    def still_duration(self, now):
        return 0.0 if self.still_since is None else now - self.still_since


# ============================================================
# Connectome 추출기
# ============================================================
class ConnectomeBuilder:
    """9M edge CSV에서 실제 neuron ID/syn_count를 유지한 sparse subgraph를 만든다."""

    def __init__(self):
        self.meta = {}
        self.df = None

    def load_neurons(self):
        if not os.path.exists(NEURON_CSV):
            raise FileNotFoundError(f"neurons.csv가 없습니다: {NEURON_CSV}")
        usecols = [
            "Root ID", "Predicted NT type", "Super Class", "Class",
            "Sub Class", "Primary Cell Type", "Alternative Cell Type(s)",
            "Top in/out region"
        ]
        print("[connectome] neurons.csv 읽는 중...")
        df = pd.read_csv(NEURON_CSV, usecols=usecols, low_memory=False)
        df["Root ID"] = pd.to_numeric(df["Root ID"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["Root ID"]).copy()
        df["Root ID"] = df["Root ID"].astype(np.int64)
        self.df = df
        self.meta = df.set_index("Root ID").to_dict("index")
        print(f"[connectome] neuron metadata: {len(df):,}")
        return df

    def _ids(self, mask):
        return set(self.df.loc[mask, "Root ID"].astype(np.int64).tolist())

    def build(self):
        if os.path.exists(CACHE_PATH):
            try:
                z = np.load(CACHE_PATH, allow_pickle=False)
                if int(z["version"]) == CACHE_VERSION:
                    print("[connectome] 기존 cache 사용")
                    return self._load_cache(z)
            except Exception as e:
                print("[connectome] cache 무시:", e)

        if self.df is None:
            self.load_neurons()

        desc = self._ids(self.df["Super Class"].eq("descending_neuron"))
        vproj = self._ids(self.df["Super Class"].eq("visual_projection"))
        olsens = self._ids(self.df["Super Class"].eq("ol_sensory"))
        vnsens = self._ids(self.df["Super Class"].eq("vnc_sensory"))
        da = self._ids(self.df["Predicted NT type"].eq("DA"))

        print(f"[connectome] descending={len(desc):,}, visual_projection={len(vproj):,}, "
              f"ol_sensory={len(olsens):,}, DA={len(da):,}")

        # 1) descending neuron으로 들어오는 실제 edge를 찾는다.
        incoming = defaultdict(float)
        print("[connectome] 1/4: descending 입력 연결 탐색...")
        for chunk in pd.read_csv(CONNECTION_CSV, usecols=["pre_root_id", "post_root_id", "syn_count"],
                                 chunksize=400000):
            q = chunk[chunk["post_root_id"].isin(desc)]
            if not q.empty:
                for pre, s in zip(q["pre_root_id"].to_numpy(), q["syn_count"].to_numpy()):
                    incoming[int(pre)] += float(s)

        top_desc_inputs = [x for x, _ in sorted(incoming.items(), key=lambda kv: kv[1], reverse=True)
                           [:MOTOR_DESCENDING * UPSTREAM_PER_TARGET]]
        # descending 자체도 output으로 유지한다.
        output_ids = list(desc)

        # 실제 descending 중 outgoing synapse가 많은 것만 출력 후보로 사용.
        out_count = Counter()
        print("[connectome] 2/4: descending 출력 세기 계산...")
        for chunk in pd.read_csv(CONNECTION_CSV, usecols=["pre_root_id", "post_root_id", "syn_count"],
                                 chunksize=400000):
            q = chunk[chunk["pre_root_id"].isin(desc)]
            if not q.empty:
                for pre, s in zip(q["pre_root_id"].to_numpy(), q["syn_count"].to_numpy()):
                    out_count[int(pre)] += float(s)
        selected_outputs = [x for x, _ in out_count.most_common(MOTOR_DESCENDING)]
        if len(selected_outputs) < 8:
            selected_outputs = list(desc)[:MOTOR_DESCENDING]

        # 3) 출력 후보에서 upstream을 여러 단계 확장.
        nodes = set(selected_outputs) | set(top_desc_inputs)
        frontier = set(top_desc_inputs)
        print("[connectome] 3/4: upstream 실제 연결망 확장...")
        for hop in range(EXPANSION_HOPS):
            if not frontier or len(nodes) >= MAX_UPSTREAM_TOTAL:
                break
            scores = defaultdict(float)
            for chunk in pd.read_csv(CONNECTION_CSV,
                                     usecols=["pre_root_id", "post_root_id", "syn_count"],
                                     chunksize=400000):
                q = chunk[chunk["post_root_id"].isin(frontier)]
                if not q.empty:
                    for pre, s in zip(q["pre_root_id"].to_numpy(), q["syn_count"].to_numpy()):
                        p = int(pre)
                        if p not in nodes:
                            scores[p] += float(s)
            # 시각계 뉴런에 우선권을 조금 준다. 실제 ID는 그대로 유지한다.
            ranked = []
            for nid, score in scores.items():
                m = self.meta.get(nid, {})
                sc = str(m.get("Super Class", ""))
                bonus = 2.0 if sc in ("visual_projection", "ol_sensory", "visual_centrifugal") else 1.0
                ranked.append((score * bonus, nid))
            ranked.sort(reverse=True)
            add = [nid for _, nid in ranked[:min(5000, MAX_UPSTREAM_TOTAL - len(nodes))]]
            nodes.update(add)
            frontier = set(add)
            print(f"  hop {hop + 1}: +{len(add):,}, total {len(nodes):,}")

        # 입력으로 쓸 실제 visual/olfactory sensory neuron 후보.
        input_candidates = [nid for nid in nodes
                            if self.meta.get(nid, {}).get("Super Class") in
                            ("ol_sensory", "visual_projection", "visual_centrifugal", "vnc_sensory")]
        if len(input_candidates) < 64:
            # 그래프 확장 결과가 부족하면 metadata상 visual projection을 일부 추가.
            input_candidates = list(vproj)[:min(2000, len(vproj))]
            nodes.update(input_candidates)
        else:
            input_candidates = input_candidates[:min(2500, len(input_candidates))]

        # DA 뉴런은 reward modulation을 위해 실제 DA neuron ID로 추가한다.
        da_candidates = [nid for nid in nodes if nid in da]
        if not da_candidates:
            # 전체 DA에서 소수만 추가. 연결 edge가 있을 때만 실제 학습에 사용된다.
            da_candidates = list(da)[:256]
            nodes.update(da_candidates)

        # 실제 graph edge를 한 번 더 읽어서 nodes 내부 연결만 저장한다.
        print("[connectome] 4/4: 최종 edge 추출...")
        edges = []
        for chunk in pd.read_csv(CONNECTION_CSV,
                                 usecols=["pre_root_id", "post_root_id", "syn_count", "neuropil"],
                                 chunksize=400000):
            q = chunk[chunk["pre_root_id"].isin(nodes) & chunk["post_root_id"].isin(nodes)]
            if not q.empty:
                edges.extend(zip(
                    q["pre_root_id"].astype(np.int64).tolist(),
                    q["post_root_id"].astype(np.int64).tolist(),
                    q["syn_count"].astype(np.float32).tolist(),
                ))

        if not edges:
            raise RuntimeError("선택된 connectome subgraph에서 edge를 찾지 못했습니다.")

        # 중복 edge는 syn_count 합산.
        pair_sum = defaultdict(float)
        for pre, post, syn in edges:
            pair_sum[(pre, post)] += float(syn)

        items = [(pre, post, syn) for (pre, post), syn in pair_sum.items()]
        # 너무 큰 edge 수는 synapse가 큰 순으로 자르되, 실제 edge만 남긴다.
        if len(items) > MAX_EDGES:
            items.sort(key=lambda x: x[2], reverse=True)
            items = items[:MAX_EDGES]

        node_ids = sorted(set([x[0] for x in items] + [x[1] for x in items] + selected_outputs + input_candidates))
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        pre = np.asarray([id_to_idx[x[0]] for x in items], dtype=np.int32)
        post = np.asarray([id_to_idx[x[1]] for x in items], dtype=np.int32)
        syn = np.asarray([x[2] for x in items], dtype=np.float32)

        # syn_count는 연결 강도이므로 log로 압축한다.
        # connections_princeton.csv의 nt_type은 비어 있으므로 neurons.csv의
        # pre neuron Predicted NT type을 이용한다. GABA는 억제성으로 취급한다.
        # (이 부호는 CSV가 직접 제공하는 값이 아니라 계산 모델의 가정이다.)
        w = np.log1p(syn).astype(np.float32) * SYN_GAIN
        pre_nt = np.array([str(self.meta.get(int(x[0]), {}).get("Predicted NT type", "")) for x in items])
        w[pre_nt == "GABA"] *= -INHIBITORY_SCALE
        w[pre_nt != "GABA"] *= EXCITATORY_SCALE
        w = np.clip(w, -2.0, 2.0)

        input_ids = np.asarray([nid for nid in input_candidates if nid in id_to_idx], dtype=np.int64)
        output_ids = np.asarray([nid for nid in selected_outputs if nid in id_to_idx], dtype=np.int64)
        da_ids = np.asarray([nid for nid in da_candidates if nid in id_to_idx], dtype=np.int64)

        # 입력 뉴런은 채널별로 deterministic하게 분배한다.
        input_channels = np.arange(len(input_ids), dtype=np.int16) % N_INPUT

        np.savez_compressed(
            CACHE_PATH,
            version=np.asarray(CACHE_VERSION),
            node_ids=np.asarray(node_ids, dtype=np.int64),
            pre=pre,
            post=post,
            weight=w,
            input_ids=input_ids,
            input_channels=input_channels,
            output_ids=output_ids,
            da_ids=da_ids,
        )
        print(f"[connectome] cache 저장: nodes={len(node_ids):,}, edges={len(w):,}, "
              f"inputs={len(input_ids):,}, outputs={len(output_ids):,}, DA={len(da_ids):,}")
        return self._load_cache(np.load(CACHE_PATH, allow_pickle=False))

    def _load_cache(self, z):
        return {
            "node_ids": z["node_ids"].astype(np.int64),
            "pre": z["pre"].astype(np.int32),
            "post": z["post"].astype(np.int32),
            "weight": z["weight"].astype(np.float32),
            "input_ids": z["input_ids"].astype(np.int64),
            "input_channels": z["input_channels"].astype(np.int16),
            "output_ids": z["output_ids"].astype(np.int64),
            "da_ids": z["da_ids"].astype(np.int64),
        }


# ============================================================
# 실제 connectome sparse SNN
# ============================================================
class ConnectomeBrain:
    """실제 CSV neuron ID/edge를 보존한 sparse LIF connectome 엔진."""

    def __init__(self):
        # main loop가 첫 update/reward 전에 참조하므로 반드시 초기화한다.
        self.steer = 0.0
        self.drive = "forward"
        self.last_t = None
        self.last_output_ids = []
        self.last_reward = 0.0
        self.reward_total = 0.0
        self.event_count = 0
        # 실제 descending neuron 집계의 좌우 비대칭을 런타임에 중립화한다.
        # Root ID 홀짝은 생물학적 좌/우 표지가 아니므로 모터 인터페이스의 영점 보정으로만 사용한다.
        self.output_bias = 0.0
        self.last_left_rate = 0.0
        self.last_right_rate = 0.0
        self.last_forward_rate = 0.0
        self.last_reverse_rate = 0.0

        self.graph = ConnectomeBuilder().build()
        g = self.graph
        self.node_ids = g["node_ids"]
        self.n = len(self.node_ids)
        self.pre = g["pre"]
        self.post = g["post"]
        self.base_w = g["weight"]
        self.w = self.base_w.copy()
        self.input_ids = g["input_ids"]
        self.input_channels = g["input_channels"]
        self.output_ids = g["output_ids"]
        self.da_ids = g["da_ids"]

        self.id_to_idx = {int(x): i for i, x in enumerate(self.node_ids.tolist())}
        self.input_idx = np.asarray([self.id_to_idx[int(x)] for x in self.input_ids if int(x) in self.id_to_idx], dtype=np.int32)
        self.output_idx = np.asarray([self.id_to_idx[int(x)] for x in self.output_ids if int(x) in self.id_to_idx], dtype=np.int32)
        self.da_idx = np.asarray([self.id_to_idx[int(x)] for x in self.da_ids if int(x) in self.id_to_idx], dtype=np.int32)

        # output descending neuron은 실제 graph의 selected output이다.
        self.output_sign = np.zeros(len(self.output_idx), dtype=np.float32)
        for i, idx in enumerate(self.output_idx):
            rid = int(self.node_ids[idx])
            # deterministic 좌우 분할. 실제 lateral function annotation이 CSV에 없는 경우의 출력 인터페이스.
            self.output_sign[i] = -1.0 if (rid % 2 == 0) else 1.0

        self.v = np.zeros(self.n, dtype=np.float32)
        self.syn = np.zeros(self.n, dtype=np.float32)
        self.rate = np.zeros(self.n, dtype=np.float32)
        self.spikes = np.zeros(self.n, dtype=np.uint8)
        self.prev_spikes = np.zeros(self.n, dtype=np.uint8)
        self.refractory = np.zeros(self.n, dtype=np.int8)

        # reward-modulated eligibility는 실제 edge에만 연결된다.
        self.plastic_mask = self._choose_plastic_edges()
        self.eligibility = np.zeros(int(self.plastic_mask.sum()), dtype=np.float32)
        self.edge_matrix = None
        self.edge_data_pos = None
        if csr_matrix is not None:
            self.edge_matrix = csr_matrix((self.w, (self.post, self.pre)), shape=(self.n, self.n), dtype=np.float32)
            # CSR가 정렬한 data 위치와 원래 edge index를 연결한다.
            pos_matrix = csr_matrix((np.arange(len(self.w), dtype=np.int32), (self.post, self.pre)),
                                    shape=(self.n, self.n), dtype=np.int32)
            self.edge_data_pos = np.asarray(pos_matrix.data, dtype=np.int32)
        self.load_learning()
        print(f"[뇌] REAL CONNECTOME subgraph: {self.n:,} neurons / {len(self.w):,} edges")
        print(f"[뇌] input={len(self.input_idx):,} / descending output={len(self.output_idx):,} / DA={len(self.da_idx):,}")
        print(f"[뇌] plastic edges={int(self.plastic_mask.sum()):,}")

    def _choose_plastic_edges(self):
        # 실제 connectome edge 중 output/DA와 연결되거나 강한 edge 일부를 plastic 후보로 사용.
        relevant = np.zeros(len(self.w), dtype=bool)
        if len(self.output_idx):
            relevant |= np.isin(self.post, self.output_idx)
            relevant |= np.isin(self.pre, self.output_idx)
        if len(self.da_idx):
            relevant |= np.isin(self.pre, self.da_idx)
            relevant |= np.isin(self.post, self.da_idx)
        idx = np.flatnonzero(relevant)
        if len(idx) > MAX_PLASTIC_EDGES:
            # 강한 실제 synapse부터 선택
            order = idx[np.argsort(self.w[idx])[-MAX_PLASTIC_EDGES:]]
            mask = np.zeros(len(self.w), dtype=bool)
            mask[order] = True
            return mask
        return relevant

    def reset_state(self):
        self.v.fill(0)
        self.syn.fill(0)
        self.rate.fill(0)
        self.spikes.fill(0)
        self.prev_spikes.fill(0)
        self.refractory.fill(0)
        self.last_output_ids = []
        self.steer = 0.0
        self.drive = "forward"
        self.last_t = None
        self.output_bias = 0.0
        self.last_left_rate = 0.0
        self.last_right_rate = 0.0
        self.last_forward_rate = 0.0
        self.last_reverse_rate = 0.0

    def _input_current(self, x_ch):
        current = np.zeros(self.n, dtype=np.float32)
        if len(self.input_idx) == 0:
            return current
        vals = np.asarray(x_ch, dtype=np.float32)
        vals = np.clip(vals, 0.0, 1.0)
        current[self.input_idx] += vals[self.input_channels]
        return current

    def _simulate_step(self, x_ch, reward_drive=0.0):
        # 외부 sensory interface -> 실제 sensory/visual neuron들
        self.syn *= np.exp(-BRAIN_DT / TAU_SYN)
        self.syn += self._input_current(x_ch)

        if len(self.da_idx):
            # 보상은 실제 DA 뉴런에 외부 spike/current로 주입한다.
            self.syn[self.da_idx] += max(0.0, float(reward_drive)) * 0.75

        # 실제 edge propagation: pre -> post, weight=syn_count 기반
        if self.edge_matrix is not None:
            self.syn += self.edge_matrix @ self.spikes.astype(np.float32)
        else:
            fired_idx = np.flatnonzero(self.spikes)
            if len(fired_idx):
                edge_mask = np.isin(self.pre, fired_idx)
                e = np.flatnonzero(edge_mask)
                if len(e):
                    np.add.at(self.syn, self.post[e], self.w[e])

        self.v *= np.exp(-BRAIN_DT / TAU_M)
        self.v += (1.0 - np.exp(-BRAIN_DT / TAU_M)) * self.syn

        active_ref = self.refractory > 0
        self.v[active_ref] = 0.0
        self.refractory[active_ref] -= 1

        fired = self.v >= V_THRESHOLD
        self.v[fired] = V_RESET
        self.refractory[fired] = REFRACTORY_STEPS
        self.prev_spikes[:] = self.spikes
        self.spikes[:] = fired.astype(np.uint8)

        self.rate *= np.exp(-BRAIN_DT / TAU_RATE)
        self.rate += (1.0 - np.exp(-BRAIN_DT / TAU_RATE)) * self.spikes

        # eligibility: pre/post spike timing을 단순한 실제 edge 기반으로 추적
        if self.plastic_mask.any():
            pidx = np.flatnonzero(self.plastic_mask)
            pre_fire = self.spikes[self.pre[pidx]].astype(np.float32)
            post_fire = self.spikes[self.post[pidx]].astype(np.float32)
            hebb = pre_fire * 0.65 + post_fire * 0.35
            self.eligibility *= ELIGIBILITY_DECAY
            self.eligibility += hebb

    def _read_output(self, wall_l=0.0, wall_r=0.0, stall=0.0):
        if len(self.output_idx) == 0:
            return 0.0, "forward", []

        r = self.rate[self.output_idx]
        left_mask = self.output_sign < 0
        right_mask = self.output_sign > 0
        left = float(r[left_mask].mean()) if np.any(left_mask) else 0.0
        right = float(r[right_mask].mean()) if np.any(right_mask) else 0.0
        forward = float(np.mean(r))
        reverse = float(np.percentile(r, 90))

        self.last_left_rate = left
        self.last_right_rate = right
        self.last_forward_rate = forward
        self.last_reverse_rate = reverse

        # 직선/저위험 구간에서는 출력 좌우 평균 차이를 천천히 0으로 보정한다.
        # 현재 CSV에는 descending neuron의 좌/우 운동 기능 annotation이 없고 Root ID 홀짝은
        # 생물학적 좌/우 근거가 아니므로, 이 보정은 '모터 인터페이스 영점'일 뿐 connectome 자체의 재배선이 아니다.
        if max(wall_l, wall_r) < 0.22 and stall < 0.18:
            self.output_bias = (1.0 - OUTPUT_BALANCE_ALPHA) * self.output_bias + OUTPUT_BALANCE_ALPHA * (right - left)

        steer_raw = np.tanh((right - left - self.output_bias) * 7.0)

        # 기본적으로는 실제 descending output이 후진을 결정한다.
        drive = "reverse" if (reverse > 0.55 and forward < 0.28) else "forward"
        active = self.output_idx[self.rate[self.output_idx] > 0.25]
        ids = [int(self.node_ids[i]) for i in active[:12]]
        return float(np.clip(steer_raw, -1, 1)), drive, ids

    def update(self, now, road_cols, wall_l, wall_r, stall, reward=0.0):
        x = np.zeros(N_INPUT, dtype=np.float32)
        x[:N_COL] = np.clip(road_cols, 0, 1)
        x[8] = np.clip(stall, 0, 1)
        x[9] = np.clip(wall_l, 0, 1)
        x[10] = np.clip(wall_r, 0, 1)

        # 실제 connectome의 update 속도를 유지하면서 frame 간 시간만큼 계산.
        dt = BRAIN_DT if not hasattr(self, "last_t") or self.last_t is None else max(0.001, now - self.last_t)
        self.last_t = now
        steps = int(np.clip(round(dt / BRAIN_DT), 1, MAX_STEPS_PER_UPDATE))
        reward_drive = max(0.0, float(reward))
        for _ in range(steps):
            self._simulate_step(x, reward_drive)

        steer, drive, ids = self._read_output(wall_l, wall_r, stall)
        self.steer = steer
        self.drive = drive
        self.last_output_ids = ids
        self.event_count += 1
        if reward != 0:
            self.apply_reward(reward)
        return steer, drive

    def apply_reward(self, reward):
        r = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))
        self.last_reward = r
        self.reward_total += r
        if not self.plastic_mask.any():
            return
        pidx = np.flatnonzero(self.plastic_mask)
        # reward-modulated Hebbian/STDP-like update.
        # graph topology와 neuron ID는 바꾸지 않고 synaptic weight만 조금 조정한다.
        delta = LEARNING_RATE * r * self.eligibility
        self.w[pidx] += delta.astype(np.float32)
        base = self.base_w[pidx]
        lower = np.where(base < 0.0, base * 4.0, base * 0.25)
        upper = np.where(base < 0.0, base * 0.25, base * 4.0 + 0.1)
        self.w[pidx] = np.clip(self.w[pidx], lower, upper)
        if self.edge_matrix is not None and self.edge_data_pos is not None:
            # edge_data_pos는 CSR data 배열에서 각 원래 edge가 위치한 곳을 가리킨다.
            self.edge_matrix.data[self.edge_data_pos[pidx]] = self.w[pidx]

    def api_snapshot(self, feat=None, stall=0.0, reward=0.0, driving=False, fps=0.0):
        """외부 brain_viewer.py가 읽을 수 있는 가벼운 상태 스냅샷."""
        out_rate = self.rate[self.output_idx] if len(self.output_idx) else np.empty(0, dtype=np.float32)
        active_idx = np.flatnonzero(self.spikes)
        # 14,000개 전체 배열을 HTTP로 보내지 않고 활동 뉴런 일부만 보낸다.
        active = []
        for idx in active_idx[:160]:
            active.append({
                "id": int(self.node_ids[idx]),
                "rate": float(self.rate[idx]),
                "v": float(self.v[idx]),
            })
        top = []
        if len(out_rate):
            order = np.argsort(out_rate)[::-1][:12]
            for j in order:
                idx = int(self.output_idx[j])
                top.append({
                    "id": int(self.node_ids[idx]),
                    "rate": float(out_rate[j]),
                    "sign": float(self.output_sign[j]),
                })
        return {
            "api_version": API_VERSION,
            "time": time.time(),
            "fps": float(fps),
            "driving": bool(driving),
            "steer": float(getattr(self, "steer", 0.0)),
            "drive": str(getattr(self, "drive", "forward")),
            "reward": float(reward),
            "reward_total": float(self.reward_total),
            "last_reward": float(self.last_reward),
            "event_count": int(self.event_count),
            "neurons": int(self.n),
            "edges": int(len(self.w)),
            "input_count": int(len(self.input_idx)),
            "output_count": int(len(self.output_idx)),
            "da_count": int(len(self.da_idx)),
            "plastic_edges": int(self.plastic_mask.sum()),
            "active_count": int(len(active_idx)),
            "active": active,
            "top_outputs": top,
            "last_output_ids": [int(x) for x in self.last_output_ids],
            "inputs": {
                "road": [float(x) for x in (feat["road"] if feat is not None else np.zeros(N_COL))],
                "wall_l": float(feat["wall_l"] if feat is not None else 0.0),
                "wall_r": float(feat["wall_r"] if feat is not None else 0.0),
                "coverage": float(feat["coverage"] if feat is not None else 0.0),
                "stall": float(stall),
            },
        }

    def save_learning(self):
        try:
            pidx = np.flatnonzero(self.plastic_mask)
            np.savez_compressed(
                LEARN_PATH.replace(".json", ".npz"),
                edge_index=pidx.astype(np.int32),
                weight=self.w[pidx].astype(np.float32),
                eligibility=self.eligibility.astype(np.float32),
            )
            with open(LEARN_PATH, "w", encoding="utf-8") as f:
                json.dump({
                    "version": 1,
                    "reward_total": self.reward_total,
                    "events": self.event_count,
                    "plastic_edges": int(len(pidx)),
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("[뇌] 학습 저장 실패:", e)

    def load_learning(self):
        p = LEARN_PATH.replace(".json", ".npz")
        if not os.path.exists(p):
            return
        try:
            z = np.load(p, allow_pickle=False)
            idx = z["edge_index"].astype(np.int64)
            w = z["weight"].astype(np.float32)
            e = z["eligibility"].astype(np.float32)
            if len(idx) == len(w) == len(e):
                valid = (idx >= 0) & (idx < len(self.w))
                idx = idx[valid]
                self.w[idx] = w[valid]
                self.eligibility = np.zeros(int(self.plastic_mask.sum()), dtype=np.float32)
                current = np.flatnonzero(self.plastic_mask)
                lookup = {int(v): i for i, v in enumerate(current.tolist())}
                for j, edge in enumerate(idx.tolist()):
                    k = lookup.get(int(edge))
                    if k is not None and j < len(e):
                        self.eligibility[k] = e[j]
            if self.edge_matrix is not None and self.edge_data_pos is not None:
                current = np.flatnonzero(self.plastic_mask)
                # edge_data_pos는 원본 edge -> CSR data 위치 매핑이다.
                self.edge_matrix.data[self.edge_data_pos[current]] = self.w[current]
            print(f"[뇌] 기존 reward 학습값 복원: {len(idx):,} edges")
        except Exception as ex:
            print("[뇌] 기존 학습값 복원 실패:", ex)


# ============================================================
# 보상 추정기
# ============================================================
class RewardSystem:
    def __init__(self):
        self.last = time.time()
        self.prev_coverage = 0.0
        self.prev_wall = 0.0
        self.last_stuck = 0.0
        self.last_drive = "forward"
        self.total = 0.0
        self.last_event = ""
        self.cooldowns = {}

    def _event(self, name, value, cooldown=0.5):
        now = time.time()
        last = self.cooldowns.get(name, -999.0)
        if now - last < cooldown:
            return 0.0
        self.cooldowns[name] = now
        self.last_event = name
        return value

    def step(self, feat, stall, drive, speed_hint=0.0, front_block=0.0):
        # 직진 중이라도 벽/전방 막힘 상태에서는 positive reward를 주지 않는다.
        wall = max(feat["wall_l"], feat["wall_r"])
        safe_forward = (drive == "forward" and stall < 0.2 and wall < 0.45 and front_block < 0.45)
        reward = REWARD_NORMAL if safe_forward else 0.0

        # 도로 coverage가 안정적으로 유지되면 작은 추가 보상.
        if feat["coverage"] > 0.42 and wall < 0.35 and stall < 0.15 and front_block < 0.35:
            reward += REWARD_STEADY

        # 코너: 좌우 도로 비대칭이 생겼다가 다시 중앙으로 돌아오는 경우.
        center_error = abs(float(np.mean(feat["road"][4:])) - float(np.mean(feat["road"][:4])))
        if center_error > 0.22 and wall < 0.65 and stall < 0.2:
            reward += self._event("corner", REWARD_CORNER, 1.0)

        # drift의 직접적인 게임 상태를 화면만으로 확정할 수 없으므로,
        # 여기서는 빠른 조향 + 도로 유지라는 대리 신호만 사용한다.
        if abs(getattr(self, "last_steer", 0.0)) > 0.65 and wall < 0.3 and stall < 0.15:
            reward += self._event("drift_proxy", REWARD_DRIFT, 1.5)

        if wall > 0.72:
            reward += self._event("wall", PENALTY_WALL, 0.7)
        if stall > 0.65:
            reward += self._event("stuck", PENALTY_STUCK, 1.0)

        self.total += reward
        self.last_stuck = stall
        self.last_drive = drive
        return reward

    def collision(self):
        r = self._event("collision", PENALTY_COLLISION, 1.0)
        self.total += r
        return r

    def reset(self):
        r = self._event("reset", PENALTY_RESET, 2.0)
        self.total += r
        return r

    def set_steer(self, steer):
        self.last_steer = steer


# ============================================================
# vJoy
# ============================================================
class Driver:
    def __init__(self, device_id):
        self.dev = None
        self.lock = threading.Lock()
        self.steer = 0.0
        self.drive = "idle"
        self.force_dir = None
        self.last_x = None
        self.last_y = None
        self.running = False
        if pyvjoy is None:
            print("[vJoy] pyvjoy가 없어 출력 비활성화")
            return
        try:
            self.dev = pyvjoy.VJoyDevice(device_id)
            self.dev.reset()
            print("[vJoy] 장치 열기 성공")
        except Exception as e:
            print("[vJoy] 장치 열기 실패:", e)
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    @staticmethod
    def _pwm(steer, now):
        mag = abs(steer)
        if mag < STEER_DEADZONE:
            return 0
        sign = 1 if steer > 0 else -1
        if STEER_MODE == "bang":
            return sign
        duty = min(1.0, mag / FULL_TURN)
        phase = (now % PWM_PERIOD) / PWM_PERIOD
        return sign if phase < duty else 0

    def _loop(self):
        while self.running:
            now = time.time()
            with self.lock:
                steer, drive, force = self.steer, self.drive, self.force_dir
            d = force if force is not None else self._pwm(steer, now)
            if STEER_INVERT:
                d = -d
            x = AXIS_CENTER if d == 0 else (AXIS_MIN if d < 0 else AXIS_MAX)
            y = {"forward": FORWARD_VALUE, "reverse": REVERSE_VALUE}.get(drive, AXIS_CENTER)
            try:
                if x != self.last_x:
                    self.dev.set_axis(pyvjoy.HID_USAGE_X, x)
                    self.last_x = x
                if y != self.last_y:
                    self.dev.set_axis(pyvjoy.HID_USAGE_Y, y)
                    self.last_y = y
            except Exception as e:
                print("[vJoy] 출력 오류:", e)
            time.sleep(0.005)

    def command(self, steer, drive, force_dir=None):
        with self.lock:
            self.steer, self.drive, self.force_dir = float(steer), drive, force_dir

    def release(self):
        with self.lock:
            self.steer, self.drive, self.force_dir = 0.0, "idle", None

    def shutdown(self):
        self.release()
        time.sleep(0.05)
        self.running = False
        if self.dev is not None:
            try:
                self.dev.reset()
            except Exception:
                pass


class Hotkey:
    def __init__(self, vk):
        self.vk = vk
        self.prev = False

    def pressed(self):
        down = bool(user32.GetAsyncKeyState(self.vk) & 0x8000)
        hit = down and not self.prev
        self.prev = down
        return hit


# ============================================================
# 외부 시각화용 상태 API
# ============================================================
class _APIHandler(BaseHTTPRequestHandler):
    server_version = "ChopariBrainAPI/1.0"

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        if self.path.split("?", 1)[0] == "/state":
            with self.server.state_lock:
                payload = dict(self.server.state)
            raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if self.path.split("?", 1)[0] == "/health":
            raw = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        self.send_response(404)
        self.end_headers()


class BrainAPIServer:
    def __init__(self, host=API_HOST, port=API_PORT):
        self.state = {"api_version": API_VERSION, "ready": False}
        self.state_lock = threading.Lock()
        self.httpd = ThreadingHTTPServer((host, port), _APIHandler)
        self.httpd.state = self.state
        self.httpd.state_lock = self.state_lock
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self):
        self.thread.start()
        print(f"[API] brain state: http://{API_HOST}:{API_PORT}/state")
        print(f"[API] health:      http://{API_HOST}:{API_PORT}/health")

    def update(self, state):
        with self.state_lock:
            self.state.clear()
            self.state.update(state)

    def stop(self):
        try:
            self.httpd.shutdown()
            self.httpd.server_close()
        except Exception:
            pass


# ============================================================
# 메인
# ============================================================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    cfg = load_config()

    if "--rebuild" in sys.argv:
        if os.path.exists(CACHE_PATH):
            os.remove(CACHE_PATH)
        print("[connectome] cache 삭제 -> 다음 실행에서 재생성")

    with mss.MSS() as sct:
        if "--select" in sys.argv or not cfg.get("region"):
            region = select_region(sct)
            if not region:
                print("영역 선택이 취소되었습니다.")
                return
            cfg["region"] = region
            save_config(cfg)
        else:
            region = cfg["region"]

        print("[connectome] 실제 connectome 로딩 시작...")
        brain = ConnectomeBrain()
        joy = Driver(VJOY_DEVICE_ID)
        stuck = StuckDetector()
        rewarder = RewardSystem()
        api = BrainAPIServer()

        # Wall AI는 게임 화면을 직접 보고, 결과를 connectome의 wall/stall 감각 채널로 전달한다.
        wall_ai = WallAI() if WallAI is not None else None
        wall_ai_result = {
            "left_wall": 0.0,
            "right_wall": 0.0,
            "front_block": 0.0,
            "center_wall": 0.0,
            "road_confidence": 0.0,
            "ready": False,
        }
        wall_ai_last = 0.0
        if wall_ai is not None and getattr(wall_ai, "loaded", False):
            print("[WallAI] 모델 연결 성공: wall_ai_model.npz")
        elif wall_ai is not None:
            print("[WallAI] 모델이 없어 기존 화면 감각만 사용합니다.")
        else:
            print("[WallAI] 모듈을 사용할 수 없어 기존 벽 감각으로 실행합니다.")

        api.start()

        hk_drive = Hotkey(0x77)  # F8
        hk_stop = Hotkey(0x78)   # F9
        hk_rebuild = Hotkey(0x79)  # F10
        hk_stats = Hotkey(0x7A)   # F11

        driving = False
        recording = False
        last_save = 0.0
        rec_count = 0
        fps_timer = time.time()
        fps_frames = 0
        fps = 0.0
        grace_until = 0.0
        prev_drive = "forward"
        prev_stall = 0.0
        smooth_steer = 0.0
        frame_count = 0
        recovery_since = None
        recovery_started = None
        last_learn_save = time.time()

        print("=" * 70)
        print("초파리라이더 REAL CONNECTOME")
        print("F8=주행 / F9=정지 / F10=cache 재생성 / F11=통계")
        print("q=종료 / s=스크린샷 / r=녹화 / c=영역 재선택")
        print("=" * 70)

        try:
            while True:
                now = time.time()
                frame = cv2.cvtColor(np.array(sct.grab(region)), cv2.COLOR_BGRA2BGR)
                feat = extract_features(frame, cfg.get("s_max", 50), cfg.get("v_min", 70), cfg.get("v_max", 210))

                # --------------------------------------------------------
                # Wall AI -> connectome sensory interface
                # --------------------------------------------------------
                if wall_ai is not None and wall_ai.loaded and (now - wall_ai_last >= WALL_AI_INTERVAL):
                    try:
                        wall_ai_result = wall_ai.analyze(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        wall_ai_last = now
                    except Exception as e:
                        # 한 프레임의 오류 때문에 주행 전체를 멈추지 않고 마지막 정상 결과를 유지한다.
                        print(f"[WallAI] 분석 오류: {e}")

                ai_l = float(np.clip(wall_ai_result.get("left_wall", 0.0) * WALL_AI_LEFT_GAIN, 0.0, 1.0))
                ai_r = float(np.clip(wall_ai_result.get("right_wall", 0.0) * WALL_AI_RIGHT_GAIN, 0.0, 1.0))
                ai_front = float(np.clip(wall_ai_result.get("front_block", 0.0) * WALL_AI_FRONT_GAIN, 0.0, 1.0))

                # Wall AI가 준비되면 학습된 wall 신호를 우선 사용한다.
                # road 열 정보는 기존 connectome의 8개 시각 채널을 그대로 유지한다.
                if wall_ai is not None and wall_ai.loaded:
                    feat["wall_l"] = ai_l
                    feat["wall_r"] = ai_r
                    feat["wall_ai"] = wall_ai_result
                else:
                    feat["wall_ai"] = wall_ai_result

                gray = motion_gray(frame)
                stuck.update(gray, now)

                stall = 0.0
                if driving and now >= grace_until:
                    stall = float(np.clip(stuck.still_duration(now) / STALL_FULL_SEC, 0, 1))

                if hk_drive.pressed():
                    driving = not driving
                    if driving:
                        grace_until = now + STUCK_GRACE
                        stuck.reset()
                        brain.reset_state()
                        print("[주행] ON")
                    else:
                        joy.release()
                        print("[주행] OFF")

                if hk_stop.pressed():
                    driving = False
                    joy.release()
                    print("[정지] F9")

                if hk_stats.pressed():
                    print(f"[통계] events={brain.event_count:,} reward={brain.reward_total:+.3f} "
                          f"last={brain.last_reward:+.3f} neurons={brain.n:,} edges={len(brain.w):,}")
                    print(f"[통계] active descending IDs={brain.last_output_ids}")

                if hk_rebuild.pressed():
                    driving = False
                    joy.release()
                    print("[connectome] cache를 삭제합니다. 프로그램을 재시작하면 재생성됩니다.")
                    if os.path.exists(CACHE_PATH):
                        os.remove(CACHE_PATH)
                    break

                reward = 0.0
                if driving:
                    # 전방 Wall AI 신호를 기존 stall 감각에 혼합한다.
                    # 즉, Wall AI가 앞을 막았다고 판단하면 connectome의 막힘/회피 입력이 즉시 강해진다.
                    sensory_stall = float(np.clip(max(stall, ai_front * WALL_AI_STALL_MIX), 0.0, 1.0))

                    reward = rewarder.step(feat, sensory_stall, brain.drive, 0.0, front_block=ai_front)

                    # 화면 변화가 갑자기 떨어지는 경우 reset/stuck에 대한 추가 penalty.
                    if prev_stall < 0.4 and (stall >= 0.85 or ai_front >= 0.82):
                        reward += rewarder.collision()
                    if prev_drive == "reverse" and brain.drive == "forward" and sensory_stall < 0.15:
                        reward += REWARD_CORNER

                    steer, drive = brain.update(
                        now,
                        feat["road"], feat["wall_l"], feat["wall_r"], sensory_stall,
                        reward=reward,
                    )
                    rewarder.set_steer(steer)

                    # ----------------------------------------------------
                    # 비상 탈출 반사: connectome이 계속 전진을 고집해도
                    # 실제 화면에서 일정 시간 막혀 있으면 후진해서 빠져나온다.
                    # 정상 주행에서는 이 경로가 개입하지 않는다.
                    # ----------------------------------------------------
                    # 벽이 실제로 가까워졌을 때는 connectome의 조향보다 Wall AI 회피를 우선한다.
                    # 왼쪽 벽이 높으면 오른쪽, 오른쪽 벽이 높으면 왼쪽으로 강제 조향한다.
                    wall_near = max(ai_l, ai_r) >= WALL_ESCAPE_THRESHOLD
                    wall_very_near = max(ai_l, ai_r) >= WALL_REVERSE_THRESHOLD
                    blocked = (ai_front >= RECOVERY_FRONT or sensory_stall >= RECOVERY_STALL or wall_very_near)
                    if blocked:
                        if recovery_since is None:
                            recovery_since = now
                        if recovery_started is None and now - recovery_since >= RECOVERY_HOLD_SEC:
                            recovery_started = now
                            print("[회복] 벽/전방 막힘 -> 후진 + 반대 조향")
                    else:
                        recovery_since = None
                        if recovery_started is not None and now - recovery_started >= RECOVERY_MIN_SEC:
                            recovery_started = None

                    # 회복이 너무 오래 지속되면 한 방향만 계속 누르지 않고 정상 주행으로 복귀한다.
                    if recovery_started is not None and now - recovery_started >= RECOVERY_MAX_SEC:
                        recovery_started = None
                        recovery_since = None
                        drive = "forward"
                        print("[회복] 최대 후진 시간 도달 -> 전진 복귀")

                    if recovery_started is not None:
                        # 후진 중에도 반드시 조향한다.
                        # left wall가 강하면 +1(우회전), right wall가 강하면 -1(좌회전).
                        # 현재 vJoy 방향에서는 음수 = 오른쪽, 양수 = 왼쪽으로
                        # 동작하므로 벽의 반대 방향을 명시적으로 강제한다.
                        if ai_l > ai_r + 0.08:
                            recovery_steer = -1.0   # 왼쪽 벽 -> 오른쪽
                        elif ai_r > ai_l + 0.08:
                            recovery_steer = 1.0    # 오른쪽 벽 -> 왼쪽
                        else:
                            recovery_steer = -float(np.sign(steer)) if abs(steer) >= 0.12 else -1.0
                        smooth_steer = recovery_steer
                        drive = "reverse"
                    else:
                        # 평상시에는 Wall AI가 조향을 직접 덮어쓰지 않는다.
                        # Wall AI는 감각/회복 트리거로만 사용하고 조향은 connectome 출력에 맡긴다.
                        smooth_steer = (1.0 - STEER_SMOOTH) * smooth_steer + STEER_SMOOTH * steer

                    force_dir = None
                    if recovery_started is not None:
                        force_dir = 1 if smooth_steer > 0 else -1
                    joy.command(smooth_steer, drive, force_dir)
                    prev_drive = drive
                else:
                    joy.release()
                    recovery_since = None
                    recovery_started = None

                # 학습은 온라인으로 계속 적용되고, 주기적으로 디스크에 저장한다.
                if now - last_learn_save >= LEARN_SAVE_INTERVAL and brain.event_count > 0:
                    brain.save_learning()
                    last_learn_save = now

                prev_stall = stall
                frame_count += 1
                fps_frames += 1
                if now - fps_timer >= 1.0:
                    fps = fps_frames / (now - fps_timer)
                    fps_frames = 0
                    fps_timer = now

                if recording and now - last_save >= 1.0 / RECORD_FPS:
                    path = os.path.join(SAVE_DIR, f"rec_{int(now * 1000)}.jpg")
                    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
                    last_save = now
                    rec_count += 1

                preview = cv2.resize(
                    feat["vis"],
                    (PREVIEW_WIDTH, max(1, int(PREVIEW_WIDTH * frame.shape[0] / frame.shape[1]))),
                    interpolation=cv2.INTER_NEAREST,
                )
                # Wall AI 상태를 화면에 직접 표시한다.
                hpre, wpre = preview.shape[:2]
                panel_x1 = max(6, wpre - 300)
                panel_y1 = 8
                panel_x2 = wpre - 8
                panel_y2 = 126
                cv2.rectangle(preview, (panel_x1, panel_y1), (panel_x2, panel_y2), (25, 25, 25), -1)
                cv2.rectangle(preview, (panel_x1, panel_y1), (panel_x2, panel_y2), (180, 180, 180), 1)
                cv2.putText(preview, "WALL AI", (panel_x1 + 8, panel_y1 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                def _bar(y, label, val):
                    val = float(np.clip(val, 0.0, 1.0))
                    x0 = panel_x1 + 78
                    x1 = panel_x2 - 10
                    cv2.putText(preview, label, (panel_x1 + 8, y + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)
                    cv2.rectangle(preview, (x0, y - 6), (x1, y + 5), (70, 70, 70), -1)
                    cv2.rectangle(preview, (x0, y - 6), (x0 + int((x1-x0)*val), y + 5), (0, 210, 255), -1)
                    cv2.putText(preview, f"{val:.2f}", (x1 - 36, y + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
                _bar(panel_y1 + 38, "LEFT", ai_l)
                _bar(panel_y1 + 60, "RIGHT", ai_r)
                _bar(panel_y1 + 82, "FRONT", ai_front)
                cv2.putText(preview, f"RECOVERY: {'ON' if recovery_started is not None else 'OFF'}",
                            (panel_x1 + 8, panel_y1 + 106), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (0, 220, 255) if recovery_started is not None else (180, 180, 180), 1)

                state = "RUN" if driving else "IDLE"
                lines = [
                    f"FPS {fps:.0f} | {state} | cov {feat['coverage']:.2f} | motion {stuck.motion:.1f} | stall {stall:.2f}",
                    f"steer {smooth_steer:+.2f} | drive {brain.drive} | reward {reward:+.3f} | total {brain.reward_total:+.2f}",
                    f"connectome N={brain.n:,} E={len(brain.w):,} | input={len(brain.input_idx):,} out={len(brain.output_idx):,} DA={len(brain.da_idx):,}",
                    f"active DN: {brain.last_output_ids[:6]}",
                    f"wall L/R {feat['wall_l']:.2f}/{feat['wall_r']:.2f} | front {ai_front:.2f} | plastic {int(brain.plastic_mask.sum()):,}",
                    f"learn R={brain.last_reward:+.3f} total={brain.reward_total:+.2f} | outbias={brain.output_bias:+.3f}",
                ]
                for i, text in enumerate(lines):
                    cv2.putText(preview, text, (6, 20 + i * 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)

                api.update(brain.api_snapshot(
                    feat=feat, stall=stall, reward=reward, driving=driving, fps=fps
                ))

                cv2.imshow("chopari-connectome", preview)
                cv2.imshow("mask", feat["mask"])

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    path = os.path.join(SAVE_DIR, f"shot_{int(now * 1000)}.png")
                    cv2.imwrite(path, frame)
                    print("저장:", path)
                elif key == ord("r"):
                    recording = not recording
                    print("녹화", "시작" if recording else "중지")
                elif key == ord("c"):
                    driving = False
                    joy.release()
                    new_region = select_region(sct)
                    if new_region:
                        cfg["region"] = region = new_region
                        save_config(cfg)

        finally:
            api.stop()
            joy.shutdown()
            brain.save_learning()
            cfg.update({"s_max": cfg.get("s_max", 50), "v_min": cfg.get("v_min", 70), "v_max": cfg.get("v_max", 210)})
            save_config(cfg)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
