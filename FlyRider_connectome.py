# -*- coding: utf-8 -*-
# FlyRider_connectome.py
#
# FlyRider - 실제 Drosophila connectome 기반 제어 버전
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
#       python FlyRider_connectome.py
#
# 영역 재선택:
#       python FlyRider_connectome.py --select
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
#
# ------------------------------------------------------------
# 이 파일에서 이번에 바뀐 부분 (동작 로직은 바꾸지 않음, 디버그 정보만 추가):
#   - RewardSystem: reward 합계 외에 구성요소별 값(last_breakdown)도 기록
#   - ConnectomeBrain.apply_reward: 이번 프레임 weight 변화량/횟수 기록
#   - ConnectomeBrain.api_snapshot: reward_breakdown / wall_ai / recovery / learning 필드 추가
#   - main(): 위 값들을 모아서 API로 전송
# ------------------------------------------------------------

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
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    print(f"[ObjectAI] Ultralytics 로드 실패: {e}")

try:
    import torch
except Exception:
    torch = None

# ============================================================
# 경로 / 기본 설정
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NEURON_CSV = os.path.join(BASE_DIR, "data/neurons.csv")
CONNECTION_CSV = os.path.join(BASE_DIR, "data/connections_princeton.csv")
CONFIG_PATH = os.path.join(BASE_DIR, "data/flyrider_config.json")
OVERNIGHT_PROCESS = "--overnight" in sys.argv
LEARN_SUFFIX = "_overnight" if OVERNIGHT_PROCESS else "_v7"
CACHE_SUFFIX = "_overnight" if OVERNIGHT_PROCESS else "_v7"
LEARN_PATH = os.path.join(BASE_DIR, f"data/flyrider_connectome_learning{LEARN_SUFFIX}.json")
CACHE_PATH = os.path.join(BASE_DIR, f"data/flyrider_connectome_cache{CACHE_SUFFIX}.npz")
OVERNIGHT_LOG_PATH = os.path.join(BASE_DIR, "data/flyrider_overnight_log.jsonl")
SAVE_DIR = os.path.join(BASE_DIR, "captures")

MONITOR_INDEX = 1
PREVIEW_WIDTH = 760
PROC_WIDTH = 320
RECORD_FPS = 10

# 최초 connectome 추출 설정.
# 실제 뉴런/연결을 사용하되, 166k 뉴런/9m edge 전체를 매 5ms Python에서 돌리는 대신
# 실제 그래프에서 모터 출력까지 연결되는 부분을 sparse subgraph로 추출한다.
CACHE_VERSION = 7
MOTOR_DESCENDING = 64
UPSTREAM_PER_TARGET = 32
MAX_UPSTREAM_TOTAL = 14000
MAX_EDGES = 450000
EXPANSION_HOPS = 3

# 실시간 뇌 계산
BRAIN_DT = 0.001      # 1 ms
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
N_INPUT = 12            # road 8 + stall + wall L/R + front wall

# 출력
STEER_SMOOTH = 0.18
# 카트라이더 조이스틱 입력은 연속 조향이 아니라 좌/중립/우의 디지털 입력으로 사용한다.
STEER_ENTER_THRESHOLD = 0.30
STEER_EXIT_THRESHOLD = 0.20
STEER_PULSE_PERIOD = 0.24
STEER_MIN_PULSE = 0.045
STEER_FULL_HOLD_THRESHOLD = 0.90
STEER_INVERT = False
MOTOR_FORWARD_PRESS = 0.25
MOTOR_FORWARD_RELEASE = 0.16
MOTOR_REVERSE_PRESS = 0.60
MOTOR_REVERSE_RELEASE = 0.42
MOTOR_REVERSE_MARGIN = 0.25

VJOY_DEVICE_ID = 1
AXIS_MIN = 0x0001
AXIS_CENTER = 0x4000
AXIS_MAX = 0x8000
# Throttle and reverse use the vJoy Y axis. Steering uses X.
FORWARD_VALUE = AXIS_MIN
REVERSE_VALUE = AXIS_MAX

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
WALL_BASE = 0.4

# 보상. 학습은 connectome의 선택된 실제 edge에만 적용된다.
REWARD_NORMAL = 0.003
REWARD_STEADY = 0.007
REWARD_CORNER = 0.010
REWARD_DRIFT = 0.1
REWARD_LAP = 1.5
REWARD_RECORD = 4.0
PENALTY_WALL = -0.02
PENALTY_COLLISION = -0.04
PENALTY_STUCK = -0.025
PENALTY_RESET = -2.5

LEARNING_RATE = 0.0025
ELIGIBILITY_DECAY = 0.96
MAX_PLASTIC_EDGES = 120000
REWARD_CLIP = 2.5

# 뇌 상태 API
API_HOST = "127.0.0.1"
API_PORT = 8765
API_VERSION = 1

OBJECT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "best.pt")
OBJECT_AI_INTERVAL = 0.10
OBJECT_AI_CONF = 0.10
WALL_NEAR_THRESHOLD = 0.35
STUCK_THRESHOLD = 0.60
SAFE_REWARD_INTERVAL = 3.0
SAFE_REWARD_AMOUNT = 0.08
SAFE_DANGER_THRESHOLD = 0.28
APPROACH_DANGER_THRESHOLD = 0.42
APPROACH_DELTA_THRESHOLD = 0.035
APPROACH_PENALTY = 0.06
STUCK_DANGER_THRESHOLD = 0.70
STUCK_CONFIRM_SEC = 0.75
STUCK_PENALTY_INTERVAL = 3.0
STUCK_PENALTY_AMOUNT = 0.08
ESCAPE_PROGRESS_REWARD = 0.12
REWARD_SMOOTH_TAU = 0.18
REWARD_MAX_DT = 0.10
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


from object_vision import ObjectVision

_ENGINE_MUTEX = None


def acquire_engine_mutex():
    """Only one process may send vJoy input at a time (normal or overnight)."""
    global _ENGINE_MUTEX
    if os.name != "nt":
        return True
    kernel = ctypes.windll.kernel32
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_bool
    handle = kernel.CreateMutexW(None, True, "Local\\FlyRiderConnectomeEngine")
    if not handle:
        print("[engine] 실행 잠금을 만들 수 없어 시작을 취소합니다.")
        return False
    if kernel.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
        kernel.CloseHandle(handle)
        print("[engine] 이미 다른 주행/오버나이트 엔진이 실행 중입니다.")
        return False
    _ENGINE_MUTEX = handle
    return True

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
            "Top in/out region", "Soma side"
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
        visual_sensory = self._ids(self.df["Super Class"].isin(("ol_sensory", "visual_projection")))
        vnc_motor = self._ids(self.df["Super Class"].eq("vnc_motor"))
        da = self._ids(self.df["Predicted NT type"].eq("DA"))
        # DOOMFLY의 좌우 DNp20 차동 디코더를 참고한다. 이 연결은
        # 공학적 출력 매핑이지 확립된 자연 행동 기능을 뜻하지 않는다.
        cell_type = self.df["Primary Cell Type"].fillna("").astype(str)
        soma_side = self.df["Soma side"].fillna("").astype(str).str.lower()
        turn_left = self._ids(cell_type.eq("DNp20") & soma_side.eq("left"))
        turn_right = self._ids(cell_type.eq("DNp20") & soma_side.eq("right"))
        # DOOMFLY-inspired button readouts: DNpe017=forward, MDN=reverse.
        # These are explicit engineering mappings, not claims of native KartRider actions.
        forward_ids = self._ids(cell_type.eq("DNpe017"))
        reverse_ids = self._ids(cell_type.eq("MDN"))
        action_ids = turn_left | turn_right | forward_ids | reverse_ids

        print(f"[connectome] descending={len(desc):,}, visual inputs={len(visual_sensory):,}, "
              f"VNC motor={len(vnc_motor):,}, DA={len(da):,}")

        # 1) descending neuron으로 들어오는 실제 edge를 찾는다.
        incoming = defaultdict(float)
        action_incoming = defaultdict(float)
        print("[connectome] 1/4: descending 입력 연결 탐색...")
        for chunk in pd.read_csv(CONNECTION_CSV, usecols=["pre_root_id", "post_root_id", "syn_count"],
                                 chunksize=400000):
            q = chunk[chunk["post_root_id"].isin(desc)]
            if not q.empty:
                for pre, post, s in zip(q["pre_root_id"].to_numpy(), q["post_root_id"].to_numpy(),
                                        q["syn_count"].to_numpy()):
                    incoming[int(pre)] += float(s)
                    if int(post) in action_ids:
                        action_incoming[int(pre)] += float(s)

        top_desc_inputs = [x for x, _ in sorted(incoming.items(), key=lambda kv: kv[1], reverse=True)
                           [:MOTOR_DESCENDING * UPSTREAM_PER_TARGET]]
        # Preserve actual presynaptic partners for the named motor readouts.
        top_action_inputs = [x for x, _ in sorted(action_incoming.items(), key=lambda kv: kv[1], reverse=True)[:384]]
        top_desc_inputs = list(dict.fromkeys(top_action_inputs + top_desc_inputs))
        # descending 자체도 output으로 유지한다.
        output_ids = list(desc)

        # 실제 descending 중 outgoing synapse가 많은 것만 출력 후보로 사용.
        out_count = Counter()
        motor_out_count = Counter()
        motor_side_count = defaultdict(float)
        print("[connectome] 2/4: descending 출력 세기 계산...")
        for chunk in pd.read_csv(CONNECTION_CSV, usecols=["pre_root_id", "post_root_id", "syn_count"],
                                 chunksize=400000):
            q = chunk[chunk["pre_root_id"].isin(desc)]
            if not q.empty:
                for pre, post, s in zip(q["pre_root_id"].to_numpy(), q["post_root_id"].to_numpy(),
                                        q["syn_count"].to_numpy()):
                    pre_id, post_id, weight = int(pre), int(post), float(s)
                    out_count[pre_id] += weight
                    if post_id in vnc_motor:
                        motor_out_count[pre_id] += weight
                        side = str(self.meta.get(post_id, {}).get("Soma side", "")).lower()
                        if side in ("left", "right"):
                            motor_side_count[(pre_id, side)] += weight
        # Prefer descending neurons with direct, annotated VNC motor targets.
        selected_outputs = [x for x, _ in motor_out_count.most_common(MOTOR_DESCENDING)]
        if len(selected_outputs) < 8:
            print("[connectome] 직접 VNC motor 출력이 부족해 전체 하행 연결로 대체합니다.")
            selected_outputs = [x for x, _ in out_count.most_common(MOTOR_DESCENDING)]
        if len(selected_outputs) < 8:
            selected_outputs = list(desc)[:MOTOR_DESCENDING]
        # Keep named turn/forward/reverse cells as explicit action readouts,
        # even when their direct motor targets are absent from the reduced graph.
        selected_outputs = list(dict.fromkeys(selected_outputs + sorted(action_ids)))

        # 3) 출력 후보에서 upstream을 여러 단계 확장.
        # Keep actual VNC motor targets in the simulated graph so selected
        # descending outputs retain their downstream connectome path.
        nodes = set(selected_outputs) | set(top_desc_inputs) | set(vnc_motor)
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

        # Camera features enter the visual pathway, not olfactory or VNC sensory cells.
        input_candidates = [nid for nid in nodes if nid in visual_sensory]
        if len(input_candidates) < 64:
            # If graph expansion found too few, add only annotated visual neurons.
            input_candidates = sorted(visual_sensory)[:min(2000, len(visual_sensory))]
            nodes.update(input_candidates)
        input_candidates = self._balanced_visual_candidates(input_candidates, 2500)

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

        # Annotated visual laterality gives a coarse, deterministic screen map.
        input_channels = self._visual_input_channels(input_ids)
        output_signs = []
        for nid in selected_outputs:
            left = motor_side_count.get((nid, "left"), 0.0)
            right = motor_side_count.get((nid, "right"), 0.0)
            total = left + right
            output_signs.append((right - left) / total if total > 0 else 0.0)
        output_signs = np.asarray(output_signs, dtype=np.float32)

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
            output_signs=output_signs,
            turn_left_ids=np.asarray(sorted(turn_left.intersection(output_ids)), dtype=np.int64),
            turn_right_ids=np.asarray(sorted(turn_right.intersection(output_ids)), dtype=np.int64),
            forward_ids=np.asarray(sorted(forward_ids.intersection(output_ids)), dtype=np.int64),
            reverse_ids=np.asarray(sorted(reverse_ids.intersection(output_ids)), dtype=np.int64),
            da_ids=da_ids,
        )
        print(f"[connectome] cache 저장: nodes={len(node_ids):,}, edges={len(w):,}, "
              f"visual inputs={len(input_ids):,}, motor-linked outputs={len(output_ids):,}, DA={len(da_ids):,}")
        return self._load_cache(np.load(CACHE_PATH, allow_pickle=False))

    def _balanced_visual_candidates(self, candidates, limit):
        """Choose a deterministic, cell-type/side-balanced visual subset."""
        groups = defaultdict(list)
        for nid in sorted(set(candidates)):
            meta = self.meta.get(nid, {})
            key = (str(meta.get("Super Class", "")), str(meta.get("Soma side", "")),
                   str(meta.get("Primary Cell Type", "")), str(meta.get("Sub Class", "")))
            groups[key].append(nid)
        keys = sorted(groups)
        selected = []
        offset = 0
        while len(selected) < limit:
            added = False
            for key in keys:
                if offset < len(groups[key]):
                    selected.append(groups[key][offset])
                    added = True
                    if len(selected) >= limit:
                        break
            if not added:
                break
            offset += 1
        return selected

    def _visual_input_channels(self, input_ids):
        """Use a coarse soma-side prior; MCNS does not supply measured screen RFs."""
        channels_by_side = {
            "left": (0, 1, 2, 3, 8, 9, 11),
            "right": (4, 5, 6, 7, 8, 10, 11),
            "center": (3, 4, 8, 11),
            "unknown": tuple(range(8)),
        }
        counts = Counter()
        channels = np.empty(len(input_ids), dtype=np.int16)
        for i, raw_id in enumerate(input_ids):
            nid = int(raw_id)
            meta = self.meta.get(nid, {})
            side = str(meta.get("Soma side", "")).lower()
            if side not in channels_by_side:
                side = "unknown"
            super_class = str(meta.get("Super Class", ""))
            pool = channels_by_side[side]
            key = (side, super_class)
            channels[i] = pool[counts[key] % len(pool)]
            counts[key] += 1
        return channels

    def _load_cache(self, z):
        return {
            "node_ids": z["node_ids"].astype(np.int64),
            "pre": z["pre"].astype(np.int32),
            "post": z["post"].astype(np.int32),
            "weight": z["weight"].astype(np.float32),
            "input_ids": z["input_ids"].astype(np.int64),
            "input_channels": z["input_channels"].astype(np.int16),
            "output_ids": z["output_ids"].astype(np.int64),
            "output_signs": z["output_signs"].astype(np.float32),
            "turn_left_ids": z["turn_left_ids"].astype(np.int64),
            "turn_right_ids": z["turn_right_ids"].astype(np.int64),
            "forward_ids": z["forward_ids"].astype(np.int64),
            "reverse_ids": z["reverse_ids"].astype(np.int64),
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
        self.drive = "idle"
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

        # 학습 디버그용 통계 (동작에는 영향 없음, 시각화 전용)
        self.total_weight_updates = 0
        self.last_update_count = 0
        self.last_delta_mean = 0.0
        self.last_delta_max = 0.0

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
        self.output_sign = g["output_signs"]
        self.turn_left_ids = g["turn_left_ids"]
        self.turn_right_ids = g["turn_right_ids"]
        self.forward_ids = g["forward_ids"]
        self.reverse_ids = g["reverse_ids"]
        self.da_ids = g["da_ids"]

        self.id_to_idx = {int(x): i for i, x in enumerate(self.node_ids.tolist())}
        self.input_idx = np.asarray([self.id_to_idx[int(x)] for x in self.input_ids if int(x) in self.id_to_idx], dtype=np.int32)
        self.output_idx = np.asarray([self.id_to_idx[int(x)] for x in self.output_ids if int(x) in self.id_to_idx], dtype=np.int32)
        self.turn_left_idx = np.asarray([self.id_to_idx[int(x)] for x in self.turn_left_ids if int(x) in self.id_to_idx], dtype=np.int32)
        self.turn_right_idx = np.asarray([self.id_to_idx[int(x)] for x in self.turn_right_ids if int(x) in self.id_to_idx], dtype=np.int32)
        self.forward_idx = np.asarray([self.id_to_idx[int(x)] for x in self.forward_ids if int(x) in self.id_to_idx], dtype=np.int32)
        self.reverse_idx = np.asarray([self.id_to_idx[int(x)] for x in self.reverse_ids if int(x) in self.id_to_idx], dtype=np.int32)
        self.turn_readout_source = ("DNp20" if len(self.turn_left_idx) and len(self.turn_right_idx)
                                    else "VNC")
        self.da_idx = np.asarray([self.id_to_idx[int(x)] for x in self.da_ids if int(x) in self.id_to_idx], dtype=np.int32)

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
        print(f"[뇌] throttle readout=DNpe017 {len(self.forward_idx)} / MDN {len(self.reverse_idx)}")
        if len(self.turn_left_idx) and len(self.turn_right_idx):
            print(f"[뇌] 조향 readout=좌우 DNp20 ({len(self.turn_left_idx)}/{len(self.turn_right_idx)})")
        else:
            print("[뇌] 조향 readout=VNC 좌우 연결 (DNp20 annotation 없음)")
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
        self.drive = "idle"
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
            # Signed value signal: positive reward excites DA activity and
            # negative wall/stuck feedback suppresses it by the same amount.
            # Wall/stall remain separate sensory inputs to the connectome.
            self.syn[self.da_idx] += float(np.clip(reward_drive, -1.0, 1.0)) * 0.75

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
            return 0.0, "idle", []

        r = self.rate[self.output_idx]
        if len(self.turn_left_idx) and len(self.turn_right_idx):
            # DNp20 right-minus-left is a dedicated, inspectable steering channel.
            left = float(np.mean(self.rate[self.turn_left_idx]))
            right = float(np.mean(self.rate[self.turn_right_idx]))
        else:
            # Fall back to VNC laterality where the dataset lacks annotated DNp20 cells.
            left_weights = np.clip(-self.output_sign, 0.0, 1.0)
            right_weights = np.clip(self.output_sign, 0.0, 1.0)
            left_total, right_total = float(left_weights.sum()), float(right_weights.sum())
            left = float(np.dot(r, left_weights) / left_total) if left_total > 0 else 0.0
            right = float(np.dot(r, right_weights) / right_total) if right_total > 0 else 0.0
        forward = (float(np.max(self.rate[self.forward_idx])) if len(self.forward_idx)
                   else float(np.max(r)))
        reverse = (float(np.max(self.rate[self.reverse_idx])) if len(self.reverse_idx)
                   else float(np.percentile(r, 90)))

        self.last_left_rate = left
        self.last_right_rate = right
        self.last_forward_rate = forward
        self.last_reverse_rate = reverse

        # 직선/저위험 구간에서는 정규화된 좌우 차이를 천천히 0으로 보정한다.
        # 현재 CSV에는 descending neuron의 좌/우 운동 기능 annotation이 없고 Root ID 홀짝은
        # 생물학적 좌/우 근거가 아니므로, 이 보정은 '모터 인터페이스 영점'일 뿐 connectome 자체의 재배선이 아니다.
        total_lateral = left + right
        lateral_balance = ((right - left) / max(total_lateral, 1e-5)
                           if total_lateral > 1e-5 else 0.0)
        if max(wall_l, wall_r) < 0.22 and stall < 0.18:
            self.output_bias = ((1.0 - OUTPUT_BALANCE_ALPHA) * self.output_bias
                                 + OUTPUT_BALANCE_ALPHA * lateral_balance)

        # Output rates are usually small, so normalize before the vJoy output.
        # 신경 출력의 좌우 차이는 작으므로 디지털 입력 문턱을 넘도록 정규화한다.
        steer_raw = np.tanh((lateral_balance - self.output_bias) * 12.0)

        # Hold a throttle direction while its connectome output remains active.
        # Once the filtered firing rate decays, Y returns to neutral.
        if self.drive == "forward" and forward >= MOTOR_FORWARD_RELEASE:
            drive = ("reverse" if reverse >= MOTOR_REVERSE_PRESS
                     and reverse > forward + MOTOR_REVERSE_MARGIN
                     else "forward")
        elif self.drive == "reverse" and reverse >= MOTOR_REVERSE_RELEASE:
            # While backing, prefer returning to forward as soon as its readout
            # is credible; reverse requires a high and clearly dominant signal.
            drive = ("reverse" if reverse >= MOTOR_REVERSE_PRESS
                     and reverse > forward + MOTOR_REVERSE_MARGIN
                     else "forward" if forward >= MOTOR_FORWARD_PRESS
                     else "reverse")
        elif forward >= MOTOR_FORWARD_PRESS:
            # Forward wins ambiguous cases; reverse needs the stricter threshold
            # and a clear margin over the forward population readout.
            drive = ("reverse" if reverse >= MOTOR_REVERSE_PRESS
                     and reverse > forward + MOTOR_REVERSE_MARGIN
                     else "forward")
        else:
            # Forward locomotion is the default motor state. Only a clear,
            # high-confidence MDN signal should replace it with reverse.
            drive = "forward"
        active = self.output_idx[self.rate[self.output_idx] > 0.25]
        ids = [int(self.node_ids[i]) for i in active[:12]]
        return float(np.clip(steer_raw, -1, 1)), drive, ids

    def update(self, now, road_cols, wall_l, wall_r, stall, reward=0.0, front_wall=0.0):
        x = np.zeros(N_INPUT, dtype=np.float32)
        x[:N_COL] = np.clip(road_cols, 0, 1)
        x[8] = np.clip(stall, 0, 1)
        x[9] = np.clip(wall_l, 0, 1)
        x[10] = np.clip(wall_r, 0, 1)
        x[11] = np.clip(front_wall, 0, 1)

        # 실제 connectome의 update 속도를 유지하면서 frame 간 시간만큼 계산.
        dt = BRAIN_DT if not hasattr(self, "last_t") or self.last_t is None else max(0.001, now - self.last_t)
        self.last_t = now
        steps = int(np.clip(round(dt / BRAIN_DT), 1, MAX_STEPS_PER_UPDATE))
        # Apply the same signed feedback both to DA current and plasticity.
        reward_drive = float(np.clip(reward, -1.0, 1.0))
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
            self.last_update_count = 0
            self.last_delta_mean = 0.0
            self.last_delta_max = 0.0
            return
        pidx = np.flatnonzero(self.plastic_mask)
        prev_w = self.w[pidx].copy()
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

        # ---- 학습 디버그 통계 (가중치 갱신 로직 자체에는 영향 없음) ----
        applied = self.w[pidx] - prev_w
        touched = np.abs(applied) > 1e-9
        count = int(touched.sum())
        self.last_update_count = count
        self.total_weight_updates += count
        self.last_delta_mean = float(np.mean(np.abs(applied))) if len(applied) else 0.0
        self.last_delta_max = float(np.max(np.abs(applied))) if len(applied) else 0.0

    def api_snapshot(self, feat=None, stall=0.0, reward=0.0, driving=False, fps=0.0,
                      reward_breakdown=None, wall_ai=None, recovery=False, controller=None,
                      reward_context=None):
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

        pidx_plastic = np.flatnonzero(self.plastic_mask)
        if len(pidx_plastic):
            w_mean = float(np.mean(self.w[pidx_plastic]))
            w_std = float(np.std(self.w[pidx_plastic]))
        else:
            w_mean = 0.0
            w_std = 0.0

        return {
            "api_version": API_VERSION,
            "time": time.time(),
            "fps": float(fps),
            "driving": bool(driving),
            "steer": float(getattr(self, "steer", 0.0)),
            "turn_readout": {
                "source": self.turn_readout_source,
                "left_rate": float(self.last_left_rate),
                "right_rate": float(self.last_right_rate),
                "balance": float(self.last_right_rate - self.last_left_rate),
            },
            "motor_readout": {
                "forward_rate": float(self.last_forward_rate),
                "reverse_rate": float(self.last_reverse_rate),
                "forward_cells": int(len(self.forward_idx)),
                "reverse_cells": int(len(self.reverse_idx)),
            },
            "drive": str(getattr(self, "drive", "idle")),
            "controller": dict(controller) if controller else {},
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
                "front_wall": float(feat.get("front_wall", 0.0) if feat is not None else 0.0),
                "coverage": float(feat["coverage"] if feat is not None else 0.0),
                "stall": float(stall),
            },
            # ---- 이번에 추가된 디버그 필드 ----
            "reward_breakdown": dict(reward_breakdown) if reward_breakdown else {},
            "reward_context": dict(reward_context) if reward_context else {},
            "wall_ai": dict(wall_ai) if wall_ai else {},
            "recovery": bool(recovery),
            "learning": {
                "plastic_edges": int(self.plastic_mask.sum()),
                "updates_this_frame": int(self.last_update_count),
                "total_updates": int(self.total_weight_updates),
                "avg_delta": float(self.last_delta_mean),
                "max_delta": float(self.last_delta_max),
                "weight_mean": w_mean,
                "weight_std": w_std,
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
        self.prev_danger = None
        self.prev_stuck = False
        self.stuck_since = None
        self.last_stuck_penalty_at = None
        self.prev_time = None
        self.filtered_danger = None
        self.safe_since = None
        self.last_safe_reward_at = None
        self.total = 0.0
        self.last_event = ""
        self.last_breakdown = {}
        self.last_debug = {}
        self.last_steer = 0.0

    def step(self, feat, stall, drive, speed_hint=0.0, front_block=0.0, now=None):
        now = time.time() if now is None else now
        self.last_event = ""
        left, right, front = float(feat.get("wall_l", 0)), float(feat.get("wall_r", 0)), float(front_block)
        wall_ai = feat.get("wall_ai", {})
        # Keep the image-based side sensors as a collision-only backup. They are
        # noisy while driving, so use them only after prolonged stillness.
        sensor_ready = bool(wall_ai.get("ready", False) or stall >= STUCK_THRESHOLD)
        backup = max(float(feat.get("vision_wall_l", 0.0)), float(feat.get("vision_wall_r", 0.0)))
        raw_danger = float(np.clip(max(left, right, front,
                                       backup if stall >= STUCK_THRESHOLD else 0.0), 0, 1)) if sensor_ready else 0.0
        dt = (1.0 / 60.0 if self.prev_time is None else
              float(np.clip(now - self.prev_time, 0.0, REWARD_MAX_DT)))
        self.prev_time = now
        if self.filtered_danger is None:
            self.filtered_danger = raw_danger
        else:
            alpha = 1.0 - np.exp(-dt / max(1e-3, REWARD_SMOOTH_TAU))
            self.filtered_danger += alpha * (raw_danger - self.filtered_danger)
        danger = float(np.clip(self.filtered_danger, 0.0, 1.0))
        delta = 0.0 if self.prev_danger is None else danger - self.prev_danger
        moving = float(speed_hint) >= STUCK_DIFF_THRESHOLD
        stuck = bool(sensor_ready and stall >= STUCK_THRESHOLD
                     and danger >= STUCK_DANGER_THRESHOLD)
        breakdown = {"approach": 0.0, "escape_success": 0.0,
                     "stuck": 0.0, "safe_drive": 0.0}

        # A wall detection by itself is not a penalty. Penalize only a clear,
        # sustained increase in the trained object's wall-proximity signal.
        if (sensor_ready and moving and danger >= APPROACH_DANGER_THRESHOLD
                and delta >= APPROACH_DELTA_THRESHOLD):
            strength = float(np.clip(delta / 0.12, 0.25, 1.0))
            breakdown["approach"] = -APPROACH_PENALTY * strength
            self.last_event = "approaching_wall"

        # Confirm a collision/stuck state before giving occasional penalties;
        # this avoids turning one still frame or one noisy mask into punishment.
        if stuck:
            if self.stuck_since is None:
                self.stuck_since = now
            elif (now - self.stuck_since >= STUCK_CONFIRM_SEC
                  and (self.last_stuck_penalty_at is None
                       or now - self.last_stuck_penalty_at >= STUCK_PENALTY_INTERVAL)):
                breakdown["stuck"] = -STUCK_PENALTY_AMOUNT
                self.last_stuck_penalty_at = now
                self.last_event = "stuck_at_wall"
        elif self.prev_stuck and self.stuck_since is not None:
            if moving and danger < SAFE_DANGER_THRESHOLD:
                breakdown["escape_success"] = ESCAPE_PROGRESS_REWARD
                self.last_event = "escape_success"
            self.stuck_since = None
            self.last_stuck_penalty_at = None

        # Reward sustained, moving, forward travel every three seconds.
        # Standing still receives neither this reward nor a proximity penalty.
        safe_drive = (sensor_ready and drive == "forward" and moving
                      and danger < SAFE_DANGER_THRESHOLD and stall < 0.2)
        if safe_drive:
            if self.safe_since is None:
                self.safe_since = now
                self.last_safe_reward_at = now
            elif now - self.last_safe_reward_at >= SAFE_REWARD_INTERVAL:
                breakdown["safe_drive"] = SAFE_REWARD_AMOUNT
                self.last_safe_reward_at = now
                self.last_event = "safe_drive_milestone"
        else:
            self.safe_since = None
            self.last_safe_reward_at = None

        reward = float(np.clip(sum(breakdown.values()), -0.15, 0.20))
        self.prev_danger, self.prev_stuck = danger, stuck
        self.total += reward
        self.last_breakdown = breakdown
        self.last_debug = {
            "sensor_ready": sensor_ready,
            "danger": danger,
            "danger_delta": delta,
            "moving": moving,
            "safe_seconds": 0.0 if self.safe_since is None else max(0.0, now - self.safe_since),
            "stuck": stuck,
            "event": self.last_event,
        }
        return float(reward)

    def collision(self):
        return 0.0

    def reset(self):
        self.prev_danger = None
        self.prev_stuck = False
        self.stuck_since = None
        self.last_stuck_penalty_at = None
        self.prev_time = None
        self.filtered_danger = None
        self.safe_since = None
        self.last_safe_reward_at = None
        self.last_event = ""
        self.last_breakdown = {}
        self.last_debug = {}
        return 0.0

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
        self.steer_button = 0
        self.steer_active = False
        self.steer_direction = 0
        self.steer_phase_started = 0.0
        self.output_turn = 0
        self.output_drive = "idle"
        self.output_x_value = AXIS_CENTER
        self.output_y_value = AXIS_CENTER
        self.last_x = None
        self.last_y = None
        self.running = False
        if pyvjoy is None:
            print("[vJoy] pyvjoy가 없어 출력 비활성화")
            return
        try:
            self.dev = pyvjoy.VJoyDevice(device_id)
            self.dev.reset()
            self.dev.set_axis(pyvjoy.HID_USAGE_X, AXIS_CENTER)
            self.dev.set_axis(pyvjoy.HID_USAGE_Y, AXIS_CENTER)
            print(f"[vJoy] 장치 열기 및 X/Y 중립 출력 성공 (X={AXIS_CENTER}, Y={AXIS_CENTER})")
        except Exception as e:
            print("[vJoy] 장치 열기 실패:", e)
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _button_steer(self, steer, now):
        """작은 조향 출력은 좌/우 펄스로, 큰 출력은 계속 누른다."""
        magnitude = abs(steer)
        direction = 1 if steer > 0 else -1 if steer < 0 else 0
        if self.steer_active:
            if direction != self.steer_direction or magnitude < STEER_EXIT_THRESHOLD:
                self.steer_active = False
        elif magnitude >= STEER_ENTER_THRESHOLD:
            self.steer_active = True
            self.steer_direction = direction
            self.steer_phase_started = now

        if not self.steer_active:
            self.steer_button = 0
            return 0
        if direction != self.steer_direction:
            self.steer_direction = direction
            self.steer_phase_started = now

        if magnitude >= STEER_FULL_HOLD_THRESHOLD:
            self.steer_button = self.steer_direction
            return self.steer_button
        duty = float(np.clip(magnitude / STEER_FULL_HOLD_THRESHOLD,
                             STEER_MIN_PULSE / STEER_PULSE_PERIOD, 0.98))
        phase = (now - self.steer_phase_started) % STEER_PULSE_PERIOD
        self.steer_button = self.steer_direction if phase < duty * STEER_PULSE_PERIOD else 0
        return self.steer_button

    def _loop(self):
        while self.running:
            now = time.time()
            with self.lock:
                steer, drive, force = self.steer, self.drive, self.force_dir
            d = force if force is not None else self._button_steer(steer, now)
            if STEER_INVERT:
                d = -d
            self.output_turn = d
            self.output_drive = drive
            x = AXIS_CENTER if d == 0 else (AXIS_MIN if d < 0 else AXIS_MAX)
            y = {"forward": FORWARD_VALUE, "reverse": REVERSE_VALUE}.get(drive, AXIS_CENTER)
            self.output_x_value = x
            self.output_y_value = y
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

    def pulse_button(self, button=8):
        if self.dev is None:
            return False
        try:
            self.dev.set_button(int(button), 1)
            time.sleep(0.12)
            self.dev.set_button(int(button), 0)
            return True
        except Exception as exc:
            print(f"[vJoy] 버튼 {button} 출력 실패: {exc}")
            return False

    def status(self):
        return {
            "turn": int(self.output_turn),
            "drive": str(self.output_drive),
            "drive_level": 1 if self.output_drive == "forward" else -1 if self.output_drive == "reverse" else 0,
            "x_value": int(self.output_x_value),
            "y_value": int(self.output_y_value),
            "available": bool(self.dev is not None and self.running),
        }

    def shutdown(self):
        self.release()
        time.sleep(0.05)
        self.running = False
        if self.dev is not None:
            try:
                self.dev.reset()
            except Exception:
                pass


def send_rescue_key():
    """Best-effort no-install R key path; game must be focused and accept synthetic input."""
    try:
        user32 = ctypes.windll.user32
        user32.keybd_event(0x52, 0, 0, 0)       # R down (SendInput-compatible user path)
        time.sleep(0.08)
        user32.keybd_event(0x52, 0, 0x0002, 0)  # R up
        return True
    except Exception as exc:
        print(f"[recovery] R 키 전송 불가: {exc}")
        return False


def send_rescue_action(joy):
    config_path = os.path.join(BASE_DIR, "data", "flyrider_rescue_mode.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            mode = json.load(f).get("mode", "keyboard")
    except Exception:
        mode = "keyboard"
    if mode == "vjoy_button8":
        return "vJoy button 8", joy.pulse_button(8)
    return "Windows R key", send_rescue_key()


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
    server_version = "flyriderBrainAPI/1.0"

    def log_message(self, fmt, *args):
        # 로그 메시지를 무시하여 콘솔 출력을 줄임
        pass

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/state":
            self._handle_state_request()
        elif path == "/health":
            self._handle_health_request()
        else:
            self._handle_not_found()

    def _handle_state_request(self):
        with self.server.state_lock:
            payload = dict(self.server.state)
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send_response(200, "application/json; charset=utf-8", raw)

    def _handle_health_request(self):
        raw = b'{"ok":true}'
        self._send_response(200, "application/json", raw)

    def _handle_not_found(self):
        self.send_response(404)
        self.end_headers()

    def _send_response(self, status_code, content_type, content):
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


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
        print(f"[API] Brain state available at: http://{API_HOST}:{API_PORT}/state")
        print(f"[API] Health check available at: http://{API_HOST}:{API_PORT}/health")

    def update(self, state):
        with self.state_lock:
            self.state.clear()
            self.state.update(state)

    def stop(self):
        try:
            self.httpd.shutdown()
            self.httpd.server_close()
        except Exception as e:
            print(f"[API] Error during shutdown: {e}")

# ============================================================
# 메인
# ============================================================
def main():
    if not acquire_engine_mutex():
        return
    os.makedirs(SAVE_DIR, exist_ok=True)
    cfg = load_config()
    overnight = "--overnight" in sys.argv
    run_hours = 12.0
    for arg in sys.argv:
        if arg.startswith("--hours="):
            try:
                run_hours = max(0.1, float(arg.split("=", 1)[1]))
            except ValueError:
                print("[overnight] --hours 값이 잘못되어 기본 12시간을 사용합니다.")
    run_deadline = time.time() + run_hours * 3600.0 if overnight else None
    overnight_started = time.time()
    episode = 0
    reset_count = 0

    def overnight_log(event, **values):
        if not overnight:
            return
        record = {"time": time.time(), "event": event, "episode": episode,
                  "reset_count": reset_count, **values}
        try:
            with open(OVERNIGHT_LOG_PATH, "a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            print(f"[overnight] 로그 저장 실패: {exc}")

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

        object_ai = ObjectVision(OBJECT_MODEL_PATH, conf=OBJECT_AI_CONF)
        object_result = {"left_wall": 0.0, "right_wall": 0.0, "front_wall": 0.0,
                         "drifting": 0.0, "masks": [], "ready": False}
        object_ai_last = 0.0
        if object_ai.loaded:
            print(f"[ObjectAI] 세그멘테이션 모델 연결: {OBJECT_MODEL_PATH}")
        else:
            print(f"[ObjectAI] 모델 비활성화: {object_ai.error}")

        print(f"[preflight] vJoy={'OK' if joy.dev is not None else '사용 불가'} · X/Y 축 중립 출력 완료")
        try:
            with open(os.path.join(BASE_DIR, "data", "flyrider_rescue_mode.json"), "r", encoding="utf-8") as f:
                rescue_mode = json.load(f).get("mode", "keyboard")
        except Exception:
            rescue_mode = "keyboard"
        keyboard_api_ready = hasattr(ctypes.windll.user32, "keybd_event")
        route_name = "vJoy 버튼 8" if rescue_mode == "vjoy_button8" else "Windows R 키"
        print(f"[preflight] 복구 경로={route_name} · Windows 키 API={'OK' if keyboard_api_ready else '사용 불가'} · 실제 게임 수신은 주행 중 움직임으로 확인")
        api.start()

        hk_drive = Hotkey(0x77)  # F8
        hk_stop = Hotkey(0x78)   # F9
        hk_rebuild = Hotkey(0x79)  # F10
        hk_stats = Hotkey(0x7A)   # F11
        hk_capture = Hotkey(0x7B) # F12: collect frame for later manual labeling
        hk_debug = Hotkey(0x76)   # F7: toggle low-resolution Object AI debug
        object_ai.debug_enabled = "--debug-window" in sys.argv
        debug_window_placed = False
        inbox_dir = os.path.join(BASE_DIR, "ai_traing", "data", "state_ai", "inbox")
        os.makedirs(inbox_dir, exist_ok=True)

        driving = overnight or "--autostart" in sys.argv
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
        last_learn_save = time.time()
        previous_wall_danger = 0.0
        previous_stall = 0.0
        stuck_reset_at = 0.0
        recovery_phase = 0  # 0 idle, 1 brain reset wait, 2 R sent / verify motion
        recovery_started = 0.0
        recovery_failures = 0
        stop_request_path = os.path.join(BASE_DIR, "data", "flyrider_stop.request")
        capture_request_path = os.path.join(BASE_DIR, "data", "flyrider_capture.request")
        debug_request_path = os.path.join(BASE_DIR, "data", "flyrider_debug_toggle.request")
        drive_request_path = os.path.join(BASE_DIR, "data", "flyrider_drive_toggle.request")
        immediate_stop_path = os.path.join(BASE_DIR, "data", "flyrider_immediate_stop.request")
        if overnight:
            episode = 1
            grace_until = time.time() + STUCK_GRACE
            stuck.reset()
            rewarder.reset()
            brain.reset_state()
            overnight_log("start", hours=run_hours)
            print(f"[overnight] 자동 주행 시작 ({run_hours:.2f}시간 제한)")

        print("=" * 70)
        print("초파리라이더 REAL CONNECTOME")
        print("F8=주행 / F9=정지 / F10=cache 재생성 / F11=통계")
        print("q=종료 / s=스크린샷 / r=녹화 / c=영역 재선택")
        if overnight:
            print(f"자동 학습 모드: {run_hours:.2f}시간 / 로그: {OVERNIGHT_LOG_PATH}")
        print("=" * 70)

        try:
            while True:
                now = time.time()
                if os.path.exists(stop_request_path):
                    print("[overnight] GUI 정지 요청: 저장 후 안전 종료합니다.")
                    overnight_log("stop_requested", reward_total=brain.reward_total)
                    break
                if overnight and run_deadline is not None and now >= run_deadline:
                    overnight_log("deadline", elapsed=now - overnight_started,
                                  reward_total=brain.reward_total)
                    print("[overnight] 지정 시간이 끝나 안전하게 종료합니다.")
                    break
                frame = cv2.cvtColor(np.array(sct.grab(region)), cv2.COLOR_BGRA2BGR)
                if os.path.exists(capture_request_path):
                    try: os.remove(capture_request_path)
                    except OSError: pass
                    capture_path = os.path.join(BASE_DIR, "ai_traing", "data", "state_ai", "inbox", f"drive_{int(now*1000)}.jpg")
                    os.makedirs(os.path.dirname(capture_path), exist_ok=True)
                    cv2.imwrite(capture_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    print(f"[ObjectAI] GUI 장면 수집: {capture_path}")
                if os.path.exists(debug_request_path):
                    try: os.remove(debug_request_path)
                    except OSError: pass
                    object_ai.debug_enabled = not object_ai.debug_enabled
                    print(f"[ObjectAI] 디버그 오버레이 {'ON' if object_ai.debug_enabled else 'OFF'}")
                if os.path.exists(immediate_stop_path):
                    try: os.remove(immediate_stop_path)
                    except OSError: pass
                    driving = False; joy.release()
                    print("[UI] 즉시 정지: vJoy 입력을 해제했습니다.")
                elif os.path.exists(drive_request_path):
                    try: os.remove(drive_request_path)
                    except OSError: pass
                    driving = not driving
                    if driving:
                        grace_until = now + STUCK_GRACE; stuck.reset(); rewarder.reset(); brain.reset_state()
                    else: joy.release()
                    print(f"[UI] 주행 {'시작' if driving else '정지'}")
                feat = extract_features(frame, cfg.get("s_max", 50), cfg.get("v_min", 70), cfg.get("v_max", 210))
                feat["vision_wall_l"] = feat["wall_l"]
                feat["vision_wall_r"] = feat["wall_r"]

                # --------------------------------------------------------
                # YOLO segmentation -> continuous connectome sensory interface
                # --------------------------------------------------------
                if object_ai.loaded and (now - object_ai_last >= OBJECT_AI_INTERVAL):
                    object_ai_last = now
                    try:
                        object_result = object_ai.analyze(frame)
                    except Exception as e:
                        print(f"[ObjectAI] 추론 오류: {e}")
                        object_result = {"left_wall": 0.0, "right_wall": 0.0,
                                         "front_wall": 0.0, "drifting": 0.0,
                                         "masks": [], "ready": False}
                ai_fresh = bool(object_result.get("ready") and not object_result.get("stale"))
                ai_l = float(np.clip(object_result.get("left_wall", 0.0), 0.0, 1.0)) if ai_fresh else float(feat["wall_l"])
                ai_r = float(np.clip(object_result.get("right_wall", 0.0), 0.0, 1.0)) if ai_fresh else float(feat["wall_r"])
                ai_front = float(np.clip(object_result.get("front_wall", 0.0), 0.0, 1.0)) if ai_fresh else 0.0

                if object_ai.loaded and ai_fresh:
                    feat["wall_l"] = ai_l
                    feat["wall_r"] = ai_r
                feat["wall_ai"] = object_result
                feat["front_wall"] = ai_front
                feat["drifting"] = float(object_result.get("drifting", 0.0)) if ai_fresh else 0.0

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
                        rewarder.reset()
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

                if hk_debug.pressed():
                    object_ai.debug_enabled = not object_ai.debug_enabled
                    print(f"[ObjectAI] 디버그 오버레이 {'ON' if object_ai.debug_enabled else 'OFF'}")
                if hk_capture.pressed():
                    capture_path = os.path.join(inbox_dir, f"drive_{int(now*1000)}.jpg")
                    cv2.imwrite(capture_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    print(f"[ObjectAI] 라벨링 대기 장면 저장: {capture_path}")

                reward = 0.0
                if driving:
                    # Wall/front/stall are sensory channels. Motor direction and
                    # steering are produced only by the connectome output.
                    sensory_stall = float(np.clip(stall, 0.0, 1.0))
                    reward = rewarder.step(feat, sensory_stall, brain.drive, stuck.motion,
                                           front_block=ai_front, now=now)
                    steer, drive = brain.update(
                        now,
                        feat["road"], feat["wall_l"], feat["wall_r"], sensory_stall,
                        reward=reward, front_wall=ai_front,
                    )
                    rewarder.set_steer(steer)
                    smooth_steer = (1.0 - STEER_SMOOTH) * smooth_steer + STEER_SMOOTH * steer
                    joy.command(smooth_steer, drive)
                else:
                    joy.release()

                # Confirm both prolonged stillness and wall proximity before recovery.
                backup_wall = max(feat.get("vision_wall_l", 0.0), feat.get("vision_wall_r", 0.0))
                wall_contact = max(ai_l, ai_r, ai_front,
                                   backup_wall if stall >= STUCK_THRESHOLD else 0.0) >= WALL_NEAR_THRESHOLD
                stuck_at_wall = bool(driving and stall >= 0.95 and wall_contact)
                if overnight and stuck_at_wall:
                    if recovery_phase == 0:
                        if stuck_reset_at <= 0.0: stuck_reset_at = now
                        elif now - stuck_reset_at >= 2.0:
                            print("[recovery] 벽 + 정지 확인. 먼저 connectome 내부 상태를 초기화합니다.")
                            joy.release(); brain.reset_state(); rewarder.reset(); stuck.reset()
                            smooth_steer = 0.0; recovery_phase = 1; recovery_started = now
                            grace_until = now + 2.0
                            overnight_log("brain_reset_attempt", wall=max(ai_l,ai_r,ai_front))
                    elif recovery_phase == 1 and now - recovery_started >= 2.0 and stall >= 0.8:
                        route, sent = send_rescue_action(joy)
                        print(f"[recovery] {route} 전송 시도: {'호출 성공' if sent else '실패'}; 화면 움직임을 확인합니다.")
                        recovery_phase = 2; recovery_started = now; stuck.reset(); grace_until = now
                        overnight_log("r_key_attempt", sent=sent)
                    elif recovery_phase == 2:
                        moved = stuck.motion >= STUCK_DIFF_THRESHOLD or stall < 0.5
                        if moved:
                            recovery_phase = 0; stuck_reset_at = 0.0
                            grace_until = now + STUCK_GRACE
                            episode += 1; reset_count += 1
                            overnight_log("recovered", motion=stuck.motion, stall=stall)
                            print(f"[recovery] 화면 움직임 확인. 주행 재개 episode={episode}")
                        elif now - recovery_started >= 3.0:
                            recovery_failures += 1; recovery_phase = 0; stuck_reset_at = 0.0
                            overnight_log("recovery_failed", failures=recovery_failures, motion=stuck.motion)
                            if recovery_failures >= 2:
                                print("[recovery] 2회 복구 실패. 입력을 해제하고 저장 후 안전 정지합니다.")
                                joy.release(); brain.save_learning(); driving = False
                                overnight_log("safe_stop", reason="recovery_failed")
                                break
                            stuck.reset(); grace_until = now + 0.5
                else:
                    stuck_reset_at = 0.0
                    if recovery_phase == 1 and now - recovery_started >= 0.8 and stuck.motion >= STUCK_DIFF_THRESHOLD:
                        recovery_phase = 0; episode += 1; reset_count += 1
                        grace_until = now + STUCK_GRACE
                        overnight_log("brain_reset_recovered", motion=stuck.motion)
                        print("[recovery] 뇌 상태 초기화만으로 화면 움직임이 회복됐습니다.")
                    if recovery_phase == 2 and now - recovery_started >= 0.8 and (stall < 0.5 or stuck.motion >= STUCK_DIFF_THRESHOLD):
                        recovery_phase = 0; episode += 1; reset_count += 1
                        grace_until = now + STUCK_GRACE
                        overnight_log("recovered", motion=stuck.motion, stall=stall)

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
                cv2.putText(preview, "OBJECT SEGMENTATION SENSE", (panel_x1 + 8, panel_y1 + 18),
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
                cv2.putText(preview, f"STUCK SENSE: {stall:.2f} | rear/steer from connectome",
                            (panel_x1 + 8, panel_y1 + 106), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                            (0, 220, 255), 1)

                state = "RUN" if driving else "IDLE"
                steer_button = joy.steer_button
                steer_label = {-1: "LEFT", 0: "CENTER", 1: "RIGHT"}.get(steer_button, "CENTER")
                lines = [
                    f"FPS {fps:.0f} | {state} | cov {feat['coverage']:.2f} | motion {stuck.motion:.1f} | stall {stall:.2f} | ObjectAI {'STALE' if object_result.get('stale') else 'LIVE'}",
                    f"steer {steer_label} ({smooth_steer:+.2f}) | drive {brain.drive} | reward {reward:+.3f} | total {brain.reward_total:+.2f}",
                    f"connectome N={brain.n:,} E={len(brain.w):,} | input={len(brain.input_idx):,} out={len(brain.output_idx):,} DA={len(brain.da_idx):,}",
                    f"active DN: {brain.last_output_ids[:6]}",
                    f"wall L/R {feat['wall_l']:.2f}/{feat['wall_r']:.2f} | front {ai_front:.2f} | vJoy X/Y {joy.output_x_value}/{joy.output_y_value} | plastic {int(brain.plastic_mask.sum()):,}",
                    f"turn source={brain.turn_readout_source} | neural L/R {brain.last_left_rate:.3f}/{brain.last_right_rate:.3f}",
                    f"learn R={brain.last_reward:+.3f} total={brain.reward_total:+.2f} | outbias={brain.output_bias:+.3f}",
                ]
                for i, text in enumerate(lines):
                    cv2.putText(preview, text, (6, 20 + i * 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)

                wall_ai_state = {
                    "left": ai_l, "right": ai_r, "front": ai_front,
                    "drifting": float(object_result.get("drifting", 0.0)),
                    "ready": bool(object_result.get("ready", False)),
                    "loaded": bool(object_ai.loaded),
                    "device": object_ai.device,
                }
                api.update(brain.api_snapshot(
                    feat=feat, stall=stall, reward=reward, driving=driving, fps=fps,
                    reward_breakdown=rewarder.last_breakdown, wall_ai=wall_ai_state,
                    recovery=bool(stall >= STUCK_THRESHOLD and max(ai_l, ai_r, ai_front) >= WALL_NEAR_THRESHOLD),
                    controller=joy.status(),
                    reward_context=rewarder.last_debug,
                ))

                cv2.imshow("flyrider-connectome", preview)
                cv2.imshow("mask", feat["mask"])
                if object_ai.debug_enabled and object_result.get("ready"):
                    if not debug_window_placed:
                        cv2.namedWindow("object AI debug", cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("object AI debug", 660, 520)
                        screen_width = ctypes.windll.user32.GetSystemMetrics(0)
                        cv2.moveWindow("object AI debug", max(0, screen_width - 680), 42)
                        debug_window_placed = True
                    cv2.imshow("object AI debug", ObjectVision.render(frame, object_result))
                elif not object_ai.debug_enabled and debug_window_placed:
                    try: cv2.destroyWindow("object AI debug")
                    except cv2.error: pass
                    debug_window_placed = False

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
            object_ai.close()
            joy.shutdown()
            brain.save_learning()
            cfg.update({"s_max": cfg.get("s_max", 50), "v_min": cfg.get("v_min", 70), "v_max": cfg.get("v_max", 210)})
            save_config(cfg)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
