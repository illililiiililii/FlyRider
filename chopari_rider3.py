# chopari_rider.py  (초파리라이더 - 초파리 영감 SNN 뇌 버전)
#
# 실행: 관리자 권한 터미널에서  python chopari_rider.py
# 영역 다시 선택: python chopari_rider.py --select
#
# 구조:  눈(extract_features) -> 뇌(FlyBrain, LIF 스파이킹 신경망) -> 손발(Driver, vJoy)
#
# 미리보기 창 키 (미리보기 창에 포커스가 있을 때):
#   q : 종료   s : 스크린샷 저장   r : 녹화 켜기/끄기   c : 게임 영역 다시 선택
# 전역 단축키 (게임 창에 포커스가 있어도 동작):
#   F8 : 뇌 주행 켜기/끄기      F9 : 즉시 정지 (모든 입력 해제)
import os
import sys
import json
import time
import ctypes
import threading

import cv2
import numpy as np
import mss

try:
    import pyvjoy
except ImportError:
    pyvjoy = None

# ================= 설정 =================
CONFIG_PATH = "chopari_config.json"
BRAIN_PATH = "chopari_brain.json"
SAVE_DIR = "captures"
MONITOR_INDEX = 1
PREVIEW_WIDTH = 640
PROC_WIDTH = 320
RECORD_FPS = 10

# ---- 손발 (vJoy) ----
VJOY_DEVICE_ID = 1
AXIS_MIN = 0x0001
AXIS_CENTER = 0x4000
AXIS_MAX = 0x8000
FORWARD_VALUE = AXIS_MIN     # Y축이 '앞'인 값. 반대로 가면 AXIS_MAX
REVERSE_VALUE = AXIS_MAX if FORWARD_VALUE == AXIS_MIN else AXIS_MIN
STEER_MODE = "pwm"           # "pwm" = 세기에 따라 톡톡 누름, "bang" = 계속 누름
STEER_DEADZONE = 0.08
STEER_INVERT = False         # 좌우가 반대로 꺾이면 True
FULL_TURN = 0.6              # 이 세기 이상이면 키를 계속 누름
PWM_PERIOD = 0.12

# ---- 눈 ----
N_COL = 8                                  # 도로 열 감각 뉴런 수
LOOKAHEAD_BAND = (0.0, 0.33, 1.0, 0.46)    # 카트 위쪽, 도로를 읽는 띠
MINIMAP_RECT = (0.82, 0.36, 1.0, 0.88)
HUD_RECT = (0.78, 0.0, 1.0, 0.30)
SPEEDO_RECT = (0.38, 0.85, 0.62, 1.0)
LEFT_SIDE_RECT = (0.10, 0.55, 0.40, 0.85)  # 카트 왼쪽 옆 (벽 감각)
RIGHT_SIDE_RECT = (0.60, 0.55, 0.80, 0.85) # 카트 오른쪽 옆 (벽 감각)
WALL_BASE = 0.4                            # 옆 영역의 비도로 비율이 이 값을 넘어야 '벽 가까움'

# ---- 정지 감각 (화면 변화량) ----
STUCK_INTERVAL = 0.25
STUCK_DIFF_THRESHOLD = 1.5   # 화면 motion 값이 이보다 작으면 '정지'
STALL_FULL_SEC = 1.0         # 정지가 이 시간 이어지면 정지 감각이 최대
STUCK_GRACE = 3.0            # F8을 켠 뒤 이 시간 동안은 정지 감각 끔 (출발 카운트다운)
RECOVERY_GRACE = 1.5         # 후진을 끝낸 뒤 정지 감각을 다시 켜기까지의 시간

# ---- 뇌 ----
BRAIN_DT = 0.005             # 시뮬레이션 스텝 5ms
TAU_M = 0.02                 # 막전위 시정수
TAU_TRACE = 0.05             # 발화율 읽기용 시정수
TAU_SYN_FAST = 0.008
TAU_SYN_SLOW = 0.9           # M 뉴런(후진 지속)의 느린 시냅스
P_REF = 0.4                  # 발화율 1.0으로 보는 기준 (스텝당 발화확률 = 80Hz)
REP, P_INT, P_M, P_MOT = 4, 6, 6, 8
N_CH = 11                    # 감각 채널: 0~7 도로 열, 8 정지, 9 왼쪽 벽, 10 오른쪽 벽
NS = N_CH * REP
NI = 4 * P_INT               # 중간뉴런: A(왼쪽으로 꺾기) B(오른쪽으로 꺾기) C(막힘) D(순항)
NM = P_M                     # M: 후진 지속
NO = 4 * P_MOT               # 운동뉴런: LEFT RIGHT FWD REV

G_STALL = 1.6                # 정지 감각 -> C
G_C2M = 9.0                  # C -> M (클수록 후진이 오래 지속)
G_STEER = 3.2                # A->LEFT, B->RIGHT (조향 이득)
G_CRUISE = 3.0               # D -> FWD
BIAS_CRUISE = 1.5            # D의 기본 발화 (계속 전진)
STEER_OUT_GAIN = 1.0

M_SIDE_INIT = 1.4            # 후진 중 좌/우 조향 시냅스 초기값 (학습됨)
W_M_MIN, W_M_MAX = 0.2, 2.5
GAIN_STEP = 1.08             # 벽에 박을 때마다 조향 이득 증가 배율
GAIN_MAX = 2.0
GAIN_DECAY = 0.02            # 초당 원래 이득으로 복귀하는 비율
RECOVERY_WINDOW = 6.0        # 후진 후 이 시간 안에 또 막히면 '실패'

DEFAULT_CFG = {"region": None, "s_max": 50, "v_min": 70, "v_max": 210}
# ========================================

user32 = ctypes.windll.user32


# ---------------- 설정 파일 / 영역 선택 ----------------
def load_config():
    cfg = dict(DEFAULT_CFG)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception as e:
            print("설정 파일을 읽지 못해 기본값을 씁니다:", e)
    return cfg


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def select_region(sct):
    mon = sct.monitors[MONITOR_INDEX]
    img = cv2.cvtColor(np.array(sct.grab(mon)), cv2.COLOR_BGRA2BGR)
    scale = 0.5 if img.shape[1] > 1600 else 1.0
    small = cv2.resize(img, None, fx=scale, fy=scale)
    print("게임 화면(제목 표시줄 제외)을 마우스로 드래그한 뒤 Enter를 누르세요.")
    title = "select game area"
    x, y, w, h = cv2.selectROI(title, small, False)
    cv2.destroyWindow(title)
    if w == 0 or h == 0:
        return None
    x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)
    return {"left": mon["left"] + x, "top": mon["top"] + y, "width": w, "height": h}


def rel_to_px(rect, w, h):
    return int(rect[0] * w), int(rect[1] * h), int(rect[2] * w), int(rect[3] * h)


# ---------------- 눈 ----------------
def extract_features(frame, s_max, v_min, v_max):
    """화면 -> 감각 입력. 도로 열 8개(0~1), 좌우 벽 근접(0~1)."""
    h0, w0 = frame.shape[:2]
    small = cv2.resize(frame, (PROC_WIDTH, int(PROC_WIDTH * h0 / w0)))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, v_min), (179, s_max, v_max))
    h, w = mask.shape

    mx1, my1, mx2, my2 = rel_to_px(MINIMAP_RECT, w, h)
    mask[my1:my2, mx1:mx2] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    road = mask > 0

    bx1, by1, bx2, by2 = rel_to_px(LOOKAHEAD_BAND, w, h)
    band = road[by1:by2, bx1:bx2]
    idx_groups = np.array_split(np.arange(band.shape[1]), N_COL)
    road_cols = np.array([band[:, idx].mean() if len(idx) else 0.0 for idx in idx_groups])

    def wall_score(rect):
        x1, y1, x2, y2 = rel_to_px(rect, w, h)
        region = road[y1:y2, x1:x2]
        nonroad = 1.0 - (float(region.mean()) if region.size else 1.0)
        return float(np.clip((nonroad - WALL_BASE) / (1.0 - WALL_BASE), 0.0, 1.0))

    wall_l = wall_score(LEFT_SIDE_RECT)
    wall_r = wall_score(RIGHT_SIDE_RECT)

    vis = small.copy()
    cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 255, 255), 1)
    cv2.rectangle(vis, (mx1, my1), (mx2, my2), (255, 0, 255), 1)
    for rect, val in ((LEFT_SIDE_RECT, wall_l), (RIGHT_SIDE_RECT, wall_r)):
        x1, y1, x2, y2 = rel_to_px(rect, w, h)
        color = (0, int(255 * (1 - val)), int(255 * val))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
    colw = max(1, (bx2 - bx1) // N_COL)
    for i, val in enumerate(road_cols):
        x0 = bx1 + i * colw
        cv2.rectangle(vis, (x0, by1 - int(val * 20)), (x0 + colw - 2, by1), (0, 255, 0), -1)
    cv2.line(vis, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)

    return {"road": road_cols, "wall_l": wall_l, "wall_r": wall_r,
            "mask": mask, "vis": vis, "coverage": float(band.mean())}


def motion_gray(frame):
    h0, w0 = frame.shape[:2]
    g = cv2.cvtColor(cv2.resize(frame, (160, int(160 * h0 / w0))), cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    h, w = g.shape
    for rect in (MINIMAP_RECT, HUD_RECT, SPEEDO_RECT):
        x1, y1, x2, y2 = rel_to_px(rect, w, h)
        g[y1:y2, x1:x2] = 0
    return g


class StuckDetector:
    """화면 변화량(옵틱 플로우 대용). 정지 감각 뉴런의 입력이 된다."""

    def __init__(self):
        self.motion = 0.0
        self.reset()

    def reset(self):
        self.ref = None
        self.ref_time = 0.0
        self.still_since = None

    def update(self, gray, now):
        if self.ref is None:
            self.ref, self.ref_time = gray, now
            return
        if now - self.ref_time >= STUCK_INTERVAL:
            self.motion = float(np.mean(cv2.absdiff(gray, self.ref)))
            if self.motion < STUCK_DIFF_THRESHOLD:
                if self.still_since is None:
                    self.still_since = self.ref_time
            else:
                self.still_since = None
            self.ref, self.ref_time = gray, now

    def still_duration(self, now):
        return 0.0 if self.still_since is None else now - self.still_since


# ---------------- 뇌 ----------------
class FlyBrain:
    """초파리에서 영감을 받은 작은 LIF 스파이킹 신경망.
    실제 connectome이 아니라, 알려진 회로 개념(감각 반사, 좌우 상호억제, 후진 하강 뉴런,
    도파민식 보상 조절)을 손으로 배선한 것. 나중에 이 클래스를 connectome 기반으로 교체하면 된다.

    입력  update(now, road_cols[8], wall_l, wall_r, stall)
    출력  self.steer (-1 왼쪽 ~ +1 오른쪽), self.drive ("forward" | "reverse")
    """

    def __init__(self, path=BRAIN_PATH):
        self.path = path
        self.rng = np.random.default_rng()
        self.a_m = float(np.exp(-BRAIN_DT / TAU_M))
        self.tr_decay = float(np.exp(-BRAIN_DT / TAU_TRACE))
        self.ds_fast = float(np.exp(-BRAIN_DT / TAU_SYN_FAST))
        self.ds_slow = float(np.exp(-BRAIN_DT / TAU_SYN_SLOW))
        self.w_m = [M_SIDE_INIT, M_SIDE_INIT]   # 후진 중 LEFT/RIGHT 시냅스 (학습)
        self.gain = 1.0                          # 조향 이득 (학습)
        self.load()
        self._build()
        self.reset_state()

    # --- 저장/불러오기 ---
    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                self.w_m = [float(np.clip(v, W_M_MIN, W_M_MAX)) for v in d.get("w_m", self.w_m)]
                self.gain = float(np.clip(d.get("gain", 1.0), 1.0, GAIN_MAX))
                print("[뇌] 저장된 학습값 불러옴:", self.w_m, self.gain)
            except Exception as e:
                print("[뇌] 학습값을 읽지 못했습니다:", e)

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"w_m": self.w_m, "gain": self.gain}, f)
        except Exception as e:
            print("[뇌] 저장 실패:", e)

    # --- 배선 ---
    def _build(self):
        # 행=시냅스 뒤 그룹, 열=시냅스 앞 그룹. 값=그룹 단위 연결 세기(문턱 대비)
        g_s2i = np.zeros((4, N_CH))          # 감각 -> 중간뉴런 (A, B, C, D)
        g_s2i[0, 0:4] = 0.35
        g_s2i[0, 4:8] = -0.15
        g_s2i[0, 9] = -0.5                   # 왼쪽 벽 -> A 억제
        g_s2i[0, 10] = 1.0                   # 오른쪽 벽 -> A (왼쪽으로 피하기)
        g_s2i[1, 0:4] = -0.15
        g_s2i[1, 4:8] = 0.35
        g_s2i[1, 9] = 1.0                    # 왼쪽 벽 -> B (오른쪽으로 피하기)
        g_s2i[1, 10] = -0.5
        g_s2i[2, 8] = G_STALL                # 정지 -> C

        g_i2m = np.array([[0.0, 0.0, G_C2M, 0.0]])   # C -> M

        g_i2o = np.zeros((4, 4))             # 중간 -> 운동 (LEFT RIGHT FWD REV)
        g_i2o[0, 0] = G_STEER * self.gain    # A -> LEFT
        g_i2o[1, 1] = G_STEER * self.gain    # B -> RIGHT
        g_i2o[2, 3] = G_CRUISE               # D -> FWD
        g_i2o[2, 2] = -4.0                   # C -> FWD 억제
        g_i2d = np.zeros((4, 4))             # C -> D 억제 (막히면 순항 중단)
        g_i2d[3, 2] = -3.0

        g_m2o = np.array([[self.w_m[0]], [self.w_m[1]], [-5.0], [3.0]])  # M -> 운동

        g_o2o = np.zeros((4, 4))             # 운동 뉴런끼리 상호억제
        g_o2o[0, 1] = -2.0
        g_o2o[1, 0] = -2.0
        g_o2o[2, 3] = -3.0
        g_o2o[3, 2] = -3.0

        # 중간뉴런 내부(C->D)는 g_s2i 대신 별도 행렬로 처리
        self.W_s2i = np.kron(g_s2i, np.ones((P_INT, REP))) / (REP * P_REF)
        self.W_i2i = np.kron(g_i2d, np.ones((P_INT, P_INT))) / (P_INT * P_REF)
        self.W_i2m = np.kron(g_i2m, np.ones((P_M, P_INT))) / (P_INT * P_REF)
        self.W_i2o = np.kron(g_i2o, np.ones((P_MOT, P_INT))) / (P_INT * P_REF)
        self.W_m2o = np.kron(g_m2o, np.ones((P_MOT, P_M))) / (P_M * P_REF)
        self.W_o2o = np.kron(g_o2o, np.ones((P_MOT, P_MOT))) / (P_MOT * P_REF)
        self.bias_i = np.repeat(np.array([0.0, 0.0, 0.0, BIAS_CRUISE]), P_INT)

    def reset_state(self):
        self.v_i = np.zeros(NI)
        self.I_i = np.zeros(NI)
        self.s_i = np.zeros(NI)
        self.v_m = np.zeros(NM)
        self.I_m = np.zeros(NM)
        self.v_o = np.zeros(NO)
        self.I_o = np.zeros(NO)
        self.s_o = np.zeros(NO)
        self.x_i = np.zeros(4)
        self.x_m = 0.0
        self.x_o = np.zeros(4)
        self.steer = 0.0
        self.drive = "forward"
        self.last_t = None
        self.rev_active = False
        self.side_acc = np.zeros(2)
        self.pending = None

    # --- 시뮬레이션 ---
    def _sim_step(self, x_ch):
        p = np.repeat(x_ch, REP) * P_REF
        s_s = (self.rng.random(NS) < p).astype(float)

        drive_i = self.W_s2i @ s_s + self.W_i2i @ self.s_i + self.bias_i
        self.I_i = self.ds_fast * self.I_i + (1 - self.ds_fast) * drive_i
        self.v_i = self.a_m * self.v_i + (1 - self.a_m) * self.I_i
        fired = self.v_i >= 1.0
        self.v_i[fired] = 0.0
        self.s_i = fired.astype(float)

        drive_m = self.W_i2m @ self.s_i
        self.I_m = self.ds_slow * self.I_m + (1 - self.ds_slow) * drive_m
        self.v_m = self.a_m * self.v_m + (1 - self.a_m) * self.I_m
        fired_m = self.v_m >= 1.0
        self.v_m[fired_m] = 0.0
        s_m = fired_m.astype(float)

        drive_o = self.W_i2o @ self.s_i + self.W_m2o @ s_m + self.W_o2o @ self.s_o
        self.I_o = self.ds_fast * self.I_o + (1 - self.ds_fast) * drive_o
        self.v_o = self.a_m * self.v_o + (1 - self.a_m) * self.I_o
        fired_o = self.v_o >= 1.0
        self.v_o[fired_o] = 0.0
        self.s_o = fired_o.astype(float)

        frac_i = self.s_i.reshape(4, P_INT).mean(axis=1) / P_REF
        frac_o = self.s_o.reshape(4, P_MOT).mean(axis=1) / P_REF
        frac_m = float(s_m.mean()) / P_REF
        d = self.tr_decay
        self.x_i = d * self.x_i + (1 - d) * frac_i
        self.x_o = d * self.x_o + (1 - d) * frac_o
        self.x_m = d * self.x_m + (1 - d) * frac_m

    def update(self, now, road_cols, wall_l, wall_r, stall):
        x_ch = np.zeros(N_CH)
        x_ch[0:8] = np.clip(road_cols, 0.0, 1.0)
        x_ch[8] = float(np.clip(stall, 0.0, 1.0))
        x_ch[9] = float(np.clip(wall_l, 0.0, 1.0))
        x_ch[10] = float(np.clip(wall_r, 0.0, 1.0))

        dt = 0.03 if self.last_t is None else max(0.0, now - self.last_t)
        self.last_t = now
        n = int(np.clip(round(dt / BRAIN_DT), 1, 12))
        for _ in range(n):
            self._sim_step(x_ch)

        xl, xr, xf, xv = self.x_o
        self.steer = float(np.clip((xr - xl) * STEER_OUT_GAIN, -1.0, 1.0))
        self.drive = "reverse" if (xv > xf + 0.1 and xv > 0.15) else "forward"

        self._learn(now, dt, xl, xr)

    # --- 학습 (보상 조절) ---
    def _outcome(self, success):
        side = self.pending["side"]
        other = 1 - side
        if success:
            self.w_m[side] += 0.15
        else:
            self.w_m[side] -= 0.3
            self.w_m[other] += 0.3
        self.w_m = [float(np.clip(v, W_M_MIN, W_M_MAX)) for v in self.w_m]
        name = "왼쪽" if side == 0 else "오른쪽"
        print(f"[뇌] 후진 {name} 조향 -> {'성공' if success else '실패'} | "
              f"wM L{self.w_m[0]:.2f} R{self.w_m[1]:.2f}")
        self.pending = None
        self._build()
        self.save()

    def _learn(self, now, dt, xl, xr):
        self.gain += (1.0 - self.gain) * min(1.0, GAIN_DECAY * dt)

        if self.drive == "reverse":
            if not self.rev_active:
                if self.pending is not None and now - self.pending["end"] <= RECOVERY_WINDOW:
                    self._outcome(False)          # 후진 직후 또 막힘 = 실패
                self.pending = None
                self.rev_active = True
                self.side_acc = np.zeros(2)
                self.gain = min(GAIN_MAX, self.gain * GAIN_STEP)   # 벽에 박음 -> 더 강하게 꺾기
                print(f"[뇌] 막힘 -> 후진 시작 (조향 이득 {self.gain:.2f})")
            self.side_acc += (xl, xr)
        elif self.rev_active:
            self.rev_active = False
            self.pending = {"end": now, "side": int(np.argmax(self.side_acc))}

        if self.pending is not None:
            since = now - self.pending["end"]
            if since >= 1.0 and self.x_i[2] > 0.3:
                self._outcome(False)
            elif since > RECOVERY_WINDOW:
                self._outcome(True)

        self._build()


# ---------------- 손발 (vJoy) ----------------
class Driver:
    """vJoy 출력. 별도 스레드가 5ms마다 조향 PWM을 처리한다."""

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
            print("[vJoy] pyvjoy가 없어 출력은 비활성화됩니다.")
            return
        try:
            self.dev = pyvjoy.VJoyDevice(device_id)
            self.dev.reset()
            print("[vJoy] 장치 열기 성공")
        except Exception as e:
            print("[vJoy] 장치를 열지 못했습니다:", e)
            self.dev = None
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
            self.steer, self.drive, self.force_dir = steer, drive, force_dir

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


# ---------------- 메인 ----------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    cfg = load_config()
    joy = Driver(VJOY_DEVICE_ID)
    brain = FlyBrain()
    stuck = StuckDetector()
    hk_drive = Hotkey(0x77)   # F8
    hk_stop = Hotkey(0x78)    # F9

    with mss.mss() as sct:
        if MONITOR_INDEX >= len(sct.monitors):
            print("모니터 번호가 잘못되었습니다.")
            joy.shutdown()
            return
        if "--select" in sys.argv or not cfg.get("region"):
            region = select_region(sct)
            if region is None:
                print("영역 선택이 취소되었습니다.")
                joy.shutdown()
                return
            cfg["region"] = region
            save_config(cfg)
        region = cfg["region"]
        print("캡처 영역:", region)

        aspect = region["height"] / region["width"]
        cv2.namedWindow("chopari", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("chopari", PREVIEW_WIDTH, int(PREVIEW_WIDTH * aspect))
        cv2.moveWindow("chopari", 0, 0)
        cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("mask", PROC_WIDTH, int(PROC_WIDTH * aspect))
        cv2.moveWindow("mask", 0, int(PREVIEW_WIDTH * aspect) + 60)
        cv2.createTrackbar("S_MAX", "mask", cfg["s_max"], 255, lambda v: None)
        cv2.createTrackbar("V_MIN", "mask", cfg["v_min"], 255, lambda v: None)
        cv2.createTrackbar("V_MAX", "mask", cfg["v_max"], 255, lambda v: None)

        driving = False
        prev_drive = "forward"
        grace_until = 0.0
        recording = False
        last_save = 0.0
        rec_count = 0
        fps, fps_frames, fps_timer = 0.0, 0, time.time()
        s_max, v_min, v_max = cfg["s_max"], cfg["v_min"], cfg["v_max"]
        print("F8 뇌 주행 | F9 즉시 정지 | q 종료 | s 저장 | r 녹화 | c 영역 재선택")

        while True:
            shot = sct.grab(region)
            frame = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)
            now = time.time()

            s_max = cv2.getTrackbarPos("S_MAX", "mask")
            v_min = cv2.getTrackbarPos("V_MIN", "mask")
            v_max = cv2.getTrackbarPos("V_MAX", "mask")
            feat = extract_features(frame, s_max, v_min, v_max)

            if hk_drive.pressed():
                driving = not driving
                print("뇌 주행", "ON" if driving else "OFF")
                if driving:
                    brain.reset_state()
                    stuck.reset()
                    prev_drive = "forward"
                    grace_until = now + STUCK_GRACE
                else:
                    joy.release()
                    brain.save()
            if hk_stop.pressed():
                driving = False
                joy.release()
                brain.save()
                print("즉시 정지")

            stall = 0.0
            if driving:
                stuck.update(motion_gray(frame), now)
                if now >= grace_until:
                    stall = min(1.0, stuck.still_duration(now) / STALL_FULL_SEC)
                brain.update(now, feat["road"], feat["wall_l"], feat["wall_r"], stall)

                if brain.drive != prev_drive:
                    if prev_drive == "reverse":
                        stuck.reset()
                        grace_until = now + RECOVERY_GRACE
                    prev_drive = brain.drive
                joy.command(brain.steer, brain.drive, None)
            else:
                stuck.reset()

            if recording and now - last_save >= 1.0 / RECORD_FPS:
                cv2.imwrite(os.path.join(SAVE_DIR, f"rec_{int(now * 1000)}.jpg"),
                            frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                last_save = now
                rec_count += 1

            fps_frames += 1
            if now - fps_timer >= 1.0:
                fps = fps_frames / (now - fps_timer)
                fps_frames, fps_timer = 0, now

            preview = cv2.resize(feat["vis"], (PREVIEW_WIDTH, int(PREVIEW_WIDTH * aspect)),
                                 interpolation=cv2.INTER_NEAREST)
            xi, xo = brain.x_i, brain.x_o
            state = brain.drive.upper() if driving else "idle"
            lines = [
                f"FPS {fps:.0f} | {state} | cov {feat['coverage']:.2f} | "
                f"motion {stuck.motion:.1f} | stall {stall:.2f}"
                f"{' | REC ' + str(rec_count) if recording else ''}",
                f"A {xi[0]:.2f}  B {xi[1]:.2f}  C {xi[2]:.2f}  D {xi[3]:.2f}  M {brain.x_m:.2f}",
                f"L {xo[0]:.2f}  R {xo[1]:.2f}  F {xo[2]:.2f}  Rv {xo[3]:.2f} | steer {brain.steer:+.2f}",
                f"gain {brain.gain:.2f} | wM L{brain.w_m[0]:.2f} R{brain.w_m[1]:.2f} | "
                f"wall {feat['wall_l']:.2f}/{feat['wall_r']:.2f}",
            ]
            for i, text in enumerate(lines):
                cv2.putText(preview, text, (6, 20 + i * 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)
            cv2.imshow("chopari", preview)
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
                    aspect = region["height"] / region["width"]

        joy.shutdown()
        brain.save()
        cfg.update({"s_max": s_max, "v_min": v_min, "v_max": v_max})
        save_config(cfg)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()