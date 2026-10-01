# chopari_rider.py  (초파리라이더 - 디지털 조향 버전)
#
# 실행: 관리자 권한 터미널에서  python chopari_rider.py
# 영역 다시 선택: python chopari_rider.py --select
#
# 미리보기 창 키 (미리보기 창에 포커스가 있을 때):
#   q : 종료   s : 스크린샷 저장   r : 녹화 켜기/끄기   c : 게임 영역 다시 선택
# 전역 단축키 (게임 창에 포커스가 있어도 동작):
#   F8 : 자동 주행 켜기/끄기      F9 : 즉시 정지 (모든 입력 해제)
#
# 입력 규칙
#   - 조향: 좌/중앙/우 세 가지뿐 (X축 최소=왼쪽, 최대=오른쪽). 세밀한 조향은 PWM(짧게 톡톡)으로 흉내
#   - 전진: 계속 누름 (Y축 FORWARD_VALUE)
#   - 벽에 끼면: 자동으로 잠깐 후진 (Y축 REVERSE_VALUE)
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

# ---------------- 설정 ----------------
CONFIG_PATH = "chopari_config.json"
SAVE_DIR = "captures"
MONITOR_INDEX = 1
PREVIEW_WIDTH = 640
PROC_WIDTH = 320
RECORD_FPS = 10

VJOY_DEVICE_ID = 1
AXIS_MIN = 0x0001
AXIS_CENTER = 0x4000
AXIS_MAX = 0x8000
FORWARD_VALUE = AXIS_MIN     # Y축이 '앞'인 값. 반대로 가면 AXIS_MAX로 바꾸세요.
REVERSE_VALUE = AXIS_MAX if FORWARD_VALUE == AXIS_MIN else AXIS_MIN

# 조향 (디지털)
STEER_MODE = "pwm"           # "pwm" = 세기에 따라 톡톡 누름, "bang" = 조금이라도 필요하면 계속 누름
STEER_GAIN = 1.4             # 도로 쏠림 -> 조향 세기 배율
STEER_DEADZONE = 0.08        # 이보다 작으면 직진
STEER_SMOOTH = 0.5           # 0~1, 클수록 반응이 빠름
STEER_INVERT = False         # 좌우가 반대로 꺾이면 True
FULL_TURN = 0.6              # 이 세기 이상이면 키를 계속 누름
PWM_PERIOD = 0.12            # 초. 조향이 굼뜨거나 너무 떨리면 조절

# 막힘(벽에 낀 상태) 감지 -> 후진
STUCK_INTERVAL = 0.25        # 이 간격으로 화면 변화량을 비교
STUCK_DIFF_THRESHOLD = 1.5   # 변화량이 이보다 작으면 '정지'로 판단 (화면의 motion 값을 보고 조절)
STUCK_SECONDS = 1.2          # 정지가 이 시간 이어지면 막힌 것으로 판단
STUCK_GRACE = 3.0            # F8을 켜거나 후진을 끝낸 뒤 이 시간 동안은 막힘 판단 안 함
REVERSE_SECONDS = 0.9        # 후진 지속 시간
REVERSE_STEER = "none"       # 후진 중 조향: "none" 직진 | "toward" 도로 쪽 | "away" 반대쪽

# 화면 안의 상대 좌표 (x1, y1, x2, y2), 0~1
LOOKAHEAD_BAND = (0.0, 0.33, 1.0, 0.46)   # 카트 위쪽, 도로를 읽는 띠
MINIMAP_RECT = (0.82, 0.36, 1.0, 0.88)
HUD_RECT = (0.78, 0.0, 1.0, 0.30)         # 순위/시간
SPEEDO_RECT = (0.38, 0.85, 0.62, 1.0)     # 속도계

DEFAULT_CFG = {"region": None, "s_max": 50, "v_min": 70, "v_max": 210}
# --------------------------------------

user32 = ctypes.windll.user32


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
    print("게임 화면(제목 표시줄 제외)을 마우스로 드래그한 뒤 Enter를 누르세요. 취소는 c.")
    title = "select game area"
    x, y, w, h = cv2.selectROI(title, small, False)
    cv2.destroyWindow(title)
    if w == 0 or h == 0:
        return None
    x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)
    return {"left": mon["left"] + x, "top": mon["top"] + y, "width": w, "height": h}


def rel_to_px(rect, w, h):
    return int(rect[0] * w), int(rect[1] * h), int(rect[2] * w), int(rect[3] * h)


def estimate_steer(frame, s_max, v_min, v_max):
    """도로(회색) 픽셀의 좌우 쏠림을 -1(왼쪽)~+1(오른쪽)로 반환. 못 찾으면 None."""
    h0, w0 = frame.shape[:2]
    small = cv2.resize(frame, (PROC_WIDTH, int(PROC_WIDTH * h0 / w0)))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, v_min), (179, s_max, v_max))
    h, w = mask.shape

    mx1, my1, mx2, my2 = rel_to_px(MINIMAP_RECT, w, h)
    mask[my1:my2, mx1:mx2] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    bx1, by1, bx2, by2 = rel_to_px(LOOKAHEAD_BAND, w, h)
    band = mask[by1:by2, bx1:bx2]
    xs = np.nonzero(band)[1]
    coverage = len(xs) / max(band.size, 1)

    vis = small.copy()
    cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 255, 255), 1)
    cv2.rectangle(vis, (mx1, my1), (mx2, my2), (255, 0, 255), 1)
    cv2.line(vis, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)

    if coverage < 0.02:
        return None, mask, vis, coverage

    cx = float(xs.mean()) + bx1
    err = (cx - w / 2) / (w / 2)
    cv2.line(vis, (int(cx), by1), (int(cx), by2), (0, 255, 0), 2)
    return err, mask, vis, coverage


def motion_gray(frame):
    """막힘 감지용 작은 흑백 이미지. HUD/미니맵/속도계는 지워서 비교에서 뺀다."""
    h0, w0 = frame.shape[:2]
    g = cv2.cvtColor(cv2.resize(frame, (160, int(160 * h0 / w0))), cv2.COLOR_BGR2GRAY)
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


class Driver:
    """vJoy 출력 담당. 별도 스레드가 5ms마다 조향 PWM을 처리한다."""

    def __init__(self, device_id):
        self.dev = None
        self.lock = threading.Lock()
        self.steer = 0.0
        self.drive = "idle"       # "idle" | "forward" | "reverse"
        self.force_dir = None     # None이면 steer로 PWM, -1/0/1이면 그 방향 고정
        self.last_x = None
        self.last_y = None
        self.running = False
        if pyvjoy is None:
            print("[vJoy] pyvjoy가 없어 주행 출력은 비활성화됩니다.")
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


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    cfg = load_config()
    joy = Driver(VJOY_DEVICE_ID)
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
        mode = "forward"
        drive_start = 0.0
        reverse_until = 0.0
        last_err = 0.0
        recording = False
        last_save = 0.0
        rec_count = 0
        smoothed = 0.0
        fps, fps_frames, fps_timer = 0.0, 0, time.time()
        s_max, v_min, v_max = cfg["s_max"], cfg["v_min"], cfg["v_max"]
        print("F8 자동 주행 | F9 즉시 정지 | q 종료 | s 저장 | r 녹화 | c 영역 재선택")

        while True:
            shot = sct.grab(region)
            frame = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)
            now = time.time()

            s_max = cv2.getTrackbarPos("S_MAX", "mask")
            v_min = cv2.getTrackbarPos("V_MIN", "mask")
            v_max = cv2.getTrackbarPos("V_MAX", "mask")
            err, mask, vis, coverage = estimate_steer(frame, s_max, v_min, v_max)

            if hk_drive.pressed():
                driving = not driving
                print("자동 주행", "ON" if driving else "OFF")
                if driving:
                    mode = "forward"
                    drive_start = now
                    smoothed = 0.0
                    stuck.reset()
                else:
                    joy.release()
            if hk_stop.pressed():
                driving = False
                smoothed = 0.0
                joy.release()
                print("즉시 정지")

            if driving:
                stuck.update(motion_gray(frame), now)

                if err is None:
                    smoothed *= 0.9
                else:
                    last_err = err
                    target = max(-1.0, min(1.0, err * STEER_GAIN))
                    smoothed += STEER_SMOOTH * (target - smoothed)

                if mode == "forward":
                    if (now - drive_start >= STUCK_GRACE
                            and stuck.still_duration(now) >= STUCK_SECONDS):
                        mode = "reverse"
                        reverse_until = now + REVERSE_SECONDS
                        print("막힘 감지 -> 후진")
                elif now >= reverse_until:
                    mode = "forward"
                    drive_start = now
                    stuck.reset()
                    print("전진 재개")

                force = None
                if mode == "reverse":
                    if REVERSE_STEER == "toward":
                        force = 1 if last_err >= 0 else -1
                    elif REVERSE_STEER == "away":
                        force = -1 if last_err >= 0 else 1
                    else:
                        force = 0
                joy.command(smoothed, mode, force)
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

            err_txt = "none" if err is None else f"{err:+.2f}"
            state = mode.upper() if driving else "idle"
            line1 = f"FPS {fps:.0f} | {state} | err {err_txt} | steer {smoothed:+.2f}"
            line2 = (f"cov {coverage:.2f} | motion {stuck.motion:.1f} | "
                     f"still {stuck.still_duration(now):.1f}s"
                     f"{' | REC ' + str(rec_count) if recording else ''}")
            cv2.putText(vis, line1, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(vis, line2, (5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            preview = cv2.resize(vis, (PREVIEW_WIDTH, int(PREVIEW_WIDTH * aspect)))
            cv2.imshow("chopari", preview)
            cv2.imshow("mask", mask)

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
        cfg.update({"s_max": s_max, "v_min": v_min, "v_max": v_max})
        save_config(cfg)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()