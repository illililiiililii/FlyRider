# chopari_rider.py  (초파리라이더 - 3단계 도로 인식 + 4단계 조향 첫 버전)
#
# 실행: 관리자 권한 터미널에서  python chopari_rider.py
# 영역 다시 선택: python chopari_rider.py --select
#
# 미리보기 창 키 (미리보기 창에 포커스가 있을 때):
#   q : 종료   s : 스크린샷 저장   r : 녹화 켜기/끄기   c : 게임 영역 다시 선택
# 전역 단축키 (게임 창에 포커스가 있어도 동작):
#   F8 : 자동 주행 켜기/끄기      F9 : 즉시 정지 (조향 중앙, 가속 해제)
import os
import sys
import json
import time
import ctypes

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
PROC_WIDTH = 320             # 분석용으로 줄이는 가로 크기
RECORD_FPS = 10

VJOY_DEVICE_ID = 1
AXIS_MIN = 0x0001
AXIS_CENTER = 0x4000
AXIS_MAX = 0x8000
FORWARD_VALUE = AXIS_MIN     # Y축이 앞(가속)으로 매핑된 값. 반대로 가면 AXIS_MAX로 바꾸세요.

STEER_GAIN = 1.4             # 클수록 민감
STEER_DEADZONE = 0.05
STEER_SMOOTH = 0.5           # 0~1, 클수록 반응이 빠름
STEER_INVERT = False         # 좌우가 반대로 꺾이면 True

# 화면 안의 상대 좌표 (x1, y1, x2, y2), 0~1
LOOKAHEAD_BAND = (0.0, 0.33, 1.0, 0.46)   # 카트 위쪽, 도로를 읽는 띠
MINIMAP_RECT = (0.82, 0.36, 1.0, 0.88)    # 미니맵은 도로로 오인하지 않게 제외

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


class Driver:
    def __init__(self, device_id):
        self.dev = None
        if pyvjoy is None:
            print("[vJoy] pyvjoy가 없어 주행 출력은 비활성화됩니다.")
            return
        try:
            self.dev = pyvjoy.VJoyDevice(device_id)
            self.dev.reset()
            print("[vJoy] 장치 열기 성공")
        except Exception as e:
            print("[vJoy] 장치를 열지 못했습니다:", e)

    def set_steer(self, s):
        if self.dev is None:
            return
        s = max(-1.0, min(1.0, s))
        value = int(AXIS_CENTER + s * 0x3FFF)
        self.dev.set_axis(pyvjoy.HID_USAGE_X, max(AXIS_MIN, min(AXIS_MAX, value)))

    def set_throttle(self, on):
        if self.dev is None:
            return
        self.dev.set_axis(pyvjoy.HID_USAGE_Y, FORWARD_VALUE if on else AXIS_CENTER)

    def release(self):
        if self.dev is None:
            return
        self.dev.set_axis(pyvjoy.HID_USAGE_X, AXIS_CENTER)
        self.dev.set_axis(pyvjoy.HID_USAGE_Y, AXIS_CENTER)


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
    hk_drive = Hotkey(0x77)   # F8
    hk_stop = Hotkey(0x78)    # F9

    with mss.mss() as sct:
        if MONITOR_INDEX >= len(sct.monitors):
            print("모니터 번호가 잘못되었습니다.")
            return
        if "--select" in sys.argv or not cfg.get("region"):
            region = select_region(sct)
            if region is None:
                print("영역 선택이 취소되었습니다.")
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
        recording = False
        last_save = 0.0
        rec_count = 0
        smoothed = 0.0
        fps, fps_frames, fps_timer = 0.0, 0, time.time()
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
                if not driving:
                    joy.release()
            if hk_stop.pressed():
                driving = False
                smoothed = 0.0
                joy.release()
                print("즉시 정지")

            if driving:
                if err is None:
                    smoothed *= 0.9           # 도로를 못 찾으면 조향을 서서히 중앙으로
                else:
                    target = err * STEER_GAIN
                    if abs(target) < STEER_DEADZONE:
                        target = 0.0
                    target = max(-1.0, min(1.0, target))
                    smoothed += STEER_SMOOTH * (target - smoothed)
                out = -smoothed if STEER_INVERT else smoothed
                joy.set_steer(out)
                joy.set_throttle(True)

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
            label = (f"FPS {fps:.0f} | {'DRIVE' if driving else 'idle'} | "
                     f"err {err_txt} | steer {smoothed:+.2f} | cov {coverage:.2f}"
                     f"{' | REC ' + str(rec_count) if recording else ''}")
            cv2.putText(vis, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 0), 1)
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

        joy.release()
        cfg.update({"s_max": s_max, "v_min": v_min, "v_max": v_max})
        save_config(cfg)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()