# chopari_capture.py  (초파리라이더 - 2단계: 화면 캡처 + vJoy 축 테스트)
# 키(미리보기 창을 클릭한 상태에서):
#   q : 종료
#   s : 현재 프레임 1장 저장
#   r : 연속 녹화 켜기/끄기 (나중에 모방학습용 데이터가 됨)
#   t : vJoy 테스트 시작 (5초 카운트다운 뒤 X, Y축, 버튼 순서로 움직임)
#       카운트다운 동안 카트라이더 창을 클릭해서 포커스를 주세요.
import os
import sys
import time
import threading

import cv2
import numpy as np
import mss

try:
    import pyvjoy
except ImportError:
    pyvjoy = None

# ---------------- 설정 ----------------
MONITOR_INDEX = 1        # 1 = 주 모니터
REGION = None            # 예: {"left": 0, "top": 0, "width": 1280, "height": 720}
PREVIEW_SCALE = 0.4      # 미리보기 크기 (게임 화면을 가리지 않게 작게)
SAVE_DIR = "captures"
RECORD_FPS = 10          # 녹화 시 초당 저장 장수
JPEG_QUALITY = 90

VJOY_DEVICE_ID = 1
AXIS_MIN = 0x0001
AXIS_CENTER = 0x4000
AXIS_MAX = 0x8000
# --------------------------------------


class VJoyController:
    def __init__(self, device_id):
        self.dev = None
        self.testing = False
        if pyvjoy is None:
            print("[vJoy] pyvjoy가 없어 vJoy 테스트는 비활성화됩니다.")
            return
        try:
            self.dev = pyvjoy.VJoyDevice(device_id)
            self.dev.reset()
            print("[vJoy] 장치 열기 성공")
        except Exception as e:
            print("[vJoy] 장치를 열지 못했습니다:", e)

    def _set(self, axis, value):
        self.dev.set_axis(axis, value)

    def run_test(self):
        if self.dev is None or self.testing:
            return
        self.testing = True
        try:
            print("[vJoy] 5초 뒤 테스트 시작. 카트라이더 창을 클릭하세요.")
            for i in range(5, 0, -1):
                print(f"  {i}...")
                time.sleep(1)

            X, Y = pyvjoy.HID_USAGE_X, pyvjoy.HID_USAGE_Y
            steps = [
                ("X 왼쪽", X, AXIS_MIN, 4),
                ("X 가운데", X, AXIS_CENTER, 1),
                ("X 오른쪽", X, AXIS_MAX, 4),
                ("X 가운데", X, AXIS_CENTER, 1),
                ("Y 최소(보통 위/앞)", Y, AXIS_MIN, 4),
                ("Y 가운데", Y, AXIS_CENTER, 1),
                ("Y 최대(보통 아래/뒤)", Y, AXIS_MAX, 4),
                ("Y 가운데", Y, AXIS_CENTER, 1),
            ]
            for name, axis, value, wait in steps:
                print("[vJoy]", name)
                self._set(axis, value)
                time.sleep(wait)

            for btn in (1, 2):
                print(f"[vJoy] 버튼 {btn}")
                self.dev.set_button(btn, 1)
                time.sleep(3)
                self.dev.set_button(btn, 0)
                time.sleep(3)

            self.dev.reset()
            print("[vJoy] 테스트 끝")
        except Exception as e:
            print("[vJoy] 테스트 중 오류:", e)
        finally:
            self.testing = False


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    joy = VJoyController(VJOY_DEVICE_ID)

    with mss.mss() as sct:
        if MONITOR_INDEX >= len(sct.monitors):
            print("모니터 번호가 잘못되었습니다. 사용 가능:", len(sct.monitors) - 1)
            sys.exit(1)
        region = REGION if REGION else sct.monitors[MONITOR_INDEX]
        print("캡처 영역:", region)
        print("q 종료 | s 저장 | r 녹화 | t vJoy 테스트")

        recording = False
        last_save = 0.0
        frame_count = 0
        fps = 0.0
        fps_timer = time.time()
        fps_frames = 0

        cv2.namedWindow("chopari", cv2.WINDOW_NORMAL)
        cv2.moveWindow("chopari", 0, 0)

        while True:
            shot = sct.grab(region)
            frame = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)
            now = time.time()

            if recording and now - last_save >= 1.0 / RECORD_FPS:
                path = os.path.join(SAVE_DIR, f"rec_{int(now * 1000)}.jpg")
                cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                last_save = now
                frame_count += 1

            fps_frames += 1
            if now - fps_timer >= 1.0:
                fps = fps_frames / (now - fps_timer)
                fps_frames = 0
                fps_timer = now

            preview = cv2.resize(frame, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)
            label = f"FPS {fps:.0f}  {'REC' if recording else 'idle'} ({frame_count})"
            cv2.putText(preview, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            cv2.imshow("chopari", preview)

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
            elif key == ord("t"):
                threading.Thread(target=joy.run_test, daemon=True).start()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()