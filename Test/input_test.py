# vjoy_test.py  (초파리라이더 - 1단계: vJoy 동작 확인)
# 반드시 관리자 권한으로 실행한 터미널에서 돌리세요.
import sys
import time

try:
    import pyvjoy
except ImportError:
    print("pyvjoy가 없습니다. 먼저: pip install pyvjoy")
    sys.exit(1)

DEVICE_ID = 1
AXIS_MIN = 0x0001
AXIS_CENTER = 0x4000
AXIS_MAX = 0x8000


def countdown(sec):
    for i in range(sec, 0, -1):
        print(f"  {i}...")
        time.sleep(1)


def main():
    try:
        j = pyvjoy.VJoyDevice(DEVICE_ID)
    except Exception as e:
        print("vJoy 장치를 열지 못했습니다:", e)
        print("확인: vJoy 설치 여부, vJoyInterface.dll 위치, Configure vJoy에서 장치 1 활성화")
        sys.exit(1)

    j.reset()
    print("vJoy 장치 열기 성공")
    print("joy.cpl > vJoy Device > 속성 창을 열어 두세요.")
    print("테스트는 카트라이더 키 설정 화면을 띄워 둔 상태로 해도 됩니다.")
    print("5초 뒤 시작합니다.")
    countdown(5)

    steps = [
        ("가운데", AXIS_CENTER, 2),
        ("왼쪽", AXIS_MIN, 4),
        ("가운데", AXIS_CENTER, 1),
        ("오른쪽", AXIS_MAX, 4),
        ("가운데", AXIS_CENTER, 1),
    ]
    for name, value, wait in steps:
        print(f"X축 {name}")
        j.set_axis(pyvjoy.HID_USAGE_X, value)
        time.sleep(wait)

    for btn in (1, 2, 3):
        print(f"버튼 {btn} 누름")
        j.set_button(btn, 1)
        time.sleep(1)
        j.set_button(btn, 0)
        time.sleep(0.5)

    j.reset()
    print("테스트 끝")


if __name__ == "__main__":
    main()