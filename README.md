# FlyRider

초파리(*Drosophila*) 커넥톰을 계산 모델로 연결해 카트라이더를 조작하는 실험 프로젝트입니다. 게임 화면을 감각 입력으로 바꾸고, MaleCNS/FlyWire 계열의 뉴런 연결 구조를 희소 LIF 신경망으로 시뮬레이션한 뒤 하행성 뉴런 활동을 vJoy 입력으로 변환합니다.

> 현재는 연구·개발 단계입니다. 실제 초파리의 완전한 디지털 복제나 게임 플레이를 보장하는 완성형 AI가 아닙니다.

카트라이더 파일은 제공하지 않습니다. 합법적인 실행 가능한 카트라이더의 파일은 사용자가 직접 구해야 합니다. 
-디시인사이드 카트라이더 갤러리에 원하는 파일이 있을 수 도 있습니다.-

```text
카트라이더 화면 → 화면 특징 + Object AI → 감각 채널
       → 희소 connectome LIF → 운동 출력 → vJoy → 카트라이더
```

## 주요 기능

- `main.py`: 다크/라이트 테마 데스크톱 제어 앱. 주행, 뇌 상태 뷰어, Object AI 라벨링·학습 화면, 오버나이트, 기록 페이지를 한 창 안에서 관리합니다.
- `FlyRider_connectome.py`: 화면 캡처, 감각 입력, connectome 시뮬레이션, 보상·가소성 학습, vJoy 출력. `--overnight` 옵션으로 오버나이트도 실행합니다.
- `brain_viewer.py`: 뇌 활동, 감각 입력, 보상, 방향키 형태의 실시간 출력 모니터
- `ai_traing/state_object_ai.py`: 객체 라벨링, YOLO 학습, 모델 실행 미리보기, 이전 라벨 변환
- `run.bat`: Windows 환경 확인, 의존성 설치, 데이터 준비, 메인 프로그램 실행
- Object AI: `data/best.pt`가 있으면 벽/앞 장애물 세그멘테이션을 감각 입력에 사용
- 온라인 학습: 보상과 끼임·탈출 상태를 이용해 선택된 connectome 연결 가중치를 갱신

## 요구 사항

- Windows 10/11, Python 3.10 이상 권장
- 카트라이더 클라이언트
- vJoy 드라이버와 활성화된 vJoy Device 1
- `data/neurons.csv`, `data/connections_princeton.csv`
- Python 패키지: [`requirements.txt`](requirements.txt)

CUDA가 설치된 PyTorch 환경이면 Object AI 추론은 GPU를 사용합니다. connectome 계산 자체는 현재 NumPy/SciPy 기반 CPU 계산입니다.

## 설치

아래 배치 파일을 실행하면 자동으로 설치 됩니다.
```text
run.bat
```
수동 설치 방법
```powershell
git clone https://github.com/illililiiililii/FlyRider.git
cd FlyRider
python -m pip install -r requirements.txt
```

## 실행

1. vJoy를 설치하고 Device 1을 활성화합니다.
2. 카트라이더를 실행합니다.
3. `run.bat` 또는 `python main.py`를 실행합니다.
4. 처음 실행할 때 게임 화면 영역을 선택합니다. 영역은 `data/flyrider_config.json`에 저장됩니다.
5. 주행 페이지에서 `엔진과 함께 자동 주행 시작`을 켜면 엔진 시작과 동시에 조작합니다. 끄면 엔진만 켜지고 `F8` 또는 버튼을 눌러 주행을 시작합니다.
6. `Object AI 디버그 창 표시` 토글은 시각화 창만 제어합니다. 꺼도 Object AI 추론은 계속 감각 입력으로 사용됩니다.
7. 뇌 상태, Object AI, 오버나이트, 실행 로그는 메인 앱의 왼쪽 메뉴에서 이동합니다. 뇌 뷰어와 라벨링·학습 화면도 메인 창 안에 표시됩니다. Object AI 실시간 인식 미리보기와 수동 라벨 편집 화면은 별도 창을 사용합니다.

| 키 | 동작 |
|---|---|
| `F8` | 자동 주행 시작/정지 전환 |
| `F9` | vJoy 입력 즉시 해제 (엔진은 계속 실행) |
| `F10` | connectome 캐시 삭제 후 재생성 준비 |
| `F11` | 학습·보상 통계 출력 |
| `F7` | Object AI 디버그 창 켜기/끄기 (7개 클래스·박스/마스크·벽 감각값) |
| `F12` | 현재 프레임을 수동 라벨링 검토함에 저장 (자동 학습 안 함) |
| `q` | 종료 |
| `s` | 화면 저장 |
| `r` | 녹화 시작/정지 |
| `c` | 게임 화면 영역 다시 선택 |

## 오버나이트 주행

메인 앱 왼쪽 메뉴에서 **오버나이트**를 선택해 실행 시간을 설정하고 시작합니다. 실행 중인 엔진은 주행 페이지의 **저장 후 안전 종료** 버튼으로 정지합니다. `F9`는 입력만 즉시 해제하며 엔진은 계속 실행합니다. 이전 실행 명령도 유지됩니다.

```powershell
python overnight_trainer_ui.py
```

`overnight_trainer_ui.py`는 메인 앱의 오버나이트 페이지를 바로 여는 호환 실행 파일입니다. 끼임 복구와 저장 종료는 `FlyRider_connectome.py --overnight`가 처리합니다.

끼임은 정지와 벽 감지를 함께 확인합니다. 먼저 connectome 내부 상태를 초기화하고 회복을 확인합니다. 회복하지 못하면 선택한 경로로 구조 입력을 한 번 보내고 화면 움직임을 검증합니다. 2회 연속 실패하면 입력을 해제하고 저장한 뒤 안전 정지합니다.

기본 구조 입력은 Windows `R` 키 이벤트입니다. 게임이 이를 무시하면 GUI에서 `vJoy 버튼 8`을 선택할 수 있습니다. 이 경로를 쓰려면 카트라이더 컨트롤 설정에서 vJoy 버튼 8을 구조/리셋 동작에 먼저 지정해야 합니다. vJoy 드라이버가 이미 설치되어 있다면 이 방식은 재부팅 없이 시험할 수 있습니다. Object AI 추론은 최신 프레임만 별도 작업 흐름에서 처리하고, 디버그 화면에는 지연 상태도 표시합니다.

오버나이트 중에도 최신 프레임 추론은 계속 사용하지만, 추론 결과를 자동으로 학습 데이터에 추가하지 않습니다. 새로운 라벨을 직접 만들고 학습하려면 메인 앱의 **Object AI** 메뉴를 사용합니다.

## vJoy 입력 매핑

현재 구현은 카트라이더의 디지털 조작을 기준으로 합니다.

| 동작 | vJoy 입력 |
|---|---|
| 좌/우 조향 | X축 최소값 / 최대값 |
| 조향 중립 | X축 중앙값 |
| 전진 | Y축 전진 끝값 유지 |
| 후진 | Y축 후진 끝값 유지 |
| 앞뒤 중립 | Y축 중앙값 |

카트라이더 설정에서 vJoy의 Y축을 가속/후진 축으로 지정해야 합니다. 전진·후진 방향은 게임의 축 반전 설정에 따라 다를 수 있습니다. 뇌 뷰어의 방향 패드는 프로그램이 vJoy에 보낸 상태를 보여 주며 게임의 실제 움직임까지 측정하지는 않습니다.

약한 조향 신호는 좌우 축을 짧게 반복하고, 강한 신호는 계속 유지합니다. 전진은 기본 주행 상태이며, 후진은 MDN 계열 readout이 높은 확신으로 우세할 때 선택됩니다.

## Object AI 학습

메인 앱의 **Object AI** 메뉴에서 라벨링, 학습, 실행 미리보기, 결과 확인, 이전 라벨 변환을 할 수 있습니다. `python ai_traing/state_object_ai.py`를 직접 실행하는 기존 방법도 지원합니다. 기존 이미지 폴더를 가져오면 프로젝트 데이터셋에 복사하며 원본은 보존합니다. 라벨 없는 이미지는 검토 대기 항목으로 표시되어 수동 라벨링할 수 있습니다. 주행 중 수집된 장면은 `ai_traing/data/state_ai/inbox`에 저장하고, 주행을 멈춘 뒤 가져와 직접 라벨링합니다. 추론 결과를 자동으로 학습 데이터에 추가하지 않습니다.

```text
FlyRider/data/best.pt
```

게임 화면이 작다면 라벨링과 실행에서 같은 화면 영역을 사용해야 합니다.

## 프로젝트 구조

```text
FlyRider/
├─ main.py
├─ FlyRider_connectome.py
├─ object_vision.py
├─ brain_viewer.py
├─ overnight_trainer_ui.py  (호환 실행 파일)
├─ FlyRider_connectome_overnight.py  (엔진 호환 실행 파일)
├─ run.bat
├─ requirements.txt
├─ ai_traing/state_object_ai.py
├─ data/
│  ├─ neurons.csv
│  ├─ connections_princeton.csv
│  ├─ best.pt
│  ├─ flyrider_config.json
│  └─ flyrider_connectome_cache_v*.npz
└─ captures/
```

캐시는 실행용 희소 부분 그래프입니다. 출력 readout 구성이 바뀌면 `F10`으로 다시 만들 수 있습니다. 학습 가중치는 `data/flyrider_connectome_learning_v*.npz/json`에 저장됩니다.

## 동작 원리와 한계

원본 커넥톰은 뉴런 연결 구조와 세포 주석을 제공합니다. 현재 프로젝트는 실행 가능한 일부 연결을 추출하고 LIF 막전위·스파이크·시냅스 전파·보상 가소성을 별도의 계산 모델로 구현합니다. 화면 입력과 `DNp20`, `DNpe017`, `MDN` readout을 좌우·전진·후진 입력에 연결하는 부분도 공학적 인터페이스입니다.

따라서 Object AI 오검출, vJoy 설정, 게임 컨트롤러 바인딩에 따라 결과가 달라질 수 있습니다. 전체 커넥톰이 아닌 희소 부분 그래프를 사용하며, 보상과 온라인 학습 값은 계속 조정 중입니다.

## 개발 방향

실제 감각 뉴런에 가까운 시각 입력, 더 정확한 MaleCNS 운동 회로, GPU 기반 connectome 계산, 게임 상태 피드백, 조향·속도·드리프트 readout을 단계적으로 검토합니다.

## 라이선스와 기여

커넥톰 데이터의 출처와 라이선스는 사용하는 MaleCNS/FlyWire 배포본의 공식 안내를 따릅니다. 대용량 원본 데이터의 재배포 조건을 확인한 뒤 사용하십시오.

실행 로그, 재현 절차, vJoy 설정, 뇌 뷰어 화면을 포함한 Issue와 Pull Request를 환영합니다. 문제를 보고할 때 Windows/Python/PyTorch 버전, vJoy 설정, 사용한 cache와 `best.pt` 버전을 함께 적어 주세요.
