# -*- coding: utf-8 -*-
"""FlyRider object detection workflow: timed capture, manual labels, YOLO train/run."""
import json
import random
import shutil
import sys
import uuid
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import numpy as np
from mss import mss

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "state_ai"
IMAGES = DATA / "images"
LABELS = DATA / "labels"
MODELS = DATA / "models"
YAML = DATA / "dataset.yaml"
CFG = BASE / "data" / "flyrider_config.json"
NAMES = ["나", "드리프트 중인 나", "앞 벽", "왼쪽 벽", "오른쪽 벽", "장애물", "상대"]
WALL_CLASSES = {2, 3, 4}
WALL_CLASSES = {2, 3, 4}
KEYS = {ord(str(i + 1)): i for i in range(len(NAMES))}


def dirs():
    for path in (IMAGES, LABELS, MODELS):
        path.mkdir(parents=True, exist_ok=True)


def load_region():
    try:
        obj = json.loads(CFG.read_text(encoding="utf-8"))
        region = obj.get("region")
        if isinstance(region, dict) and all(k in region for k in ("left", "top", "width", "height")):
            return {k: int(region[k]) for k in ("left", "top", "width", "height")}
    except Exception:
        pass
    return None


def save_region(region):
    CFG.parent.mkdir(parents=True, exist_ok=True)
    try:
        config = json.loads(CFG.read_text(encoding="utf-8")) if CFG.exists() else {}
    except Exception:
        config = {}
    config["region"] = region
    CFG.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def select_region(sct):
    monitor = sct.monitors[1]
    frame = cv2.cvtColor(np.asarray(sct.grab(monitor)), cv2.COLOR_BGRA2BGR)
    scale = min(1.0, 1600 / frame.shape[1])
    shown = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1 else frame
    cv2.namedWindow("게임 화면 영역 선택", cv2.WINDOW_NORMAL)
    x, y, w, h = cv2.selectROI("게임 화면 영역 선택", shown, False, False)
    cv2.destroyWindow("게임 화면 영역 선택")
    if not w or not h:
        return None
    return {"left": int(monitor["left"] + x / scale), "top": int(monitor["top"] + y / scale),
            "width": int(w / scale), "height": int(h / scale)}


def screenshot(sct, region):
    region = region or sct.monitors[1]
    return cv2.cvtColor(np.asarray(sct.grab(region)), cv2.COLOR_BGRA2BGR)


def next_id():
    ids = []
    for p in IMAGES.glob("*.jpg"):
        try:
            ids.append(int(p.stem))
        except ValueError:
            pass
    return max(ids, default=0) + 1


class BoxLabeler:
    def __init__(self, frame, index, total):
        self.frame = frame
        self.boxes = []
        self.masks = {cls: np.zeros(frame.shape[:2], dtype=np.uint8) for cls in WALL_CLASSES}
        self.start = None
        self.cursor = None
        self.painting = False
        self.paint_class = 2
        self.brush_radius = max(5, min(frame.shape[:2]) // 70)
        self.finished = False
        self.saved = False
        self.window = "FlyRider 수동 라벨링"
        self.index, self.total = index, total
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.mouse)

    def mouse(self, event, x, y, flags, _):
        if self.painting:
            if event == cv2.EVENT_LBUTTONDOWN:
                self._paint(x, y)
            elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
                self._paint(x, y)
            elif event == cv2.EVENT_RBUTTONDOWN:
                for mask in self.masks.values():
                    cv2.circle(mask, (x, y), self.brush_radius * 2, 0, -1)
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
            self.cursor = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.start:
            self.cursor = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.start:
            x1, x2 = sorted((self.start[0], x)); y1, y2 = sorted((self.start[1], y))
            if x2 - x1 >= 4 and y2 - y1 >= 4:
                self.boxes.append([-1, x1, y1, x2, y2])
            self.start = self.cursor = None
        elif event == cv2.EVENT_RBUTTONDOWN and self.boxes:
            self.boxes.pop()

    def render(self):
        image = self.frame.copy()
        colors = {2: (0, 80, 255), 3: (255, 80, 0), 4: (0, 220, 220)}
        for cls, mask in self.masks.items():
            active = mask > 0
            if np.any(active):
                tint = np.zeros_like(image)
                tint[:] = colors[cls]
                image[active] = cv2.addWeighted(image[active], .55, tint[active], .45, 0)
        cv2.rectangle(image, (0, 0), (image.shape[1], 105), (18, 18, 18), -1)
        cv2.putText(image, f"라벨링 {self.index}/{self.total}   1 나  2 드리프트  3 앞벽  4 왼쪽벽  5 오른쪽벽  6 장애물  7 상대",
                    (8, 24), 0, .52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, "B=벽 붓 모드 전환 · 붓 모드에서 3/4/5=벽 종류, 좌클릭 드래그=칠하기, 우클릭=지우기",
                    (8, 51), 0, .48, (255, 240, 170), 1, cv2.LINE_AA)
        cv2.putText(image, "일반 모드: 드래그=박스, 숫자=최근 박스 클래스 · R=마지막 박스 삭제 · Enter=저장 · S=건너뜀 · Esc=종료",
                    (8, 78), 0, .43, (220, 220, 220), 1, cv2.LINE_AA)
        for i, (cls, x1, y1, x2, y2) in enumerate(self.boxes):
            color = (0, 255, 0) if cls >= 0 else (0, 220, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = NAMES[cls] if cls >= 0 else "클래스 선택"
            cv2.putText(image, f"{i + 1}: {label}", (x1, max(100, y1 - 6)), 0, .5, color, 1, cv2.LINE_AA)
        if self.start and self.cursor:
            cv2.rectangle(image, self.start, self.cursor, (255, 255, 0), 2)
        return image

    def run(self):
        while True:
            cv2.imshow(self.window, self.render())
            key = cv2.waitKey(20) & 255
            if key == 27:
                self.finished = True
                break
            if key in (ord("b"), ord("B")):
                self.painting = not self.painting
                continue
            if self.painting and key in (ord("3"), ord("4"), ord("5")):
                self.paint_class = {ord("3"): 2, ord("4"): 3, ord("5"): 4}[key]
                continue
            if self.painting and key in (ord("c"), ord("C")):
                self.masks = {cls: np.zeros(self.frame.shape[:2], dtype=np.uint8) for cls in WALL_CLASSES}
                continue
            if key in KEYS:
                for box in reversed(self.boxes):
                    if box[0] < 0:
                        box[0] = KEYS[key]
                        break
            elif key in (ord("r"), ord("R")) and self.boxes:
                self.boxes.pop()
            elif key in (ord("s"), ord("S")):
                break
            elif key in (10, 13):
                if any(b[0] < 0 for b in self.boxes):
                    continue
                self.saved = True
                break
        cv2.destroyWindow(self.window)
        return self.boxes

    def _paint(self, x, y):
        cv2.circle(self.masks[self.paint_class], (x, y), self.brush_radius, 255, -1)


def save_sample(frame, boxes, masks, index):
    h, w = frame.shape[:2]
    image_path = IMAGES / f"{index:07d}.jpg"
    label_path = LABELS / f"{index:07d}.txt"
    cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    lines = []
    for cls, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 10:
                continue
            contour = cv2.approxPolyDP(contour, max(1.0, cv2.arcLength(contour, True) * .003), True)
            points = contour.reshape(-1, 2)
            if len(points) >= 3:
                coords = " ".join(f"{x / w:.8f} {y / h:.8f}" for x, y in points)
                lines.append(f"{cls} {coords}")
    for cls, x1, y1, x2, y2 in boxes:
        if cls in WALL_CLASSES or not (0 <= cls < len(NAMES) and x2 > x1 and y2 > y1):
            continue
        coords = " ".join(f"{x / w:.8f} {y / h:.8f}" for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)))
        lines.append(f"{cls} {coords}")
    label_path.write_text("\n".join(lines), encoding="utf-8")
    return image_path


def import_image_folder(source):
    """Copy images and neighboring YOLO labels into this dataset; never edit source."""
    dirs()
    source = Path(source).resolve()
    image_files = [p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in
                   {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    excluded = (IMAGES.resolve(), (DATA / "dataset").resolve(), MODELS.resolve())
    image_files = [p for p in image_files
                   if not any(root == p.resolve() or root in p.resolve().parents for root in excluded)]
    index = next_id(); copied = labeled = 0
    for src in image_files:
        frame = cv2.imread(str(src))
        if frame is None:
            continue
        target = IMAGES / f"{index:07d}.jpg"
        cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        relative_label = src.relative_to(source).with_suffix(".txt")
        label_candidates = [src.with_suffix(".txt"), source.parent / "labels" / relative_label,
                            source.parent.parent / "labels" / relative_label]
        label = next((candidate for candidate in label_candidates if candidate.is_file()), None)
        if label:
            shutil.copy2(label, LABELS / f"{index:07d}.txt")
            labeled += 1
        index += 1; copied += 1
    return copied, labeled, copied - labeled


def review_unlabeled(status):
    dirs()
    images = sorted(p for p in IMAGES.glob("*.jpg")
                    if not (LABELS / f"{p.stem}.txt").exists()
                    or not (LABELS / f"{p.stem}.txt").read_text(encoding="utf-8").strip())
    saved = 0
    for pos, path in enumerate(images, 1):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        labeler = BoxLabeler(frame, pos, len(images)); boxes = labeler.run()
        if labeler.finished:
            break
        if labeler.saved:
            # save_sample writes the already-imported dataset image and its label.
            save_sample(frame, boxes, labeler.masks, int(path.stem))
            saved += 1
        status(f"검토 {pos}/{len(images)} · 라벨 저장 {saved}장")
    return saved, len(images)


def convert_legacy_labels():
    """Convert old YOLO detection boxes to rectangular YOLO segmentation polygons."""
    dirs()
    converted = 0
    skipped = 0
    backup_dir = LABELS / "backup_before_segmentation"
    for image in IMAGES.glob("*.jpg"):
        label_path = LABELS / f"{image.stem}.txt"
        if not label_path.exists():
            continue
        raw = label_path.read_text(encoding="utf-8")
        rows = [line.split() for line in raw.splitlines() if line.strip()]
        if not rows:
            continue
        # Polygon rows have class plus at least three x/y coordinate pairs.
        if all(len(row) >= 7 and len(row) % 2 == 1 for row in rows):
            skipped += 1
            continue
        parsed = []
        for row in rows:
            if len(row) != 5:
                raise RuntimeError(f"{label_path.name}에 해석할 수 없는 라벨 행이 있습니다: {' '.join(row)}")
            try:
                cls = int(row[0]); cx, cy, bw, bh = map(float, row[1:])
            except ValueError as exc:
                raise RuntimeError(f"{label_path.name}에 숫자가 아닌 라벨 값이 있습니다.") from exc
            if not 0 <= cls < len(NAMES) or bw <= 0 or bh <= 0:
                raise RuntimeError(f"{label_path.name}에 범위를 벗어난 라벨이 있습니다: {' '.join(row)}")
            x1, y1 = max(0.0, cx - bw / 2), max(0.0, cy - bh / 2)
            x2, y2 = min(1.0, cx + bw / 2), min(1.0, cy + bh / 2)
            if x2 <= x1 or y2 <= y1:
                raise RuntimeError(f"{label_path.name}에 면적이 없는 박스가 있습니다.")
            parsed.append((cls, x1, y1, x2, y2))
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / label_path.name
        if not backup_path.exists():
            shutil.copy2(label_path, backup_path)
        polygon_rows = []
        for cls, x1, y1, x2, y2 in parsed:
            polygon_rows.append(f"{cls} {x1:.8f} {y1:.8f} {x2:.8f} {y1:.8f} {x2:.8f} {y2:.8f} {x1:.8f} {y2:.8f}")
        label_path.write_text("\n".join(polygon_rows), encoding="utf-8")
        converted += 1
    return converted, skipped, backup_dir


def collect_for_duration(region, seconds, interval, status):
    dirs()
    if region is None:
        raise RuntimeError("먼저 게임 화면 영역을 지정하세요.")
    frames = []
    start = time.monotonic()
    next_capture = start
    with mss() as sct:
        while True:
            now = time.monotonic()
            remaining = seconds - (now - start)
            if remaining <= 0:
                break
            if now >= next_capture:
                frames.append(screenshot(sct, region))
                next_capture += interval
                status(f"캡처 중: {len(frames)}장 · 남은 시간 {max(0, remaining):.0f}초")
            time.sleep(min(.08, max(.01, next_capture - time.monotonic())))
    status(f"캡처 완료: {len(frames)}장. 라벨링 창을 엽니다.")
    index = next_id()
    saved = 0
    for pos, frame in enumerate(frames, 1):
        labeler = BoxLabeler(frame, pos, len(frames))
        boxes = labeler.run()
        if labeler.finished:
            break
        if labeler.saved:
            path = save_sample(frame, boxes, labeler.masks, index)
            index += 1
            saved += 1
            status(f"라벨 저장 {saved}장: {path.name}")
        else:
            status(f"건너뜀 {pos}/{len(frames)}")
    status(f"라벨링 완료: {saved}장 저장")
    return saved


def make_dataset():
    dirs()
    files = [p for p in IMAGES.glob("*.jpg") if (LABELS / f"{p.stem}.txt").exists()]
    files = [p for p in files if (LABELS / f"{p.stem}.txt").read_text(encoding="utf-8").strip()]
    if len(files) < 10:
        raise RuntimeError(f"학습 가능한 이미지가 {len(files)}장뿐입니다. 최소 10장 필요합니다.")
    for image in files:
        for line_no, line in enumerate((LABELS / f"{image.stem}.txt").read_text(encoding="utf-8").splitlines(), 1):
            values = line.split()
            if len(values) < 7 or len(values) % 2 == 0:
                raise RuntimeError(f"{image.stem}.txt {line_no}행이 이전 박스 라벨 형식입니다. 새로 라벨링하거나 이전 라벨을 변환해야 합니다.")
    random.shuffle(files)
    split = max(1, int(len(files) * .8))
    train_files, val_files = files[:split], files[split:] or files[:1]
    root = DATA / "dataset"
    if root.exists():
        shutil.rmtree(root)
    for path in (root / "images/train", root / "images/val", root / "labels/train", root / "labels/val"):
        path.mkdir(parents=True, exist_ok=True)
    for subset, name in ((train_files, "train"), (val_files, "val")):
        for image in subset:
            shutil.copy2(image, root / "images" / name / image.name)
            shutil.copy2(LABELS / f"{image.stem}.txt", root / "labels" / name / f"{image.stem}.txt")
    names = "\n".join(f"  {i}: '{label}'" for i, label in enumerate(NAMES))
    YAML.write_text(f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n", encoding="utf-8")
    return YAML, len(files), len(train_files), len(val_files)


def train_model(model_name, epochs, imgsz, batch, status):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics 패키지가 필요합니다. Python 환경에서 pip install ultralytics를 실행하세요.") from exc
    yaml, total, train_count, val_count = make_dataset()
    status(f"학습 시작: 전체 {total}장 (train {train_count}, val {val_count})")
    # A previously trained .pt must be resumed with Ultralytics' resume mode.
    # Passing the checkpoint as a normal model name together with
    # ``pretrained=True`` can make Windows/PyTorch try to reopen the checkpoint
    # with incompatible arguments and results in OSError [Errno 22].
    model_name = str(model_name or "").strip().strip('"')
    checkpoint = Path(model_name).expanduser() if model_name else None
    if checkpoint is not None and checkpoint.suffix.lower() == ".pt":
        candidates = [checkpoint] if checkpoint.is_absolute() else [
            Path.cwd() / checkpoint, BASE / checkpoint, BASE.parent / checkpoint]
        checkpoint = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
        if checkpoint is None:
            entered = str(model_name)
            raise FileNotFoundError(
                f"체크포인트 파일을 찾을 수 없습니다: {entered}\n"
                f"확인한 기준 폴더: {Path.cwd()} · {BASE} · {BASE.parent}\n"
                "학습된 best.pt를 선택하거나 기본 세그 모델을 입력하세요.")
    is_checkpoint = checkpoint is not None
    model = YOLO(str(checkpoint) if is_checkpoint else (model_name or str(BASE.parent / "yolo11n-seg.pt")))
    run_name = f"flyrider_object_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:5]}"
    train_args = dict(data=str(yaml), epochs=epochs, imgsz=imgsz, batch=batch, workers=0,
                      project=str(MODELS), name=run_name, exist_ok=False)
    if is_checkpoint:
        # Treat a selected checkpoint as starting weights. `resume=True` expects
        # a compatible Ultralytics training run and fails on a generic base .pt.
        status(f"체크포인트 가중치에서 추가 학습: {checkpoint}")
        train_args["pretrained"] = True
    else:
        train_args["pretrained"] = True
    model.train(**train_args)
    return MODELS / run_name / "weights" / "best.pt"


def latest_model():
    candidates = list(MODELS.glob("flyrider_object*/weights/best.pt"))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def run_preview(model_path, region, confidence, status):
    sys.path.insert(0, str(BASE.parent))
    from object_vision import ObjectVision
    path = Path(model_path) if model_path else latest_model()
    if path is None or not path.exists():
        raise RuntimeError(f"학습된 모델을 찾지 못했습니다: {MODELS}")
    if region is None:
        raise RuntimeError("먼저 게임 화면 영역을 지정하세요.")
    vision = ObjectVision(path, conf=confidence)
    if not vision.loaded:
        raise RuntimeError(f"Object AI를 시작할 수 없습니다: {vision.error}")
    vision.debug_enabled = True
    status(f"실시간 인식 중 · {path.name} (창에서 ESC를 누르면 종료)")
    try:
        with mss() as sct:
            while True:
                frame = screenshot(sct, region)
                result = vision.analyze(frame)
                cv2.imshow("FlyRider AI 인식 미리보기", ObjectVision.render(frame, result))
                if cv2.waitKey(1) & 255 == 27:
                    break
    finally:
        vision.close()
        cv2.destroyAllWindows()
    status("실시간 인식 미리보기 종료")


class App:
    def __init__(self, root, embedded=False):
        self.root = root
        if not embedded:
            self.root.title("FlyRider 오브젝트 AI")
            self.root.geometry("900x680")
            self.root.minsize(700, 520)
        self.region = load_region()
        self.busy = False
        self.status_var = tk.StringVar(value="게임 화면 영역을 지정한 뒤 라벨링을 시작하세요.")
        self.region_var = tk.StringVar(value=self.describe_region())
        self.duration_var = tk.StringVar(value="60")
        self.interval_var = tk.StringVar(value="3")
        self.epochs_var = tk.StringVar(value="80")
        existing_model = latest_model()
        self.model_var = tk.StringVar(value=str(existing_model or (BASE.parent / "yolo11n-seg.pt")))
        self.conf_var = tk.StringVar(value="0.15")
        self.build_ui()
        self.refresh_results()

    def describe_region(self):
        r = self.region
        return "지정 안 됨" if not r else f"left={r['left']}, top={r['top']}, {r['width']}×{r['height']}"

    def build_ui(self):
        root = self.root
        ttk.Label(root, text="FlyRider 오브젝트 AI", font=("맑은 고딕", 18, "bold")).pack(anchor="w", padx=20, pady=(16, 3))
        ttk.Label(root, text="화면을 지정하고 캡처 · 수동 라벨링 · YOLO 학습 · 인식 미리보기를 진행합니다.").pack(anchor="w", padx=20, pady=(0, 12))
        area = ttk.LabelFrame(root, text="게임 화면", padding=10); area.pack(fill="x", padx=20, pady=5)
        row = ttk.Frame(area); row.pack(fill="x")
        ttk.Button(row, text="화면 영역 선택", command=self.choose_region).pack(side="left")
        ttk.Label(row, textvariable=self.region_var).pack(side="left", padx=12)
        capture = ttk.LabelFrame(root, text="1. 라벨링", padding=10); capture.pack(fill="x", padx=20, pady=5)
        ttk.Label(capture, text="캡처 시간(초)").grid(row=0, column=0, sticky="w")
        ttk.Entry(capture, textvariable=self.duration_var, width=8).grid(row=0, column=1, padx=(5, 14))
        ttk.Label(capture, text="간격(초)").grid(row=0, column=2, sticky="w")
        ttk.Entry(capture, textvariable=self.interval_var, width=6).grid(row=0, column=3, padx=5)
        ttk.Button(capture, text="캡처 후 수동 라벨링", command=self.start_collection).grid(row=0, column=4, padx=(10, 0))
        ttk.Label(capture, text="차량·장애물은 박스, 벽은 B 키로 붓 모드를 켜고 칠하세요.").grid(row=1, column=0, columnspan=5, sticky="w", pady=(8, 0))
        ttk.Button(capture, text="이전 라벨 세그 형식으로 변환", command=self.convert_labels).grid(row=2, column=0, sticky="w", pady=(7, 0))
        ttk.Button(capture, text="기존 이미지 폴더 가져오기", command=self.import_folder).grid(row=2, column=1, columnspan=2, sticky="w", pady=(7, 0))
        ttk.Button(capture, text="라벨 없는 이미지 검토", command=self.start_review).grid(row=2, column=3, columnspan=2, sticky="w", pady=(7, 0))
        train = ttk.LabelFrame(root, text="2. 학습", padding=10); train.pack(fill="x", padx=20, pady=5)
        ttk.Label(train, text="YOLO 세그 모델").grid(row=0, column=0, sticky="w")
        ttk.Entry(train, textvariable=self.model_var, width=45).grid(row=0, column=1, padx=5)
        ttk.Label(train, text="Epoch").grid(row=0, column=2, padx=(8, 0))
        ttk.Entry(train, textvariable=self.epochs_var, width=7).grid(row=0, column=3, padx=5)
        ttk.Button(train, text="학습 시작", command=self.start_training).grid(row=0, column=4, padx=(10, 0))
        run = ttk.LabelFrame(root, text="3. 실행", padding=10); run.pack(fill="x", padx=20, pady=5)
        ttk.Label(run, text="신뢰도 기준").pack(side="left")
        ttk.Entry(run, textvariable=self.conf_var, width=7).pack(side="left", padx=6)
        ttk.Button(run, text="실시간 인식 미리보기", command=self.start_preview).pack(side="left", padx=8)
        result = ttk.LabelFrame(root, text="4. 결과", padding=10); result.pack(fill="both", expand=True, padx=20, pady=5)
        self.result_text = tk.Text(result, height=5, wrap="word", state="disabled", font=("Consolas", 9))
        self.result_text.pack(fill="both", expand=True)
        ttk.Button(result, text="결과 폴더 열기", command=self.open_results).pack(anchor="e", pady=(5, 0))
        ttk.Label(root, textvariable=self.status_var, relief="sunken", anchor="w").pack(fill="x", side="bottom")

    def set_status(self, text):
        self.root.after(0, self.status_var.set, text)

    def choose_region(self):
        try:
            with mss() as sct:
                region = select_region(sct)
            if region:
                self.region = region
                save_region(region)
                self.region_var.set(self.describe_region())
                self.set_status("화면 영역을 저장했습니다.")
        except Exception as exc:
            messagebox.showerror("화면 영역 선택", str(exc))

    def start_job(self, fn, done_message):
        if self.busy:
            return
        self.busy = True
        def worker():
            try:
                result = fn()
                if done_message:
                    self.root.after(0, lambda: messagebox.showinfo("완료", done_message(result)))
            except Exception as exc:
                self.root.after(0, lambda err=str(exc): messagebox.showerror("오류", err))
                self.set_status(f"오류: {exc}")
            finally:
                self.busy = False
                self.root.after(0, self.refresh_results)
        threading.Thread(target=worker, daemon=True).start()

    def start_collection(self):
        try:
            seconds, interval = float(self.duration_var.get()), float(self.interval_var.get())
            if seconds <= 0 or interval <= 0:
                raise ValueError("시간은 0보다 커야 합니다.")
        except ValueError as exc:
            messagebox.showerror("입력 확인", str(exc)); return
        self.start_job(lambda: collect_for_duration(self.region, seconds, interval, self.set_status),
                       lambda n: f"수동 라벨링이 끝났습니다.\n저장된 이미지: {n}장\n\n{IMAGES}")

    def convert_labels(self):
        try:
            converted, skipped, backup = convert_legacy_labels()
            self.refresh_results()
            messagebox.showinfo("라벨 변환 완료",
                                f"변환한 라벨 파일: {converted}개\n이미 세그 형식이라 건너뜀: {skipped}개\n\n원본 백업:\n{backup}")
        except Exception as exc:
            messagebox.showerror("라벨 변환 오류", str(exc))

    def import_folder(self):
        source = filedialog.askdirectory(title="가져올 이미지 폴더 선택 (원본은 변경하지 않습니다)")
        if not source: return
        def job():
            result = import_image_folder(source)
            return result
        self.start_job(job, lambda r: f"이미지 {r[0]}장 복사 · 라벨 포함 {r[1]}장 · 검토 대기 {r[2]}장\n원본 폴더는 보존했습니다.")

    def start_review(self):
        self.start_job(lambda: review_unlabeled(self.set_status),
                       lambda r: f"검토 대상 {r[1]}장 · 라벨 저장 {r[0]}장")

    def start_training(self):
        try:
            epochs = int(self.epochs_var.get())
            if epochs < 1:
                raise ValueError("Epoch는 1 이상이어야 합니다.")
        except ValueError as exc:
            messagebox.showerror("입력 확인", str(exc)); return
        self.start_job(lambda: train_model(self.model_var.get().strip() or "yolo11n-seg.pt", epochs, 640, 8, self.set_status),
                       lambda p: f"학습 완료!\n모델 저장 위치:\n{p}")

    def start_preview(self):
        try:
            conf = float(self.conf_var.get())
            if not 0 <= conf <= 1:
                raise ValueError("신뢰도는 0~1 사이여야 합니다.")
        except ValueError as exc:
            messagebox.showerror("입력 확인", str(exc)); return
        self.start_job(lambda: run_preview(None, self.region, conf, self.set_status), None)

    def refresh_results(self):
        dirs()
        images = list(IMAGES.glob("*.jpg"))
        labels = list(LABELS.glob("*.txt"))
        unlabeled = sum(1 for image in images
                        if not (LABELS / f"{image.stem}.txt").exists()
                        or not (LABELS / f"{image.stem}.txt").read_text(encoding="utf-8").strip())
        class_rows = [0] * len(NAMES)
        for label_file in labels:
            for row in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = row.split()
                if parts and parts[0].isdigit() and 0 <= int(parts[0]) < len(class_rows):
                    class_rows[int(parts[0])] += 1
        model = latest_model()
        lines = [f"이미지 폴더: {IMAGES}", f"라벨 폴더: {LABELS}", f"데이터셋 설정: {YAML}",
                 f"캡처 이미지: {len(images)}장 · 라벨 파일: {len(labels)}개 · 검토 대기: {unlabeled}장",
                 "클래스 라벨 수: " + " · ".join(f"{i}:{name} {class_rows[i]}" for i, name in enumerate(NAMES)),
                 ("주의: 앞 벽 라벨이 적습니다. 앞벽 장면을 더 라벨링하면 인식 개선에 도움이 됩니다."
                  if class_rows[2] < 50 else ""),
                 f"주행 중 수집함: {DATA / 'inbox'} (F12로 수집 후 이 폴더를 가져오기)",
                 f"학습 모델: {model if model else '아직 학습 전'}"]
        self.result_text.configure(state="normal"); self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", "\n".join(lines)); self.result_text.configure(state="disabled")

    def open_results(self):
        dirs()
        import os
        os.startfile(str(DATA))


def main():
    dirs()
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
