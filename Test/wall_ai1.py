# -*- coding: utf-8 -*-
"""
초파리라이더 - Wall AI 학습기

핵심 변경점
-----------
- 라벨은 이미지별로 wall_ai_labels.json에 즉시 저장합니다.
- 다음/이전 이미지로 이동하기 전에 무조건 저장합니다.
- 프로그램 종료 시에도 저장합니다.
- 같은 이미지를 다시 열어도 SHA-1 이미지 키로 기존 라벨을 복원합니다.
- T(테스트)는 원본 이미지를 절대 덮어쓰지 않습니다.
- F(자동 라벨)는 학습된 모델의 확신도가 높은 부분만 현재 이미지에 자동 라벨링합니다.
- S(학습)는 저장된 모든 이미지 라벨에서 학습 데이터를 다시 만들어 모델을 저장합니다.

조작
----
1 : WALL
2 : ROAD
마우스 왼쪽 드래그 : 넓은 브러시로 라벨 추가 (한 줄만 그어도 다량의 샘플 생성)
SPACE / D : 다음 이미지 (이동 전 자동 저장)
A : 이전 이미지 (이동 전 자동 저장)
S : 학습 + 모델 저장
T : 현재 이미지 AI 테스트
O : 이미지 추가
X : 현재 이미지 라벨 전체 삭제
Ctrl+S : 라벨 강제 저장
ESC : 저장 후 종료

파일
----
wall_ai_labels.json      : 원본 라벨 데이터 (자동 저장)
wall_ai_labels.json.bak  : 직전 저장 백업
wall_ai_model.npz        : 학습된 Logistic Regression 모델
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path


def ensure_package(import_name: str, pip_name: str):
    try:
        __import__(import_name)
    except ImportError:
        print(f"[WallAI] {pip_name} 설치 중...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        __import__(import_name)


ensure_package("numpy", "numpy")
ensure_package("PIL", "pillow")

import numpy as np
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
from tkinter import messagebox, filedialog


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "wall_ai_model.npz"
LABEL_PATH = BASE_DIR / "wall_ai_labels.json"
LABEL_BACKUP_PATH = BASE_DIR / "wall_ai_labels.json.bak"

DEFAULT_IMAGES = [
    "shot_1790561024387.png",
    "shot_1790561041365.png",
    "shot_1790561064629.png",
    "shot_1790561073813.png",
]

PATCH_RADIUS = 7
EXPAND_OFFSETS = [
    (0, 0),
    (-5, 0), (5, 0), (0, -5), (0, 5),
    (-5, -5), (5, -5), (-5, 5), (5, 5),
]
MAX_POINTS_PER_IMAGE = 30000
MAX_TOTAL_POINTS = 500000

# 한 번 드래그했을 때 실제 라벨 영역을 넓게 잡습니다.
BRUSH_RADIUS = 12          # 원본 이미지 기준 반경
BRUSH_STEP = 4             # 드래그 선을 따라 찍는 간격
AUTO_GRID_W = 80
AUTO_GRID_H = 45
AUTO_HIGH = 0.82           # 이보다 높으면 WALL
AUTO_LOW = 0.18            # 이보다 낮으면 ROAD


# ============================================================
# Image features
# ============================================================

def rgb_to_hsv_np(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx = np.max(x, axis=-1)
    mn = np.min(x, axis=-1)
    d = mx - mn

    h = np.zeros_like(mx)
    mask = d > 1e-6
    mr = mask & (mx == r)
    mg = mask & (mx == g)
    mb = mask & (mx == b)
    h[mr] = ((g[mr] - b[mr]) / d[mr]) % 6.0
    h[mg] = ((b[mg] - r[mg]) / d[mg]) + 2.0
    h[mb] = ((r[mb] - g[mb]) / d[mb]) + 4.0
    h /= 6.0

    s = np.zeros_like(mx)
    nz = mx > 1e-6
    s[nz] = d[nz] / mx[nz]
    return np.stack([h, s, mx], axis=-1)


def grayscale(rgb: np.ndarray) -> np.ndarray:
    return (
        rgb[..., 0].astype(np.float32) * 0.299
        + rgb[..., 1].astype(np.float32) * 0.587
        + rgb[..., 2].astype(np.float32) * 0.114
    )


def feature_at(rgb: np.ndarray, x: int, y: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))

    r0 = max(0, y - PATCH_RADIUS)
    r1 = min(h, y + PATCH_RADIUS + 1)
    c0 = max(0, x - PATCH_RADIUS)
    c1 = min(w, x + PATCH_RADIUS + 1)

    patch = rgb[r0:r1, c0:c1].astype(np.float32)
    hsv = rgb_to_hsv_np(patch.astype(np.uint8))
    gray = grayscale(patch.astype(np.uint8))

    rgb_mean = patch.mean(axis=(0, 1)) / 255.0
    rgb_std = patch.std(axis=(0, 1)) / 255.0
    hsv_mean = hsv.mean(axis=(0, 1))
    hsv_std = hsv.std(axis=(0, 1))
    gmean = gray.mean() / 255.0
    gstd = gray.std() / 255.0

    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    edge_mean = float(np.mean(grad_mag)) / 255.0
    edge_std = float(np.std(grad_mag)) / 255.0
    dx = float(np.abs(np.diff(gray, axis=1)).mean() / 255.0) if gray.shape[1] > 1 else 0.0
    dy = float(np.abs(np.diff(gray, axis=0)).mean() / 255.0) if gray.shape[0] > 1 else 0.0

    nx = x / max(1, w - 1)
    ny = y / max(1, h - 1)
    center_dist = abs(nx - 0.5) * 2.0

    return np.array([
        *rgb_mean, *rgb_std,
        *hsv_mean, *hsv_std,
        gmean, gstd,
        edge_mean, edge_std,
        dx, dy,
        nx, ny, center_dist,
    ], dtype=np.float32)


# ============================================================
# Logistic Regression
# ============================================================

class WallClassifier:
    def __init__(self):
        self.w = None
        self.b = 0.0
        self.mu = None
        self.sigma = None

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y, epochs=900, lr=0.08, l2=0.001):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if len(X) < 10:
            raise ValueError("학습 샘플이 너무 적습니다.")
        if len(np.unique(y)) < 2:
            raise ValueError("WALL과 ROAD 샘플이 둘 다 필요합니다.")

        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0)
        self.sigma[self.sigma < 1e-6] = 1.0
        Z = (X - self.mu) / self.sigma

        rng = np.random.default_rng(1234)
        self.w = rng.normal(0, 0.03, size=Z.shape[1]).astype(np.float32)
        self.b = 0.0

        n = len(Z)
        pos = max(1, int(np.sum(y == 1)))
        neg = max(1, int(np.sum(y == 0)))
        wp = n / (2.0 * pos)
        wn = n / (2.0 * neg)
        sample_weight = np.where(y > 0.5, wp, wn).astype(np.float32)

        for epoch in range(epochs):
            logits = Z @ self.w + self.b
            p = self._sigmoid(logits)
            err = (p - y) * sample_weight
            grad_w = (Z.T @ err) / n + l2 * self.w
            grad_b = float(np.mean(err))
            self.w -= lr * grad_w
            self.b -= lr * grad_b

            if epoch % 100 == 0 or epoch == epochs - 1:
                pred = (p >= 0.5).astype(np.float32)
                acc = float(np.mean(pred == y))
                print(f"[WallAI] epoch {epoch:4d} | acc={acc*100:.1f}%")

    def predict_proba(self, X):
        if self.w is None:
            raise RuntimeError("모델이 학습되지 않았습니다.")
        X = np.asarray(X, dtype=np.float32)
        Z = (X - self.mu) / self.sigma
        return self._sigmoid(Z @ self.w + self.b)

    def predict_point(self, rgb, x, y):
        return float(self.predict_proba(feature_at(rgb, x, y)[None, :])[0])

    def save(self, path=MODEL_PATH):
        if self.w is None:
            raise RuntimeError("저장할 모델이 없습니다.")
        np.savez(path, w=self.w, b=np.array([self.b], dtype=np.float32), mu=self.mu, sigma=self.sigma)
        print(f"[WallAI] 모델 저장: {path}")

    def load(self, path=MODEL_PATH):
        data = np.load(path)
        self.w = data["w"].astype(np.float32)
        self.b = float(data["b"][0])
        self.mu = data["mu"].astype(np.float32)
        self.sigma = data["sigma"].astype(np.float32)
        self.sigma[self.sigma < 1e-6] = 1.0
        print(f"[WallAI] 모델 로드: {path}")


# ============================================================
# Runtime API
# ============================================================

class WallAI:
    def __init__(self, model_path=MODEL_PATH):
        self.model = WallClassifier()
        self.loaded = False
        if Path(model_path).exists():
            try:
                self.model.load(model_path)
                self.loaded = True
            except Exception as e:
                print(f"[WallAI] 모델 로드 실패: {e}")

    def analyze(self, frame):
        if not self.loaded:
            return {
                "left_wall": 0.0,
                "right_wall": 0.0,
                "front_block": 0.0,
                "center_wall": 0.0,
                "road_confidence": 0.0,
                "ready": False,
            }

        rgb = np.asarray(frame)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError("frame은 HxWx3 이미지여야 합니다.")
        rgb = rgb[:, :, :3]
        h, w = rgb.shape[:2]

        y_levels = [int(h * v) for v in (0.46, 0.52, 0.58, 0.64, 0.70)]
        left_xs = [int(w * v) for v in (0.08, 0.14, 0.20, 0.26, 0.32)]
        right_xs = [int(w * v) for v in (0.68, 0.74, 0.80, 0.86, 0.92)]
        center_xs = [int(w * v) for v in (0.40, 0.45, 0.50, 0.55, 0.60)]

        left_scores, right_scores, center_scores = [], [], []
        for y in y_levels:
            for x in left_xs:
                left_scores.append(self.model.predict_point(rgb, x, y))
            for x in right_xs:
                right_scores.append(self.model.predict_point(rgb, x, y))
            for x in center_xs:
                center_scores.append(self.model.predict_point(rgb, x, y))

        def weighted_mean(values):
            a = np.sort(np.asarray(values, dtype=np.float32))
            top = a[-max(3, len(a)//5):]
            return float(0.65 * np.mean(top) + 0.35 * np.mean(a))

        left = weighted_mean(left_scores)
        right = weighted_mean(right_scores)
        center = weighted_mean(center_scores)
        return {
            "left_wall": float(np.clip(left, 0.0, 1.0)),
            "right_wall": float(np.clip(right, 0.0, 1.0)),
            "front_block": float(np.clip(center, 0.0, 1.0)),
            "center_wall": float(np.clip(center, 0.0, 1.0)),
            "road_confidence": float(np.clip(1.0 - max(left, right, center), 0.0, 1.0)),
            "ready": True,
        }


# ============================================================
# Persistent trainer
# ============================================================

class Trainer:
    def __init__(self, root):
        self.root = root
        self.root.title("초파리라이더 - Wall AI 학습기 (자동저장)")
        self.root.geometry("1250x850")

        self.images: list[np.ndarray] = []
        self.image_names: list[str] = []
        self.image_paths: list[str] = []
        self.image_keys: list[str] = []
        self.index = 0

        # key -> [[x, y, label], ...]
        self.labels: dict[str, list[list[int]]] = {}
        self.load_labels()

        self.mode = 1
        self.model = WallClassifier()
        self.photo = None
        self.display_scale = 1.0
        self.display_w = 0
        self.display_h = 0
        self.dragging = False
        self.last_click_xy = None
        self.test_overlay = None
        self.autosave_job = None
        self.dirty = False

        self.canvas = tk.Canvas(root, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.info = tk.Label(root, text="", anchor="w", justify="left", font=("맑은 고딕", 11))
        self.info.pack(fill="x")

        self.root.bind("<KeyPress>", self.key)
        self.canvas.bind("<Button-1>", self.mouse_down)
        self.canvas.bind("<B1-Motion>", self.mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_up)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.load_default_images()
        if self.images:
            self.show_image()
        else:
            self.show_empty()

    # -----------------------------
    # Persistent labels
    # -----------------------------
    @staticmethod
    def image_key(rgb: np.ndarray) -> str:
        h = hashlib.sha1()
        h.update(str(rgb.shape).encode("utf-8"))
        h.update(rgb.tobytes())
        return h.hexdigest()

    def load_labels(self):
        path = LABEL_PATH
        data = None
        for candidate in (path, LABEL_BACKUP_PATH):
            if candidate.exists():
                try:
                    with candidate.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    print(f"[WallAI] 라벨 로드: {candidate}")
                    break
                except Exception as e:
                    print(f"[WallAI] 라벨 파일 읽기 실패 {candidate}: {e}")

        if not isinstance(data, dict):
            return

        raw = data.get("images", {})
        if not isinstance(raw, dict):
            return

        clean = {}
        for key, points in raw.items():
            if not isinstance(points, list):
                continue
            out = []
            for p in points:
                try:
                    if len(p) != 3:
                        continue
                    x, y, label = int(p[0]), int(p[1]), int(p[2])
                    if label not in (0, 1):
                        continue
                    out.append([x, y, label])
                except Exception:
                    continue
            if out:
                clean[str(key)] = out
        self.labels = clean

    def save_labels(self, reason="auto"):
        payload = {
            "version": 2,
            "description": "Wall/road raw labels. Automatically saved.",
            "images": self.labels,
        }
        tmp = LABEL_PATH.with_suffix(".tmp")
        try:
            text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            with tmp.open("w", encoding="utf-8") as f:
                f.write(text)

            # 드래그 중 자동저장은 빠르게 끝내야 하므로 fsync/백업 복사는 하지 않습니다.
            # 백업은 강제 저장/이미지 이동/종료 때만 수행합니다.
            if reason not in ("drag autosave",):
                if LABEL_PATH.exists():
                    try:
                        shutil.copy2(LABEL_PATH, LABEL_BACKUP_PATH)
                    except Exception:
                        pass
            os.replace(tmp, LABEL_PATH)
            self.dirty = False
            print(f"[WallAI] 라벨 저장 ({reason}): {LABEL_PATH}")
        except Exception as e:
            print(f"[WallAI] 라벨 저장 실패: {e}")
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

    def mark_dirty_and_autosave(self):
        self.dirty = True
        if self.autosave_job is not None:
            try:
                self.root.after_cancel(self.autosave_job)
            except Exception:
                pass
        # 드래그 중에는 마지막 입력 후 1.5초 뒤 한 번만 디스크에 기록
        self.autosave_job = self.root.after(1500, self._autosave_timer)

    def _autosave_timer(self):
        self.autosave_job = None
        if self.dirty:
            self.save_labels("drag autosave")

    def force_save(self, reason="manual"):
        if self.autosave_job is not None:
            try:
                self.root.after_cancel(self.autosave_job)
            except Exception:
                pass
            self.autosave_job = None
        self.save_labels(reason)

    # -----------------------------
    # Image loading
    # -----------------------------
    def append_image(self, path: str):
        try:
            im = Image.open(path).convert("RGB")
            rgb = np.array(im)
            self.images.append(rgb)
            self.image_names.append(Path(path).name)
            self.image_paths.append(str(Path(path).resolve()))
            self.image_keys.append(self.image_key(rgb))
            return True
        except Exception as e:
            print(f"[WallAI] 이미지 로드 실패 {path}: {e}")
            return False

    def load_default_images(self):
        for name in DEFAULT_IMAGES:
            p = BASE_DIR / name
            if p.exists():
                self.append_image(str(p))

        if not self.images:
            messagebox.showinfo(
                "Wall AI",
                "기본 스크린샷을 찾지 못했습니다.\n파일 선택창에서 카트라이더 스크린샷을 선택해주세요."
            )
            self.open_files()

    def open_files(self):
        self.force_save("before open")
        paths = filedialog.askopenfilenames(
            title="카트라이더 스크린샷 선택",
            filetypes=[("Image", "*.png *.jpg *.jpeg *.bmp"), ("All", "*.*")],
        )
        if not paths:
            return

        added = 0
        existing_keys = set(self.image_keys)
        for p in paths:
            try:
                im = Image.open(p).convert("RGB")
                rgb = np.array(im)
                key = self.image_key(rgb)
                if key in existing_keys:
                    continue
                self.images.append(rgb)
                self.image_names.append(Path(p).name)
                self.image_paths.append(str(Path(p).resolve()))
                self.image_keys.append(key)
                existing_keys.add(key)
                added += 1
            except Exception as e:
                print(f"[WallAI] 이미지 로드 실패 {p}: {e}")

        if added:
            self.index = len(self.images) - added
            self.test_overlay = None
            self.show_image()
        else:
            self.update_info()

    # -----------------------------
    # Display
    # -----------------------------
    def show_empty(self):
        self.canvas.delete("all")
        self.canvas.create_text(620, 300, text="스크린샷을 불러오세요", fill="white", font=("맑은 고딕", 24))
        self.update_info()

    def current_points(self):
        if not self.images:
            return []
        return self.labels.get(self.image_keys[self.index], [])

    def show_image(self):
        if not self.images:
            self.show_empty()
            return

        rgb = self.images[self.index]
        h, w = rgb.shape[:2]
        cw = max(400, self.root.winfo_width() - 20)
        ch = max(400, self.root.winfo_height() - 100)
        scale = min(cw / w, ch / h, 1.0)
        self.display_scale = scale
        self.display_w = max(1, int(w * scale))
        self.display_h = max(1, int(h * scale))

        if self.test_overlay is None:
            display_arr = rgb
        else:
            display_arr = self.test_overlay

        im = Image.fromarray(display_arr).resize((self.display_w, self.display_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(im)
        self.canvas.delete("all")
        self.canvas.config(width=self.display_w, height=self.display_h)
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

        # 저장된 라벨 점을 다시 표시
        for x, y, label in self.current_points():
            dx = x * self.display_scale
            dy = y * self.display_scale
            fill = "#ff3333" if label == 1 else "#33ff66"
            r = 3
            self.canvas.create_oval(dx-r, dy-r, dx+r, dy+r, fill=fill, outline=fill)

        self.update_info()

    def update_info(self):
        if self.images:
            name = self.image_names[self.index]
            pts = self.current_points()
            wall = sum(1 for p in pts if p[2] == 1)
            road = sum(1 for p in pts if p[2] == 0)
            total = sum(len(v) for v in self.labels.values())
            image_count = len(self.images)
            pos = self.index + 1
        else:
            name, wall, road, total, image_count, pos = "-", 0, 0, 0, 0, 0

        mode_text = "WALL" if self.mode == 1 else "ROAD"
        status = "저장됨" if not self.dirty else "저장 대기"
        self.info.config(
            text=(
                f"[{pos}/{image_count}] {name}    현재 모드: {mode_text}    "
                f"현재 이미지 WALL={wall} / ROAD={road}    전체 라벨={total}    [{status}]\n"
                f"1=WALL   2=ROAD   마우스 드래그=칠하기   "
                f"SPACE/D=다음   A=이전   S=학습+저장   T=AI 테스트   "
                f"O=이미지 추가   F=AI 자동 라벨   X=현재 이미지 라벨 삭제   Ctrl+S=강제 저장   ESC=종료"
            )
        )

    # -----------------------------
    # Labeling
    # -----------------------------
    def _append_raw_point(self, x, y, label, pts):
        """원본 라벨 점 1개 추가. JSON에는 중앙점만 저장합니다."""
        h, w = self.images[self.index].shape[:2]
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        label = int(label)
        # 같은 라벨의 아주 가까운 점은 중복 저장하지 않습니다.
        for q in pts[-80:]:
            if q[2] == label and abs(q[0] - x) <= 2 and abs(q[1] - y) <= 2:
                return False
        pts.append([x, y, label])
        return True

    def _add_label_at_display(self, dx, dy, redraw=False):
        if not self.images:
            return
        x = int(dx / max(self.display_scale, 1e-6))
        y = int(dy / max(self.display_scale, 1e-6))
        rgb = self.images[self.index]
        h, w = rgb.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return

        key = self.image_keys[self.index]
        pts = self.labels.setdefault(key, [])
        label = int(self.mode)

        # 중앙점 + 넓은 브러시 영역을 함께 기록합니다.
        # 사용자는 선 하나만 그어도 실제 학습에는 충분히 많은 점이 생깁니다.
        added = self._append_raw_point(x, y, label, pts)
        radius = BRUSH_RADIUS
        step = max(3, int(radius / 3))
        for oy in range(-radius, radius + 1, step):
            for ox in range(-radius, radius + 1, step):
                if ox * ox + oy * oy <= radius * radius:
                    if self._append_raw_point(x + ox, y + oy, label, pts):
                        added = True

        if len(pts) > MAX_POINTS_PER_IMAGE:
            del pts[:len(pts) - MAX_POINTS_PER_IMAGE]

        if added:
            self.mark_dirty_and_autosave()

        if redraw:
            fill = "#ff3333" if label == 1 else "#33ff66"
            r = max(3, int(3 / max(self.display_scale, 0.25)))
            # 드래그 위치에 브러시 외곽선을 보여줍니다.
            rr = max(4, int(BRUSH_RADIUS * self.display_scale))
            self.canvas.create_oval(dx-rr, dy-rr, dx+rr, dy+rr, outline=fill, width=2)

    def mouse_down(self, event):
        self.test_overlay = None
        self.dragging = True
        self.last_click_xy = (event.x, event.y)
        self._add_label_at_display(event.x, event.y, redraw=True)
        self.update_info()

    def mouse_drag(self, event):
        if not self.dragging:
            return
        x0, y0 = self.last_click_xy if self.last_click_xy else (event.x, event.y)
        x1, y1 = event.x, event.y
        dist = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, int(dist / BRUSH_STEP))
        for i in range(1, steps + 1):
            t = i / steps
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            self._add_label_at_display(x, y, redraw=False)
        self.last_click_xy = (x1, y1)
        self.update_info()

    def mouse_up(self, event):
        self.dragging = False
        self.last_click_xy = None
        self.force_save("mouse up")
        self.update_info()

    # -----------------------------
    # Semi-auto labeling
    # -----------------------------
    def auto_label_current(self):
        """학습된 모델로 확신도가 높은 영역만 자동 라벨링합니다."""
        if not self.images:
            return
        if self.model.w is None:
            if MODEL_PATH.exists():
                try:
                    self.model.load(MODEL_PATH)
                except Exception as e:
                    messagebox.showerror("Wall AI", str(e))
                    return
            else:
                messagebox.showwarning("Wall AI", "먼저 한 장 정도 직접 라벨링한 뒤 S로 학습하세요.")
                return

        rgb = self.images[self.index]
        h, w = rgb.shape[:2]
        key = self.image_keys[self.index]
        pts = self.labels.setdefault(key, [])
        added = 0

        # 격자 간격은 화면 해상도에 맞춰 충분히 촘촘하게 잡습니다.
        for gy in range(AUTO_GRID_H):
            y = int((gy + 0.5) / AUTO_GRID_H * h)
            for gx in range(AUTO_GRID_W):
                x = int((gx + 0.5) / AUTO_GRID_W * w)
                p = float(self.model.predict_point(rgb, x, y))
                if p >= AUTO_HIGH:
                    label = 1
                elif p <= AUTO_LOW:
                    label = 0
                else:
                    continue

                # 자동 라벨은 일정 간격으로만 저장해 JSON을 과도하게 키우지 않습니다.
                if self._append_raw_point(x, y, label, pts):
                    added += 1

        if len(pts) > MAX_POINTS_PER_IMAGE:
            del pts[:len(pts) - MAX_POINTS_PER_IMAGE]
        self.mark_dirty_and_autosave()
        self.test_overlay = None
        self.show_image()
        messagebox.showinfo(
            "Wall AI",
            f"확신도가 높은 영역을 자동 라벨링했습니다.\n\n추가된 라벨 점: {added}개\n\n빨간색=벽(WALL), 초록색=도로(ROAD)\n틀린 곳은 드래그해서 덧칠한 뒤 다시 학습하세요."
        )

    # -----------------------------
    # Training
    # -----------------------------
    def collect_training_data(self):
        X = []
        y = []
        total_points = 0
        # 원본 라벨 한 점을 중심으로 더 넓게 증식합니다.
        train_offsets = []
        for oy in range(-12, 13, 4):
            for ox in range(-12, 13, 4):
                if ox * ox + oy * oy <= 12 * 12:
                    train_offsets.append((ox, oy))

        for idx, rgb in enumerate(self.images):
            key = self.image_keys[idx]
            for x, yy, label in self.labels.get(key, []):
                total_points += 1
                h, w = rgb.shape[:2]
                for ox, oy in train_offsets:
                    xx = int(np.clip(x + ox, 0, w - 1))
                    y2 = int(np.clip(yy + oy, 0, h - 1))
                    X.append(feature_at(rgb, xx, y2))
                    y.append(label)
        if not X:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32), 0
        return np.stack(X), np.asarray(y, dtype=np.float32), total_points

    def train(self):
        self.force_save("before training")
        X, y, point_count = self.collect_training_data()
        if len(y) < 40:
            messagebox.showwarning("Wall AI", "WALL과 ROAD를 각각 한두 줄씩이라도 더 라벨링해주세요.")
            return
        try:
            self.model.fit(X, y)
            self.model.save(MODEL_PATH)
            wall = int(np.sum(y == 1))
            road = int(np.sum(y == 0))
            messagebox.showinfo(
                "Wall AI",
                f"학습 완료!\n\n원본 라벨 점: {point_count}\n학습 샘플: {len(y)}\nWALL: {wall}\nROAD: {road}\n\n라벨 저장:\n{LABEL_PATH}\n모델 저장:\n{MODEL_PATH}"
            )
        except Exception as e:
            messagebox.showerror("Wall AI", str(e))

    # -----------------------------
    # Test without modifying source image
    # -----------------------------
    def test(self):
        if self.model.w is None:
            if MODEL_PATH.exists():
                try:
                    self.model.load(MODEL_PATH)
                except Exception as e:
                    messagebox.showerror("Wall AI", str(e))
                    return
            else:
                messagebox.showwarning("Wall AI", "먼저 S로 학습하세요.")
                return

        rgb = self.images[self.index]
        h, w = rgb.shape[:2]
        small_w, small_h = 80, 45
        scores = np.zeros((small_h, small_w), dtype=np.float32)
        for yy in range(small_h):
            for xx in range(small_w):
                scores[yy, xx] = self.model.predict_point(rgb, int(xx / small_w * w), int(yy / small_h * h))

        base = Image.fromarray(rgb).convert("RGBA")
        heat = Image.new("RGBA", base.size, (0, 0, 0, 0))
        hd = ImageDraw.Draw(heat)
        for yy in range(small_h):
            for xx in range(small_w):
                p = float(scores[yy, xx])
                if p < 0.5:
                    continue
                alpha = int(np.clip((p - 0.5) * 2.0, 0, 1) * 150)
                x0 = int(xx / small_w * w)
                y0 = int(yy / small_h * h)
                x1 = int((xx + 1) / small_w * w)
                y1 = int((yy + 1) / small_h * h)
                hd.rectangle((x0, y0, x1, y1), fill=(255, 50, 50, alpha))

        self.test_overlay = np.array(Image.alpha_composite(base, heat).convert("RGB"))
        self.show_image()

    def clear_current_labels(self):
        if not self.images:
            return
        key = self.image_keys[self.index]
        count = len(self.labels.get(key, []))
        if count == 0:
            return
        if not messagebox.askyesno("Wall AI", f"현재 이미지의 라벨 {count}개를 모두 삭제할까요?"):
            return
        self.labels.pop(key, None)
        self.force_save("clear current image")
        self.test_overlay = None
        self.show_image()

    # -----------------------------
    # Keys / navigation
    # -----------------------------
    def go_next(self):
        if not self.images:
            return
        self.force_save("before next image")
        self.index = (self.index + 1) % len(self.images)
        self.test_overlay = None
        self.show_image()

    def go_prev(self):
        if not self.images:
            return
        self.force_save("before previous image")
        self.index = (self.index - 1) % len(self.images)
        self.test_overlay = None
        self.show_image()

    def key(self, event):
        k = event.keysym.lower()
        if k == "1":
            self.mode = 1
            self.update_info()
        elif k == "2":
            self.mode = 0
            self.update_info()
        elif k == "space" or k == "d":
            self.go_next()
        elif k == "a":
            self.go_prev()
        elif k == "s":
            self.train()
        elif k == "t":
            self.test()
        elif k == "f":
            self.auto_label_current()
        elif k == "o":
            self.open_files()
        elif k == "x":
            self.clear_current_labels()
        elif k == "escape":
            self.on_close()

    def on_close(self):
        self.force_save("program exit")
        self.root.destroy()


def main():
    root = tk.Tk()
    Trainer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
