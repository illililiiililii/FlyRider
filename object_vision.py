# -*- coding: utf-8 -*-
"""Latest-frame-only YOLO segmentation worker shared by both driving programs."""
import threading
import time

import cv2
import numpy as np

CLASS_NAMES = ["나", "드리프트 중인 나", "앞 벽", "왼쪽 벽", "오른쪽 벽", "장애물", "상대"]
COLORS = [(0, 255, 0), (220, 100, 255), (0, 80, 255), (255, 80, 0),
          (0, 220, 220), (255, 180, 0), (255, 0, 180)]


class ObjectVision:
    def __init__(self, model_path, conf=0.10):
        self.model = None
        self.loaded = False
        self.error = ""
        self.device = "cpu"
        self.conf = conf
        self.debug_enabled = False
        self.lock = threading.Lock()
        self.pending = None
        self.result = self._empty()
        self.closed = False
        self.thread = None
        try:
            from ultralytics import YOLO
            import torch
            self.model = YOLO(str(model_path))
            names = self.model.names
            if isinstance(names, dict):
                names = [names[i] for i in range(len(names))]
            names = [str(x).strip() for x in names]
            if names != CLASS_NAMES:
                raise ValueError(f"클래스 이름/번호 불일치: 모델={names}, 코드={CLASS_NAMES}")
            if torch.cuda.is_available():
                self.device = "cuda:0"
            self.model.to(self.device)
            self.loaded = True
            self.thread = threading.Thread(target=self._worker, daemon=True, name="ObjectAI-inference")
            self.thread.start()
            print(f"[ObjectAI] class map OK (0-6), inference device={self.device}")
        except Exception as exc:
            self.error = str(exc)
            print(f"[ObjectAI] disabled: {self.error}")

    @staticmethod
    def _empty():
        return {"left_wall": 0., "right_wall": 0., "front_wall": 0., "drifting": 0.,
                "detections": [], "masks": [], "ready": False, "timestamp": 0., "error": ""}

    def submit(self, frame):
        if self.loaded:
            # Replacing this one pending slot drops stale frames under GPU/CPU load.
            with self.lock:
                self.pending = (frame.copy(), time.monotonic())

    def _worker(self):
        while not self.closed:
            with self.lock:
                item, self.pending = self.pending, None
            if item is None:
                time.sleep(.005)
                continue
            frame, captured_at = item
            try:
                h, w = frame.shape[:2]
                result = self.model.predict(frame, conf=self.conf, imgsz=640,
                                            device=self.device, verbose=False)[0]
                # Wall occupancy is accumulated at small resolution, never full frame.
                sw = max(1, min(384, w)); sh = max(1, round(h * sw / w))
                walls = np.zeros((sh, sw), np.float32)
                detections, masks = [], []
                drifting = 0.
                front_box_score = 0.0
                if result.boxes is not None:
                    for i, box in enumerate(result.boxes):
                        cls, score = int(box.cls[0]), float(box.conf[0])
                        # coordinates are already in original frame pixels
                        xyxy = [int(v) for v in box.xyxy[0].tolist()]
                        if not 0 <= cls < len(CLASS_NAMES):
                            continue
                        detections.append({"class": cls, "name": CLASS_NAMES[cls], "confidence": score,
                                           "box": xyxy})
                        if cls == 2:
                            x1, y1, x2, y2 = xyxy
                            cx = ((x1 + x2) * 0.5) / max(1, w)
                            bottom = y2 / max(1, h)
                            area = max(0, x2-x1) * max(0, y2-y1) / max(1, w*h)
                            center_weight = float(np.clip(1.0 - abs(cx - .5) / .42, 0.0, 1.0))
                            proximity = float(np.clip((bottom - .28) / .62, 0.0, 1.0))
                            size_weight = float(np.clip(np.sqrt(area / .12), .15, 1.0))
                            front_box_score = max(front_box_score,
                                                  score * center_weight * proximity * size_weight)
                        if cls == 1:
                            drifting = max(drifting, score)
                        if result.masks is not None:
                            small = cv2.resize(result.masks.data[i].cpu().numpy(), (sw, sh), interpolation=cv2.INTER_LINEAR)
                            if cls in (2, 3, 4):
                                walls = np.maximum(walls, small * score)
                            if self.debug_enabled:
                                masks.append({"class": cls, "confidence": score, "mask": small >= .5})
                region = lambda y0, y1, x0, x1: float(walls[int(sh*y0):max(int(sh*y0)+1,int(sh*y1)), int(sw*x0):max(int(sw*x0)+1,int(sw*x1))].mean())
                scale = lambda v: float(np.clip((v - .012) / .16, 0., 1.))
                self.result = {"left_wall": scale(region(.25,.92,0,.44)),
                               "right_wall": scale(region(.25,.92,.56,1)),
                               "front_wall": max(scale(region(.25,.88,.20,.80)), front_box_score),
                               "drifting": drifting, "detections": detections, "masks": masks,
                               "ready": True, "timestamp": captured_at, "error": ""}
            except Exception as exc:
                self.result = self._empty(); self.result["error"] = str(exc)

    def analyze(self, frame):
        self.submit(frame)
        result = dict(self.result)
        result["age"] = time.monotonic() - result["timestamp"] if result["timestamp"] else 1e9
        result["stale"] = result["age"] > 1.5
        return result

    def close(self):
        self.closed = True
        if self.thread:
            self.thread.join(timeout=2)

    @staticmethod
    def render(frame, result):
        h, w = frame.shape[:2]
        scale = min(1.0, 640.0 / w)
        view = cv2.resize(frame, (max(1, int(w*scale)), max(1, int(h*scale))))
        sh, sw = view.shape[:2]
        for item in result.get("masks", []):
            mask = cv2.resize(item["mask"].astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
            color = np.zeros_like(view); color[:] = COLORS[item["class"]]
            view[mask] = cv2.addWeighted(view[mask], .55, color[mask], .45, 0)
        for item in result.get("detections", []):
            cls = item["class"]; x1,y1,x2,y2 = item["box"]
            box = tuple(int(v*scale) for v in (x1,y1,x2,y2))
            cv2.rectangle(view, box[:2], box[2:], COLORS[cls], 2)
            cv2.putText(view, f"{item['name']} {item['confidence']:.2f}", (box[0], max(14,box[1]-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, .42, COLORS[cls], 1, cv2.LINE_AA)
        counts = [0]*7
        for d in result.get("detections", []): counts[d["class"]] += 1
        lines = ["갱신 지연" if result.get("stale") else "LIVE",
                 f"벽 감각 L {result.get('left_wall',0):.2f}  R {result.get('right_wall',0):.2f}  앞 {result.get('front_wall',0):.2f}"]
        lines += [f"{i} {name}: {sum(d['confidence'] for d in result.get('detections',[]) if d['class']==i)/counts[i]:.2f} ({counts[i]})" if counts[i] else f"{i} {name}: - (0)" for i,name in enumerate(CLASS_NAMES)]
        for i,line in enumerate(lines):
            cv2.putText(view, line, (7, 18+i*17), cv2.FONT_HERSHEY_SIMPLEX, .42, (255,255,255), 1, cv2.LINE_AA)
        return view

