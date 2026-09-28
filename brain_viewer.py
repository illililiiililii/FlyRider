# -*- coding: utf-8 -*-
"""초파리라이더 REAL CONNECTOME 상태 뷰어

실행:
    python brain_viewer.py

메인 프로그램(chopari_rider_connectome.py)이 실행 중이면
http://127.0.0.1:8765/state 를 10회/초 정도로 읽어서 시각화합니다.
"""
import json
import math
import time
import urllib.request
import urllib.error
import tkinter as tk
from tkinter import ttk

API_URL = "http://127.0.0.1:8765/state"
REFRESH_MS = 100

BG = "#101318"
PANEL = "#181d24"
TEXT = "#e8edf3"
MUTED = "#8c98a8"
GRID = "#2b333e"
GOOD = "#4dd091"
WARN = "#f3c969"
BAD = "#ef6a6a"
ACCENT = "#72a7ff"


class Viewer:
    def __init__(self, root):
        self.root = root
        root.title("Chopari Rider — Connectome Brain Viewer")
        root.geometry("1180x760")
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", root.destroy)

        self.status = tk.StringVar(value="API 연결 대기 중...")
        self.labels = {}
        self.last_state = None

        top = tk.Frame(root, bg=BG)
        top.pack(fill="x", padx=14, pady=(12, 8))
        tk.Label(top, text="REAL CONNECTOME BRAIN", bg=BG, fg=TEXT,
                 font=("Segoe UI", 17, "bold")).pack(side="left")
        tk.Label(top, textvariable=self.status, bg=BG, fg=MUTED,
                 font=("Segoe UI", 10)).pack(side="right")

        self.canvas = tk.Canvas(root, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=14, pady=4)

        bottom = tk.Frame(root, bg=BG)
        bottom.pack(fill="x", padx=14, pady=(4, 12))
        self.detail = tk.Label(bottom, text="", bg=BG, fg=MUTED,
                               justify="left", anchor="w", font=("Consolas", 9))
        self.detail.pack(fill="x")

        self.fetch()

    def fetch(self):
        try:
            with urllib.request.urlopen(API_URL, timeout=0.4) as r:
                state = json.loads(r.read().decode("utf-8"))
            self.last_state = state
            self.status.set("● CONNECTED   127.0.0.1:8765")
            self.draw(state)
        except Exception:
            self.status.set("○ MAIN PROGRAM 대기 중   python chopari_rider_connectome.py")
            self.draw(None)
        self.root.after(REFRESH_MS, self.fetch)

    def panel(self, x1, y1, x2, y2, title):
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=PANEL, outline=GRID, width=1)
        self.canvas.create_text(x1 + 12, y1 + 12, anchor="nw", text=title,
                                fill=MUTED, font=("Segoe UI", 9, "bold"))

    def bar(self, x, y, w, h, value, color, label, right=None):
        value = max(0.0, min(1.0, float(value)))
        self.canvas.create_text(x, y - 1, anchor="sw", text=label,
                                fill=TEXT, font=("Segoe UI", 9))
        self.canvas.create_rectangle(x, y + 5, x + w, y + h, fill="#222934", outline="")
        self.canvas.create_rectangle(x, y + 5, x + w * value, y + h, fill=color, outline="")
        if right is None:
            right = f"{value:.2f}"
        self.canvas.create_text(x + w, y - 1, anchor="se", text=right,
                                fill=MUTED, font=("Consolas", 9))

    def draw(self, s):
        self.canvas.delete("all")
        w = max(800, self.canvas.winfo_width())
        h = max(560, self.canvas.winfo_height())

        if not s:
            self.panel(20, 20, w - 20, h - 20, "CONNECTOME STATUS")
            self.canvas.create_text(w / 2, h / 2, text="메인 프로그램의 API를 기다리는 중...",
                                    fill=MUTED, font=("Segoe UI", 14))
            return

        # ---------- top metrics ----------
        self.panel(20, 20, w - 20, 118, "OUTPUT / REWARD")
        drive = s.get("drive", "-").upper()
        steer = float(s.get("steer", 0))
        reward = float(s.get("reward", 0))
        total = float(s.get("reward_total", 0))
        fps = float(s.get("fps", 0))
        driving = s.get("driving", False)
        color = GOOD if drive == "FORWARD" else WARN
        self.canvas.create_text(42, 60, anchor="w", text=f"{drive}", fill=color,
                                font=("Segoe UI", 20, "bold"))
        self.canvas.create_text(165, 62, anchor="w", text=f"steer {steer:+.3f}",
                                fill=TEXT, font=("Consolas", 13, "bold"))
        self.canvas.create_text(360, 62, anchor="w", text=f"reward {reward:+.3f}",
                                fill=GOOD if reward >= 0 else BAD, font=("Consolas", 12))
        self.canvas.create_text(560, 62, anchor="w", text=f"total {total:+.2f}",
                                fill=TEXT, font=("Consolas", 12))
        self.canvas.create_text(760, 62, anchor="w", text=f"FPS {fps:.1f}",
                                fill=MUTED, font=("Consolas", 11))
        self.canvas.create_text(950, 62, anchor="w", text="RUN" if driving else "IDLE",
                                fill=GOOD if driving else MUTED, font=("Segoe UI", 11, "bold"))

        # ---------- inputs ----------
        self.panel(20, 135, 350, h - 30, "VISUAL INPUT / WALL SENSOR")
        inp = s.get("inputs", {})
        road = inp.get("road", [0] * 8)
        x = 42
        y = 175
        for i, v in enumerate(road):
            self.bar(x, y + i * 42, 275, 17, v, ACCENT, f"ROAD {i}")
        self.bar(x, y + 8 * 42, 275, 17, inp.get("stall", 0), BAD, "STALL")
        self.bar(x, y + 9 * 42, 275, 17, inp.get("wall_l", 0), WARN, "WALL LEFT")
        self.bar(x, y + 10 * 42, 275, 17, inp.get("wall_r", 0), WARN, "WALL RIGHT")
        self.canvas.create_text(x, y + 11 * 42 + 8, anchor="w",
                                text=f"coverage {inp.get('coverage', 0):.3f}",
                                fill=MUTED, font=("Consolas", 9))

        # ---------- brain activity ----------
        bx1, bx2 = 370, w - 310
        self.panel(bx1, 135, bx2, h - 30, "ACTUAL CONNECTOME ACTIVITY")
        self.canvas.create_text(bx1 + 18, 165, anchor="w",
                                text=f"neurons {s.get('neurons',0):,}   edges {s.get('edges',0):,}   plastic {s.get('plastic_edges',0):,}",
                                fill=MUTED, font=("Consolas", 9))

        active = s.get("active", [])
        # 실제 활성 뉴런을 ID 기반으로 안정적인 x 위치에 표시
        gx, gy = bx1 + 20, 195
        gw, gh = max(240, bx2 - bx1 - 40), h - 230
        self.canvas.create_rectangle(gx, gy, gx + gw, gy + gh, fill="#11161c", outline=GRID)
        count = min(len(active), 160)
        if count:
            cols = 16
            rows = math.ceil(count / cols)
            cellw = gw / cols
            cellh = min(18, gh / max(1, rows))
            for i, n in enumerate(active[:count]):
                c = i % cols
                r = i // cols
                rate = max(0.0, min(1.0, float(n.get("rate", 0)) * 5.0))
                xx = gx + c * cellw + cellw / 2
                yy = gy + r * cellh + cellh / 2
                rr = 3 + 5 * rate
                col = GOOD if rate > 0.45 else ACCENT
                self.canvas.create_oval(xx - rr, yy - rr, xx + rr, yy + rr,
                                        fill=col, outline="")
            self.canvas.create_text(gx + 8, gy + gh - 8, anchor="sw",
                                    text=f"현재 spike {s.get('active_count',0):,}개 / 표시 {count}개",
                                    fill=MUTED, font=("Consolas", 8))
        else:
            self.canvas.create_text(gx + gw / 2, gy + gh / 2,
                                    text="활성 spike 없음", fill=MUTED,
                                    font=("Segoe UI", 11))

        # ---------- outputs ----------
        ox1, ox2 = w - 290, w - 20
        self.panel(ox1, 135, ox2, h - 30, "DESCENDING OUTPUT")
        top = s.get("top_outputs", [])
        oy = 180
        for i, n in enumerate(top[:12]):
            rate = max(0, min(1, float(n.get("rate", 0)) * 4))
            sign = float(n.get("sign", 0))
            col = GOOD if sign > 0 else ACCENT
            self.canvas.create_text(ox1 + 14, oy + i * 35, anchor="w",
                                    text=str(n.get("id")), fill=TEXT,
                                    font=("Consolas", 8))
            self.canvas.create_rectangle(ox1 + 14, oy + 12 + i * 35,
                                         ox2 - 14, oy + 20 + i * 35,
                                         fill="#222934", outline="")
            self.canvas.create_rectangle(ox1 + 14, oy + 12 + i * 35,
                                         ox1 + 14 + (ox2 - ox1 - 28) * rate,
                                         oy + 20 + i * 35,
                                         fill=col, outline="")
        if not top:
            self.canvas.create_text((ox1 + ox2) / 2, 300, text="descending output 대기",
                                    fill=MUTED)

        active_ids = s.get("last_output_ids", [])
        detail = (
            f"active descending: {active_ids[:12]}\n"
            f"input {s.get('input_count',0):,} / output {s.get('output_count',0):,} / DA {s.get('da_count',0):,} | "
            f"events {s.get('event_count',0):,}"
        )
        self.detail.config(text=detail)


if __name__ == "__main__":
    root = tk.Tk()
    Viewer(root)
    root.mainloop()
