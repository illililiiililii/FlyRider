# -*- coding: utf-8 -*-
"""FlyRider connectome and vJoy monitor. Run alongside FlyRider_connectome.py."""
import json
import math
import urllib.request
import tkinter as tk
from tkinter import ttk

API_URL = "http://127.0.0.1:8765/state"
REFRESH_MS = 120

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
    def __init__(self, root, embedded=False):
        self.root = root
        if not embedded:
            root.title("FlyRider — Connectome & Joystick Monitor")
            root.geometry("1060x740")
            root.minsize(700, 540)
            root.protocol("WM_DELETE_WINDOW", root.destroy)
        root.configure(bg=BG)
        self.status = tk.StringVar(value="API 연결 대기 중...")

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 7), background=PANEL, foreground=TEXT)
        style.map("TNotebook.Tab", background=[("selected", "#26364b")])

        self.header = tk.Frame(root, bg=BG)
        self.header.pack(fill="x", padx=12, pady=(10, 6))
        self.heading = tk.Label(self.header, text="FLYRIDER CONNECTOME", bg=BG, fg=TEXT,
                                font=("Segoe UI", 15, "bold"))
        self.heading.pack(side="left")
        self.status_label = tk.Label(self.header, textvariable=self.status, bg=BG, fg=MUTED,
                                      font=("Segoe UI", 9))
        self.status_label.pack(side="right")

        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        self.canvases = {}
        self.pages = {}
        for key, title in (("control", "조이스틱"), ("brain", "뇌 활동"), ("learning", "감각·학습")):
            page = tk.Frame(self.tabs, bg=BG)
            self.tabs.add(page, text=title)
            self.pages[key] = page
            canvas = tk.Canvas(page, bg=BG, highlightthickness=0)
            canvas.pack(fill="both", expand=True)
            canvas.bind("<Configure>", lambda _event: self.redraw())
            self.canvases[key] = canvas

        self.state = None
        self.fetch()

    def set_theme(self, colors):
        global BG, PANEL, TEXT, MUTED, GRID, GOOD, WARN, BAD, ACCENT
        BG, PANEL, TEXT, MUTED, GRID = (colors[k] for k in ("bg", "panel", "text", "muted", "line"))
        GOOD, WARN, BAD, ACCENT = (colors[k] for k in ("green", "warning", "danger", "accent"))
        self.root.configure(bg=BG)
        self.header.configure(bg=BG)
        self.heading.configure(bg=BG, fg=TEXT)
        self.status_label.configure(bg=BG, fg=MUTED)
        for key, page in self.pages.items():
            page.configure(bg=BG)
            self.canvases[key].configure(bg=BG)

    def fetch(self):
        try:
            with urllib.request.urlopen(API_URL, timeout=0.35) as response:
                self.state = json.loads(response.read().decode("utf-8"))
            self.status.set("● CONNECTED   127.0.0.1:8765")
        except Exception:
            self.state = None
            self.status.set("○ 메인 프로그램 대기 중")
        self.redraw()
        self.root.after(REFRESH_MS, self.fetch)

    @staticmethod
    def panel(c, x1, y1, x2, y2, title):
        c.create_rectangle(x1, y1, x2, y2, fill=PANEL, outline=GRID)
        c.create_text(x1 + 12, y1 + 12, anchor="nw", text=title,
                      fill=MUTED, font=("Segoe UI", 9, "bold"))

    @staticmethod
    def text(c, x, y, value, color=TEXT, size=10, bold=False, anchor="w", width=None):
        c.create_text(x, y, anchor=anchor, text=value, fill=color,
                      font=("Consolas" if size <= 10 else "Segoe UI", size,
                            "bold" if bold else "normal"), width=width)

    @staticmethod
    def bar(c, x, y, w, value, label, color=ACCENT, right=None):
        value = max(0.0, min(1.0, float(value)))
        c.create_text(x, y, anchor="w", text=label, fill=TEXT, font=("Segoe UI", 8))
        c.create_text(x + w, y, anchor="e", text=(right if right is not None else f"{value:.2f}"),
                      fill=MUTED, font=("Consolas", 8))
        c.create_rectangle(x, y + 5, x + w, y + 12, fill="#222934", outline="")
        if value:
            c.create_rectangle(x, y + 5, x + w * value, y + 12, fill=color, outline="")

    def redraw(self):
        if not hasattr(self, "canvases"):
            return
        if self.state is None:
            for c in self.canvases.values():
                c.delete("all")
                w, h = max(300, c.winfo_width()), max(200, c.winfo_height())
                self.panel(c, 12, 12, w - 12, h - 12, "CONNECTOME STATUS")
                self.text(c, w / 2, h / 2, "메인 프로그램의 API를 기다리는 중...",
                          MUTED, 13, anchor="center")
            return
        self.draw_control(self.canvases["control"], self.state)
        self.draw_brain(self.canvases["brain"], self.state)
        self.draw_learning(self.canvases["learning"], self.state)

    @staticmethod
    def dimensions(c):
        return max(640, c.winfo_width()), max(400, c.winfo_height())

    def draw_control(self, c, s):
        c.delete("all")
        w, h = self.dimensions(c)
        m, gap = 14, 12
        controller = s.get("controller", {})
        turn = int(controller.get("turn", 0))
        drive = str(controller.get("drive", s.get("drive", "idle"))).lower()
        available = bool(controller.get("available", False))
        driving = bool(s.get("driving", False))
        steer = float(s.get("steer", 0.0))

        self.panel(c, m, m, w - m, 95, "RUN / CONNECTOME OUTPUT")
        self.text(c, m + 18, 48, "RUN" if driving else "IDLE",
                  GOOD if driving else MUTED, 17, True)
        self.text(c, m + 120, 49, f"brain steer {steer:+.3f}")
        self.text(c, m + 290, 49, f"drive intent {str(s.get('drive', 'idle')).upper()}")
        self.text(c, m + 500, 49, f"reward {float(s.get('reward', 0)):+.3f}",
                  GOOD if float(s.get("reward", 0)) >= 0 else BAD)
        self.text(c, w - m - 18, 49, "vJoy 연결됨" if available else "vJoy 미연결",
                  GOOD if available else BAD, 9, anchor="e")

        top, bottom = 112, h - 14
        pad_right = min(w * 0.56, 500)
        pad_box = (m, top, pad_right, bottom)
        info_box = (pad_right + gap, top, w - m, bottom)
        self.panel(c, *pad_box, "LIVE DIRECTION KEYS")
        self.panel(c, *info_box, "OUTPUT DETAILS")

        drive_level = int(controller.get("drive_level", 1 if drive == "forward" else -1 if drive == "reverse" else 0))
        dy = -drive_level
        dx = turn
        if dx == 0 and dy == 0:
            active = "center"
        else:
            active = {( -1, -1): "up_left", (0, -1): "up", (1, -1): "up_right",
                      (-1, 0): "left", (1, 0): "right",
                      (-1, 1): "down_left", (0, 1): "down", (1, 1): "down_right"}.get((dx, dy), "center")
        pad_w = pad_box[2] - pad_box[0]
        cell = min(78, max(52, int((pad_w - 70) / 3)))
        gap_cell = 5
        grid_w = cell * 3 + gap_cell * 2
        gx = pad_box[0] + (pad_w - grid_w) / 2
        gy = top + max(56, (bottom - top - grid_w) / 2)
        keys = [
            ("up_left", "↖", "UP + LEFT"), ("up", "↑", "FORWARD"), ("up_right", "↗", "UP + RIGHT"),
            ("left", "←", "LEFT"), ("center", "●", "NEUTRAL"), ("right", "→", "RIGHT"),
            ("down_left", "↙", "DOWN + LEFT"), ("down", "↓", "REVERSE"), ("down_right", "↘", "DOWN + RIGHT"),
        ]
        for i, (key, arrow, label) in enumerate(keys):
            row, col = divmod(i, 3)
            x1, y1 = gx + col * (cell + gap_cell), gy + row * (cell + gap_cell)
            is_active = active == key
            color = GOOD if is_active else GRID
            c.create_rectangle(x1, y1, x1 + cell, y1 + cell,
                               fill="#243d35" if is_active else "#222934",
                               outline=color, width=3 if is_active else 1)
            self.text(c, x1 + cell / 2, y1 + cell * .37, arrow,
                      color if is_active else MUTED, 20, is_active, anchor="center")
            self.text(c, x1 + cell / 2, y1 + cell * .76, label,
                      color if is_active else MUTED, 7, is_active, anchor="center")

        x_value = int(controller.get("x_value", 0x4000))
        y_value = int(controller.get("y_value", 0x4000))
        motor = s.get("motor_readout", {})
        self.text(c, info_box[0] + 14, top + 58,
                  f"현재 입력: {active.upper().replace('_', ' + ')}", TEXT, 11, True)
        self.text(c, info_box[0] + 14, top + 92,
                  f"X 조향 {turn:+d}  ·  vJoy 0x{x_value:04X}", TEXT, 9)
        self.text(c, info_box[0] + 14, top + 118,
                  f"Y 주행 {drive_level:+d} ({drive.upper()}) · vJoy 0x{y_value:04X}",
                  TEXT, 8, width=info_box[2] - info_box[0] - 28)
        self.text(c, info_box[0] + 14, top + 144,
                  "전진/후진은 Y축 끝값을 계속 보내고, 중립일 때만 Y축 중앙값을 보냅니다.", MUTED, 8,
                  width=info_box[2] - info_box[0] - 28)
        self.text(c, info_box[0] + 14, top + 158,
                  f"DNpe017 전진 출력  {float(motor.get('forward_rate', 0)):.4f}", MUTED, 9)
        self.text(c, info_box[0] + 14, top + 184,
                  f"MDN 후진 출력       {float(motor.get('reverse_rate', 0)):.4f}", MUTED, 9)
        self.text(c, info_box[0] + 14, top + 220,
                  f"FPS {float(s.get('fps', 0)):.1f}  ·  vJoy {'연결됨' if available else '미연결'}",
                  GOOD if available else BAD, 9)
        self.text(c, info_box[0] + 14, top + 258,
                  "방향 패드는 프로그램이 보낸 입력입니다. 게임의 키 설정/축 바인딩과 실제 카트 움직임은 별도 확인이 필요합니다.",
                  MUTED, 8, width=info_box[2] - info_box[0] - 28)
        wall = s.get("wall_ai", {})
        inputs = s.get("inputs", {})
        self.text(c, info_box[0] + 14, top + 310,
                  f"벽 감지 L/R/F {float(inputs.get('wall_l', 0)):.2f} / {float(inputs.get('wall_r', 0)):.2f} / {float(inputs.get('front_wall', 0)):.2f}",
                  WARN, 8, width=info_box[2] - info_box[0] - 28)
        self.text(c, info_box[0] + 14, top + 336,
                  f"끼임 {float(inputs.get('stall', 0)):.2f} · Object AI {wall.get('device', 'n/a')} · {'ready' if wall.get('ready') else 'not ready'}",
                  MUTED, 8, width=info_box[2] - info_box[0] - 28)

    def draw_brain(self, c, s):
        c.delete("all")
        w, h = self.dimensions(c)
        m, gap = 14, 12
        top, mid = 14, max(250, int(h * 0.60))
        side_w = max(205, min(275, int(w * 0.29)))
        left = (m, top, m + side_w, mid)
        right = (m + side_w + gap, top, w - m, mid)
        self.panel(c, *left, "SENSORY INPUT")
        inp = s.get("inputs", {})
        x, y, bw = left[0] + 14, top + 42, side_w - 28
        step = min(23, max(14, (mid - y - 22) / 12))
        for i, value in enumerate(inp.get("road", [0] * 8)[:8]):
            self.bar(c, x, y + i * step, bw, value, f"ROAD {i}")
        for j, (label, value, color) in enumerate((
            ("STALL", inp.get("stall", 0), BAD),
            ("WALL LEFT", inp.get("wall_l", 0), WARN),
            ("WALL RIGHT", inp.get("wall_r", 0), WARN),
            ("FRONT", inp.get("front_wall", 0), BAD),
        )):
            self.bar(c, x, y + (8 + j) * step, bw, value, label, color)
        self.text(c, x, mid - 16, f"coverage {float(inp.get('coverage', 0)):.3f}", MUTED, 8)

        self.panel(c, *right, "ACTIVE CONNECTOME STATE")
        self.text(c, right[0] + 14, top + 34,
                  f"{s.get('neurons', 0):,} neurons · {s.get('edges', 0):,} edges · {s.get('active_count', 0):,} active",
                  MUTED, 9)
        gx, gy = right[0] + 14, top + 58
        gw, gh = right[2] - right[0] - 28, mid - gy - 12
        c.create_rectangle(gx, gy, gx + gw, gy + gh, fill="#11161c", outline=GRID)
        active = s.get("active", [])[:160]
        if active:
            cols = max(8, min(20, int(gw / 30)))
            rows = math.ceil(len(active) / cols)
            cellw, cellh = gw / cols, gh / max(1, rows)
            for i, neuron in enumerate(active):
                rate = max(0.0, min(1.0, float(neuron.get("rate", 0)) * 5))
                cx = gx + (i % cols + 0.5) * cellw
                cy = gy + (i // cols + 0.5) * cellh
                rad = max(2, min(7, 2 + 5 * rate))
                color = GOOD if rate >= 0.45 else ACCENT
                c.create_oval(cx - rad, cy - rad, cx + rad, cy + rad, fill=color, outline="")
        else:
            self.text(c, gx + gw / 2, gy + gh / 2, "활성 spike 없음", MUTED, 10, anchor="center")

        out_y = mid + 12
        self.panel(c, m, out_y, w - m, h - m, "MOTOR READOUT")
        readout = s.get("turn_readout", {})
        half = (w - 2 * m - gap) / 2
        self.text(c, m + 16, out_y + 42,
                  f"turn source: {readout.get('source', 'n/a')}   ·   L {float(readout.get('left_rate', 0)):.4f} / R {float(readout.get('right_rate', 0)):.4f}",
                  TEXT, 10, True)
        self.text(c, m + 16, out_y + 70,
                  "Top outputs: " + ", ".join(str(n.get("id")) for n in s.get("top_outputs", [])[:8]),
                  MUTED, 8, width=w - 2 * m - 32)
        self.text(c, m + 16, out_y + 94,
                  f"최근 활성 하행 뉴런 ID: {s.get('last_output_ids', [])[:8]}", MUTED, 8,
                  width=w - 2 * m - 32)

    def draw_learning(self, c, s):
        c.delete("all")
        w, h = self.dimensions(c)
        m, gap, top = 14, 12, 14
        third = (w - 2 * m - 2 * gap) / 3
        x1, x2, x3 = m, m + third + gap, m + 2 * (third + gap)
        lower = h - m

        self.panel(c, x1, top, x1 + third, lower, "REWARD THIS FRAME")
        rb = s.get("reward_breakdown", {})
        reward_rows = [("approach", "벽 접근"), ("stuck", "벽에 끼임"),
                       ("escape_success", "탈출 보상"), ("safe_drive", "안전 3초 주행")]
        y = top + 45
        inner_w = third - 28
        for i, (key, label) in enumerate(reward_rows):
            val = float(rb.get(key, 0.0))
            yy = y + i * 50
            c.create_text(x1 + 14, yy, anchor="w", text=label, fill=TEXT, font=("Segoe UI", 8))
            c.create_text(x1 + third - 14, yy, anchor="e", text=f"{val:+.4f}",
                          fill=GOOD if val >= 0 else BAD, font=("Consolas", 8))
            bar_y = yy + 9
            c.create_rectangle(x1 + 14, bar_y, x1 + third - 14, bar_y + 8, fill="#222934", outline="")
            scale = 0.12 if key == "escape_success" else 0.08
            ratio = min(1.0, abs(val) / scale)
            mid_x = x1 + third / 2
            if val >= 0:
                c.create_rectangle(mid_x, bar_y, mid_x + (inner_w / 2) * ratio,
                                   bar_y + 8, fill=GOOD, outline="")
            else:
                c.create_rectangle(mid_x - (inner_w / 2) * ratio, bar_y, mid_x,
                                   bar_y + 8, fill=BAD, outline="")
        self.text(c, x1 + 14, lower - 28,
                  f"합계 {float(s.get('reward', 0)):+.4f} · 누적 {float(s.get('reward_total', 0)):+.2f}",
                  TEXT, 8)

        self.panel(c, x2, top, x2 + third, lower, "OBJECT / WALL SENSE")
        wall = s.get("wall_ai", {})
        for i, (key, label, color) in enumerate((
            ("left", "LEFT WALL", WARN), ("right", "RIGHT WALL", WARN),
            ("front", "FRONT WALL", BAD), ("center", "CENTER", BAD),
            ("road_confidence", "ROAD CONF.", ACCENT),
        )):
            self.bar(c, x2 + 14, top + 54 + i * 42, third - 28,
                     wall.get(key, 0), label, color)
        self.text(c, x2 + 14, lower - 52,
                  f"model {'loaded' if wall.get('loaded') else 'not loaded'} · {'ready' if wall.get('ready') else 'waiting'}",
                  GOOD if wall.get("loaded") else MUTED, 8)
        self.text(c, x2 + 14, lower - 30,
                  f"stall {float(s.get('inputs', {}).get('stall', 0)):.2f}", MUTED, 8)

        self.panel(c, x3, top, w - m, lower, "ONLINE PLASTICITY")
        learn = s.get("learning", {})
        rows = [
            ("plastic edges", f"{learn.get('plastic_edges', 0):,}"),
            ("updates this frame", f"{learn.get('updates_this_frame', 0):,}"),
            ("updates total", f"{learn.get('total_updates', 0):,}"),
            ("avg |Δweight|", f"{learn.get('avg_delta', 0.0):.6f}"),
            ("max |Δweight|", f"{learn.get('max_delta', 0.0):.6f}"),
            ("weight mean", f"{learn.get('weight_mean', 0.0):+.5f}"),
            ("weight std", f"{learn.get('weight_std', 0.0):.5f}"),
        ]
        for i, (label, value) in enumerate(rows):
            yy = top + 50 + i * 31
            self.text(c, x3 + 14, yy, label, MUTED, 8)
            self.text(c, w - m - 14, yy, value, TEXT, 9, anchor="e")

        context = s.get("reward_context", {})
        self.text(c, x3 + 14, lower - 105,
                  f"danger {float(context.get('danger', 0)):.2f} · Δ {float(context.get('danger_delta', 0)):+.3f}",
                  MUTED, 8)
        self.text(c, x3 + 14, lower - 83,
                  f"motion {'YES' if context.get('moving') else 'NO'} · safe {float(context.get('safe_seconds', 0)):.1f}s",
                  MUTED, 8)
        self.text(c, x3 + 14, lower - 61,
                  f"event {context.get('event') or '—'} · events {s.get('event_count', 0):,}",
                  MUTED, 8, width=third - 28)
        self.text(c, x3 + 14, lower - 31,
                  "온라인으로 연결 가중치를 갱신합니다. 업데이트와 주행 변화를 함께 확인하세요.",
                  TEXT, 8, width=third - 28)


if __name__ == "__main__":
    root = tk.Tk()
    Viewer(root)
    root.mainloop()
