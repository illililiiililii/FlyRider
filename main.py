# -*- coding: utf-8 -*-
"""FlyRider control center."""
from __future__ import annotations
import json, os, queue, subprocess, sys, threading, tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox, filedialog

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
ENGINE = BASE / "FlyRider_connectome.py"
RESCUE_MODE = DATA / "flyrider_rescue_mode.json"
PYTHON = Path(sys.executable)
if PYTHON.name.lower() == "pythonw.exe": PYTHON = PYTHON.with_name("python.exe")
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
THEMES = {
    "dark": {"bg":"#11161f","panel":"#1a222e","raised":"#222e3c","line":"#303c4d","text":"#f1f4f8","muted":"#9aa8b9","accent":"#79adff","blue":"#3479da","green":"#53d39b","warning":"#f3c969","danger":"#ef6a6a","input":"#141b25"},
    "light":{"bg":"#eef2f7","panel":"#ffffff","raised":"#f5f7fb","line":"#d9e1eb","text":"#192535","muted":"#65768b","accent":"#286bd5","blue":"#286bd5","green":"#168958","warning":"#ad7510","danger":"#c53f4b","input":"#ffffff"}}

class App:
    PAGES = {"drive":"주행","brain":"뇌 상태","vision":"Object AI","overnight":"오버나이트","logs":"기록"}
    def __init__(self, root):
        self.root=root; root.title("FlyRider"); root.geometry("1220x780"); root.minsize(960,640)
        root.update_idletasks(); root.geometry("+%d+%d"%((root.winfo_screenwidth()-1220)//2,(root.winfo_screenheight()-780)//2))
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.theme=self.load_theme(); self.page="drive"; self.process=None; self.mode="idle"
        self.out=queue.Queue(); self.debug=tk.BooleanVar(value=False); self.autostart=tk.BooleanVar(value=True); self.hours=tk.StringVar(value="8")
        self.rescue_mode=tk.StringVar(value=self.load_rescue_mode()); self.embedded={}
        self.status=tk.StringVar(value="엔진 대기")
        self.build(); self.apply_theme(); self.show("drive")
        root.bind("<F8>",lambda e:self.request("flyrider_drive_toggle.request","toggle"))
        root.bind("<F9>",lambda e:self.request("flyrider_immediate_stop.request","stop"))
        root.bind("<F7>",lambda e:self.flip_debug())
        root.bind("<F12>",lambda e:self.request("flyrider_capture.request","capture"))
        root.after(250,self.poll)
    def load_theme(self):
        try:return json.loads((DATA/"flyrider_ui.json").read_text(encoding="utf-8")).get("theme","dark")
        except Exception:return "dark"
    def load_rescue_mode(self):
        try:return json.loads(RESCUE_MODE.read_text(encoding="utf-8")).get("mode","keyboard")
        except Exception:return "keyboard"
    def build(self):
        self.root.columnconfigure(1,weight=1); self.root.rowconfigure(0,weight=1)
        self.side=tk.Frame(self.root,width=220,padx=16,pady=24); self.side.grid(row=0,column=0,sticky="ns"); self.side.grid_propagate(False)
        self.brand=tk.Label(self.side,text="✦  FLYRIDER",font=("Segoe UI",17,"bold"),anchor="w"); self.brand.pack(fill="x")
        self.sub=tk.Label(self.side,text="CONNECTOME CONTROL",font=("Segoe UI",8,"bold"),anchor="w"); self.sub.pack(fill="x",pady=(4,30))
        self.nav={}
        for k,t in self.PAGES.items():
            b=tk.Button(self.side,text="   "+t,anchor="w",relief="flat",bd=0,padx=12,pady=12,font=("Segoe UI",10),command=lambda key=k:self.show(key))
            b.pack(fill="x",pady=3); self.nav[k]=b
        self.shortcuts=tk.Label(self.side,text="F8  주행 전환\nF9  입력 해제\nF7  디버그 창\nF12 장면 수집",justify="left",anchor="w",font=("Segoe UI",9))
        self.shortcuts.pack(side="bottom",fill="x",pady=(8,48))
        self.theme_btn=tk.Button(self.side,text="◐  다크 / 라이트",command=self.toggle_theme,relief="flat",bd=0,pady=10); self.theme_btn.pack(side="bottom",fill="x")
        self.main=tk.Frame(self.root,padx=32,pady=24); self.main.grid(row=0,column=1,sticky="nsew"); self.main.columnconfigure(0,weight=1); self.main.rowconfigure(1,weight=1)
        self.head=tk.Frame(self.main); self.head.grid(row=0,column=0,sticky="ew",pady=(0,18)); self.head.columnconfigure(0,weight=1)
        self.eyebrow=tk.Label(self.head,text="LIVE CONTROL",font=("Segoe UI",8,"bold"),anchor="w"); self.eyebrow.grid(row=0,column=0,sticky="w")
        self.title=tk.Label(self.head,text="",font=("Segoe UI",24,"bold"),anchor="w"); self.title.grid(row=1,column=0,sticky="w",pady=(4,0))
        self.stat=tk.Label(self.head,textvariable=self.status,font=("Segoe UI",9)); self.stat.grid(row=0,column=1,rowspan=2,sticky="e")
        self.body=tk.Frame(self.main); self.body.grid(row=1,column=0,sticky="nsew"); self.body.columnconfigure(0,weight=1); self.body.rowconfigure(0,weight=1)
        self.pages={}
        self.make_drive(); self.make_brain(); self.make_vision(); self.make_overnight(); self.make_logs()
    def card(self,parent,title,hint):
        f=tk.Frame(parent,padx=20,pady=18,highlightthickness=1)
        tk.Label(f,text=title,font=("Segoe UI",12,"bold"),anchor="w").pack(fill="x")
        tk.Label(f,text=hint,font=("Segoe UI",9),anchor="w",justify="left",wraplength=700).pack(fill="x",pady=(5,0))
        return f
    def make_drive(self):
        p=tk.Frame(self.body); self.pages["drive"]=p; p.columnconfigure((0,1),weight=1); p.rowconfigure(1,weight=1)
        hero=tk.Frame(p,padx=24,pady=22,highlightthickness=1); hero.grid(row=0,column=0,columnspan=2,sticky="ew",pady=(0,14))
        tk.Label(hero,text="KARTRIDER  /  CONNECTOME",font=("Segoe UI",9,"bold"),anchor="w").pack(fill="x")
        tk.Label(hero,text="초파리 커넥톰 주행",font=("Segoe UI",25,"bold"),anchor="w").pack(fill="x",pady=(8,4))
        tk.Label(hero,text="엔진이 화면을 읽고 선택한 설정에 따라 vJoy 입력을 보냅니다.",font=("Segoe UI",10),anchor="w").pack(fill="x",pady=(0,16))
        row=tk.Frame(hero); row.pack(fill="x")
        self.go=tk.Button(row,text="주행 시작",command=self.start_drive,padx=22,pady=12,relief="flat",bd=0,font=("Segoe UI",10,"bold")); self.go.pack(side="left")
        self.drive_toggle=tk.Button(row,text="주행 전환  ·  F8",command=lambda:self.request("flyrider_drive_toggle.request","toggle"),padx=16,pady=11,relief="flat",bd=0); self.drive_toggle.pack(side="left",padx=9)
        tk.Button(row,text="저장 후 안전 종료",command=self.stop,padx=16,pady=11,relief="flat",bd=0).pack(side="left")
        self.badge=tk.Label(row,text="● 엔진 대기",font=("Segoe UI",9,"bold"),padx=12); self.badge.pack(side="right")
        options=self.card(p,"주행 설정","디버그 창을 끄면 화면만 표시하지 않습니다. Object AI 추론은 주행 감각으로 계속 사용됩니다.")
        options.grid(row=1,column=0,sticky="nsew",padx=(0,7))
        ttk.Checkbutton(options,text="엔진과 함께 자동 주행 시작",variable=self.autostart).pack(anchor="w",pady=(18,8))
        ttk.Checkbutton(options,text="Object AI 디버그 창 표시  ·  F7",variable=self.debug,command=self.debug_changed).pack(anchor="w",pady=8)
        tk.Button(options,text="장면 수집  ·  F12",command=lambda:self.request("flyrider_capture.request","capture"),relief="flat",padx=12,pady=8).pack(anchor="w",pady=(14,0))
        inputs=self.card(p,"주행 중 입력","디버그 창을 열지 않아도 게임 조작에는 영향을 주지 않습니다.")
        inputs.grid(row=1,column=1,sticky="nsew",padx=(7,0))
        self.input_info=tk.Label(inputs,text="엔진 대기 중\n\nF8  주행 전환\nF9  입력 즉시 해제",justify="left",anchor="nw",font=("Segoe UI",11)); self.input_info.pack(fill="both",expand=True,pady=(20,0))
    def make_brain(self):
        p=tk.Frame(self.body); self.pages["brain"]=p; p.rowconfigure(0,weight=1); p.columnconfigure(0,weight=1)
        self.brain_host=tk.Frame(p); self.brain_host.grid(row=0,column=0,sticky="nsew")
    def make_vision(self):
        p=tk.Frame(self.body); self.pages["vision"]=p; p.rowconfigure(0,weight=1); p.columnconfigure(0,weight=1)
        self.vision_host=tk.Frame(p); self.vision_host.grid(row=0,column=0,sticky="nsew")
        bar=tk.Frame(self.vision_host); bar.pack(fill="x",pady=(0,8))
        tk.Button(bar,text="best.pt 등록",command=self.choose_model,relief="flat",padx=12,pady=7).pack(side="left")
        self.model_label=tk.Label(bar,text="",anchor="w",font=("Segoe UI",9)); self.model_label.pack(side="left",padx=12)
    def make_overnight(self):
        p=tk.Frame(self.body); self.pages["overnight"]=p
        f=self.card(p,"예약 시간 동안 자동 주행","공유 엔진으로 오버나이트 주행을 시작합니다. 복구와 저장 후 종료도 이 화면에서 관리합니다.")
        f.pack(fill="x")
        row=tk.Frame(f); row.pack(anchor="w",pady=(20,12))
        tk.Label(row,text="시간",font=("Segoe UI",10)).pack(side="left")
        tk.Entry(row,textvariable=self.hours,width=8,relief="flat",font=("Segoe UI",11)).pack(side="left",padx=8)
        self.night=tk.Button(f,text="오버나이트 시작",command=self.start_night,relief="flat",padx=16,pady=10); self.night.pack(anchor="w")
        rescue=tk.Frame(f); rescue.pack(anchor="w",pady=(14,0))
        tk.Label(rescue,text="끼임 복구 입력:",font=("Segoe UI",9)).pack(side="left",padx=(0,8))
        ttk.Radiobutton(rescue,text="Windows R 키",variable=self.rescue_mode,value="keyboard").pack(side="left",padx=4)
        ttk.Radiobutton(rescue,text="vJoy 버튼 8",variable=self.rescue_mode,value="vjoy_button8").pack(side="left",padx=4)
        tk.Label(f,text="디버그 창 옵션은 주행 설정과 동일합니다.",font=("Segoe UI",9),anchor="w").pack(fill="x",pady=(14,0))
    def make_logs(self):
        p=tk.Frame(self.body); self.pages["logs"]=p; p.rowconfigure(0,weight=1); p.columnconfigure(0,weight=1)
        wrap=tk.Frame(p,padx=10,pady=10,highlightthickness=1); wrap.grid(sticky="nsew")
        self.log=tk.Text(wrap,state="disabled",wrap="word",relief="flat",bd=0,font=("Cascadia Mono",9))
        sb=ttk.Scrollbar(wrap,command=self.log.yview); self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left",fill="both",expand=True); sb.pack(side="right",fill="y")
    def show(self,key):
        self.page=key; self.title.configure(text=self.PAGES[key]); self.eyebrow.configure(text={"drive":"LIVE CONTROL","brain":"BRAIN ACTIVITY","vision":"VISION WORKSPACE","overnight":"OVERNIGHT RUN","logs":"ACTIVITY LOG"}[key])
        for frame in self.pages.values(): frame.grid_forget()
        self.pages[key].grid(row=0,column=0,sticky="nsew"); self.apply_theme()
        if key not in self.embedded and key in ("brain","vision"):
            try:
                if key == "brain":
                    from brain_viewer import Viewer
                    self.embedded[key]=Viewer(self.brain_host,embedded=True)
                else:
                    from ai_traing.state_object_ai import App as ObjectAIApp
                    self.embedded[key]=ObjectAIApp(self.vision_host,embedded=True)
            except Exception as exc:
                host=self.brain_host if key=="brain" else self.vision_host
                tk.Label(host,text=f"이 화면을 불러오지 못했습니다: {exc}",anchor="w",justify="left",wraplength=700).pack(fill="x",padx=20,pady=20)
            self.apply_theme()
            if key == "vision": self.update_model()
    def apply_theme(self):
        c=THEMES[self.theme]; self.root.configure(bg=c["bg"])
        self.style=ttk.Style(self.root); self.style.theme_use("clam")
        self.style.configure("TFrame",background=c["bg"])
        self.style.configure("TLabel",background=c["panel"],foreground=c["text"])
        self.style.configure("TLabelframe",background=c["panel"],foreground=c["text"],bordercolor=c["line"])
        self.style.configure("TLabelframe.Label",background=c["panel"],foreground=c["accent"])
        self.style.configure("TButton",background=c["raised"],foreground=c["text"],padding=(9,6))
        self.style.map("TButton",background=[("active",c["line"])],foreground=[("active",c["text"])])
        self.style.configure("TEntry",fieldbackground=c["input"],foreground=c["text"])
        self.style.configure("TNotebook",background=c["bg"],borderwidth=0)
        self.style.configure("TNotebook.Tab",background=c["raised"],foreground=c["text"],padding=(12,7))
        self.style.map("TNotebook.Tab",background=[("selected",c["accent"])],foreground=[("selected","#ffffff")])
        self.style.configure("TCheckbutton",background=c["panel"],foreground=c["text"],font=("Segoe UI",10))
        self.style.map("TCheckbutton",background=[("active",c["panel"])],foreground=[("active",c["text"])])
        def paint(w):
            klass=w.winfo_class()
            if klass=="Frame":
                panel=bool(w.cget("highlightthickness")); w.configure(bg=c["panel"] if panel else c["bg"],highlightbackground=c["line"])
            elif klass=="Label":
                w.configure(bg=c["panel"] if isinstance(w.master,tk.Frame) and int(w.master.cget("highlightthickness") or 0)>0 else c["bg"],fg=c["text"])
            elif klass=="Button":
                primary=w in (getattr(self,"go",None),getattr(self,"night",None),getattr(self,"drive_toggle",None))
                w.configure(bg=c["blue"] if primary else c["raised"],fg="#fff" if primary else c["text"],activebackground=c["accent"],activeforeground="#fff",cursor="hand2")
            elif klass=="Entry": w.configure(bg=c["input"],fg=c["text"],insertbackground=c["text"])
            for ch in w.winfo_children(): paint(ch)
        paint(self.root)
        if "brain" in self.embedded:
            self.embedded["brain"].set_theme(c)
        self.brand.configure(fg=c["accent"]); self.sub.configure(fg=c["muted"]); self.shortcuts.configure(fg=c["muted"])
        self.theme_btn.configure(fg=c["text"])
        self.title.configure(fg=c["text"]); self.eyebrow.configure(fg=c["muted"]); self.stat.configure(fg=c["green"] if self.process and self.process.poll() is None else c["muted"])
        for k,b in self.nav.items(): b.configure(bg=c["raised"] if k==self.page else c["bg"],fg=c["accent"] if k==self.page else c["muted"])
        self.log.configure(bg=c["input"],fg=c["text"],insertbackground=c["text"])
        if hasattr(self,"badge"): self.badge.configure(fg=c["green"] if self.process and self.process.poll() is None else c["muted"])
    def toggle_theme(self):
        self.theme="light" if self.theme=="dark" else "dark"; self.apply_theme()
        DATA.mkdir(parents=True,exist_ok=True); (DATA/"flyrider_ui.json").write_text(json.dumps({"theme":self.theme}),encoding="utf-8")
    def debug_changed(self):
        if self.process and self.process.poll() is None: self.request("flyrider_debug_toggle.request","toggle")
    def flip_debug(self): self.debug.set(not self.debug.get()); self.debug_changed()
    def start_drive(self): self.start_engine(autostart=self.autostart.get())
    def start_night(self):
        try: h=max(.1,float(self.hours.get()))
        except ValueError: messagebox.showerror("시간 입력","숫자로 실행 시간을 입력해 주세요."); return
        self.start_engine(overnight=True,hours=h)
    def start_engine(self,overnight=False,autostart=False,hours=None):
        if self.process and self.process.poll() is None: messagebox.showinfo("실행 중","엔진을 먼저 안전 정지해 주세요."); return
        if not ENGINE.exists(): messagebox.showerror("파일 없음",str(ENGINE)); return
        DATA.mkdir(parents=True,exist_ok=True)
        for name in ("flyrider_stop.request","flyrider_debug_toggle.request","flyrider_capture.request","flyrider_drive_toggle.request","flyrider_immediate_stop.request"):
            try:(DATA/name).unlink()
            except FileNotFoundError:pass
        args=[str(PYTHON),str(ENGINE)]
        if overnight: args += ["--overnight",f"--hours={hours:g}"]
        elif autostart: args.append("--autostart")
        if self.debug.get(): args.append("--debug-window")
        if overnight:
            RESCUE_MODE.parent.mkdir(parents=True,exist_ok=True)
            RESCUE_MODE.write_text(json.dumps({"mode":self.rescue_mode.get()},ensure_ascii=False),encoding="utf-8")
        env=os.environ.copy(); env["PYTHONUNBUFFERED"]="1"
        try:self.process=subprocess.Popen(args,cwd=BASE,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",bufsize=1,creationflags=NO_WINDOW)
        except OSError as e:messagebox.showerror("실행 실패",str(e)); return
        self.mode="overnight" if overnight else "drive"; threading.Thread(target=self.read_proc,args=(self.process,),daemon=True).start()
        self.write_log(f"[hub] {self.mode} start · auto={autostart} · debug={self.debug.get()}")
    def read_proc(self,p):
        if p.stdout:
            for line in p.stdout:self.out.put(line.rstrip())
        self.out.put(f"[process exit] code={p.wait()}")
    def stop(self):
        if self.process and self.process.poll() is None:self.request("flyrider_stop.request","stop")
        else:self.status.set("실행 중인 엔진 없음")
    def request(self,name,value):
        if not self.process or self.process.poll() is not None:self.status.set("엔진을 먼저 시작해 주세요"); return
        DATA.mkdir(parents=True,exist_ok=True); (DATA/name).write_text(value+"\n",encoding="utf-8")
    def choose_model(self):
        path=filedialog.askopenfilename(title="best.pt 선택",filetypes=[("PyTorch checkpoint","*.pt")])
        if not path:return
        import shutil
        DATA.mkdir(parents=True,exist_ok=True)
        try:shutil.copy2(path,DATA/"best.pt"); self.update_model(); self.status.set("data/best.pt에 등록 완료")
        except OSError as e:messagebox.showerror("복사 실패",str(e))
    def update_model(self):
        p=DATA/"best.pt"
        if hasattr(self,"model_label"):self.model_label.configure(text=f"활성 모델: {p} ({p.stat().st_size/1048576:.1f} MB)" if p.exists() else f"활성 모델이 없습니다 · {p}")
    def write_log(self,s):
        if not hasattr(self,"log"):return
        self.log.configure(state="normal"); self.log.insert("end",s+"\n"); self.log.see("end"); self.log.configure(state="disabled")
    def poll(self):
        try:
            while True:self.write_log(self.out.get_nowait())
        except queue.Empty:pass
        active=bool(self.process and self.process.poll() is None)
        self.status.set(("오버나이트 실행 중" if self.mode=="overnight" else "엔진 실행 중 · F8 주행 전환") if active else "엔진 대기")
        self.go.configure(text="엔진 실행 중" if active and self.mode=="drive" else "주행 시작")
        self.night.configure(text="엔진 실행 중" if active and self.mode=="overnight" else "오버나이트 시작")
        self.badge.configure(text="● 엔진 연결" if active else "● 엔진 대기")
        c=THEMES[self.theme]
        self.stat.configure(fg=c["green"] if active else c["muted"])
        self.badge.configure(fg=c["green"] if active else c["muted"])
        self.root.after(400,self.poll)
    def close(self):
        if self.process and self.process.poll() is None:
            if not messagebox.askyesno("종료","엔진에 저장 후 종료를 요청하고 창을 닫을까요?"):return
            self.stop(); self.wait_close()
        else:self.root.destroy()
    def wait_close(self):
        if self.process and self.process.poll() is None:self.root.after(300,self.wait_close)
        else:self.root.destroy()

def main(initial_page="drive"):
    root=tk.Tk(); app=App(root); root.after_idle(lambda: app.show(initial_page)); root.mainloop()
if __name__=="__main__":main()
