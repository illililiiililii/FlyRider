# -*- coding: utf-8 -*-
"""FlyRider State AI: label -> train -> realtime inference.
Run: python state_ai.py label | train | run
"""
import sys,json,time
from pathlib import Path
from collections import Counter
import cv2,numpy as np,mss

BASE=Path(__file__).resolve().parent
DATA=BASE/'data'/'state_ai'; IMG=DATA/'images'; MODEL=DATA/'state_ai_model.npz'; LABELS=DATA/'state_ai_labels.json'; CFG=BASE/'data'/'flyrider_config.json'
IMG.mkdir(parents=True,exist_ok=True)
NAMES={'normal':'정상 주행','left_wall':'왼쪽 벽 가까움','right_wall':'오른쪽 벽 가까움','both_wall':'양쪽 벽 가까움','front_block':'앞이 막힘','stuck':'벽에 끼임','escaping':'탈출 중','escaped':'탈출 성공'}
CLASSES=list(NAMES)
W,H=64,36

def region():
    try:
        c=json.loads(CFG.read_text(encoding='utf-8')); r=c.get('region')
        if r and all(k in r for k in ('left','top','width','height')): return {k:int(r[k]) for k in ('left','top','width','height')}
    except Exception: pass
    with mss.mss() as s: m=s.monitors[1]; return {k:int(m[k]) for k in ('left','top','width','height')}

def grab(s,r): return cv2.cvtColor(np.array(s.grab(r)),cv2.COLOR_BGRA2BGR)

def feat(f):
    im=cv2.resize(f,(W,H),interpolation=cv2.INTER_AREA); b=im.astype(np.float32)/255
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV).astype(np.float32); hsv[...,0]/=179; hsv[...,1:]/=255
    g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    gx=cv2.Sobel(g,cv2.CV_32F,1,0,3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,3); e=np.clip(cv2.magnitude(gx,gy)/4,0,1)
    parts=[cv2.resize(x,(16,9),interpolation=cv2.INTER_AREA).reshape(-1) for x in (b,hsv,g,e)]
    return np.concatenate(parts).astype(np.float32)

def rows_load():
    if not LABELS.exists(): return []
    try:return json.loads(LABELS.read_text(encoding='utf-8'))
    except:return []

def rows_save(r): LABELS.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
def stats(r):
    c=Counter(x['label'] for x in r); print('\n데이터:',len(r),'장'); [print(f' {k:12s} {c[k]:4d}장  {NAMES[k]}') for k in CLASSES]

def label_mode():
    r=rows_load(); rid=max([int(x['image'].split('/')[-1].split('_')[0]) for x in r if x.get('image','').split('/')[-1].split('_')[0].isdigit()] or [0])+1; rg=region()
    print('\nSPACE=캡처, D=마지막 삭제, Q/ESC=종료')
    with mss.mss() as s:
      while 1:
        f=grab(s,rg); v=f.copy(); cv2.putText(v,'SPACE capture / D delete / Q quit',(10,30),0,.7,(0,255,0),2); cv2.imshow('FlyRider State AI - Label',v); k=cv2.waitKey(30)&255
        if k in (ord('q'),ord('Q'),27): break
        if k in (ord('d'),ord('D')):
            if r:
                x=r.pop(); p=BASE/x['image']; p.exists() and p.unlink(); rows_save(r); print('[삭제] 마지막 데이터'); stats(r)
            continue
        if k!=32: continue
        print('\n상태 선택:'); [print(f'{i+1}. {x} = {NAMES[x]}') for i,x in enumerate(CLASSES)]
        try:n=int(input('번호 > '))-1
        except:continue
        if not 0<=n<len(CLASSES):continue
        key=CLASSES[n]; p=IMG/f'{rid:06d}_{key}.jpg'; cv2.imwrite(str(p),f,[cv2.IMWRITE_JPEG_QUALITY,95]); r.append({'image':p.relative_to(BASE).as_posix(),'label':key,'time':time.strftime('%Y-%m-%d %H:%M:%S')}); rows_save(r); print('[저장]',p.name,'->',NAMES[key]); rid+=1; stats(r)
    cv2.destroyAllWindows()

def dataset():
    X=[]; y=[]
    for x in rows_load():
      p=BASE/x.get('image',''); f=cv2.imread(str(p)) if p.exists() else None
      if f is not None and x.get('label') in CLASSES:X.append(feat(f)); y.append(CLASSES.index(x['label']))
    return np.asarray(X,np.float32),np.asarray(y,np.int64)

class Model:
    def __init__(self): self.mean=self.std=self.w=self.b=self.cls=None
    def fit(self,X,y,epochs=220):
      self.mean=X.mean(0); self.std=X.std(0); self.std[self.std<1e-6]=1; X=(X-self.mean)/self.std; self.cls=np.unique(y); self.w=np.zeros((len(self.cls),X.shape[1]),np.float32); self.b=np.zeros(len(self.cls),np.float32); mp={int(c):i for i,c in enumerate(self.cls)}; yy=np.array([mp[int(a)] for a in y]); rng=np.random.default_rng(7)
      for _ in range(epochs):
       idx=rng.permutation(len(X))
       for st in range(0,len(X),32):
        z=X[idx[st:st+32]]; t=yy[idx[st:st+32]]; q=z@self.w.T+self.b; q-=q.max(1,keepdims=True); p=np.exp(q); p/=p.sum(1,keepdims=True); p[np.arange(len(t)),t]-=1; self.w-=.025*((p.T@z)/len(t)+.0005*self.w); self.b-=.025*p.mean(0)
    def proba(self,x):
      x=(x-self.mean)/self.std; q=x@self.w.T+self.b; q-=q.max(1,keepdims=True); p=np.exp(q); return p/p.sum(1,keepdims=True)
    def save(self): np.savez_compressed(MODEL,mean=self.mean,std=self.std,w=self.w,b=self.b,cls=self.cls)
    def load(self):
      if not MODEL.exists():return False
      z=np.load(MODEL); self.mean=z['mean'];self.std=z['std'];self.w=z['w'];self.b=z['b'];self.cls=z['cls'];return True

def train():
    X,y=dataset(); print('\n총',len(X),'장');
    if len(X)<2 or len(np.unique(y))<2: print('서로 다른 상태 2개 이상, 각 상태 여러 장을 먼저 라벨링하세요.'); return
    c=Counter(y); [print(f' {CLASSES[i]:12s}: {c[i]}장') for i in range(len(CLASSES))]
    m=Model(); print('학습 중...'); m.fit(X,y); m.save(); pred=m.cls[np.argmax(m.proba(X),1)]; print(f'학습 데이터 정확도: {np.mean(pred==y)*100:.2f}%'); print('모델:',MODEL)

def run():
    m=Model()
    if not m.load(): print('모델이 없습니다. label -> train 순서로 실행하세요.'); return
    rg=region(); print('Q/ESC=종료, P=일시정지')
    pause=False
    with mss.mss() as s:
      while 1:
        f=grab(s,rg); p=m.proba(feat(f)[None])[0]; pairs=sorted([(CLASSES[int(m.cls[i])],float(p[i])) for i in range(len(p))],key=lambda x:x[1],reverse=True); top,conf=pairs[0]
        view=f.copy(); cv2.rectangle(view,(0,0),(360,35+30*len(pairs)),(20,20,20),-1); cv2.putText(view,f'STATE: {top} {conf*100:.1f}%',(10,28),0,.6,(0,255,0),2)
        for i,(k,v) in enumerate(pairs): cv2.putText(view,f'{k:12s} {v*100:5.1f}%',(10,55+i*25),0,.45,(230,230,230),1)
        cv2.imshow('FlyRider State AI - Run',view); key=cv2.waitKey(1)&255
        if key in (ord('q'),ord('Q'),27):break
        if key in (ord('p'),ord('P')):pause=not pause
        if pause: time.sleep(.05); continue
    cv2.destroyAllWindows()

class StateAI:
    """FlyRider_connectome.py에서 사용: analyze(frame)만 호출하면 상태를 반환합니다."""
    def __init__(self):
        self.m=Model()
        if not self.m.load(): raise FileNotFoundError('data/state_ai/state_ai_model.npz가 없습니다.')
    def analyze(self,frame):
        p=self.m.proba(feat(frame)[None])[0]; raw={CLASSES[int(self.m.cls[i])]:float(p[i]) for i in range(len(p))}; state=max(raw,key=raw.get)
        return {'state':state,'confidence':raw[state],'normal':raw.get('normal',0),'left_wall':raw.get('left_wall',0)+.5*raw.get('both_wall',0),'right_wall':raw.get('right_wall',0)+.5*raw.get('both_wall',0),'front_block':raw.get('front_block',0),'stuck':raw.get('stuck',0),'escaping':raw.get('escaping',0),'escaped':raw.get('escaped',0),'raw':raw}

def main():
    a=sys.argv[1].lower() if len(sys.argv)>1 else ''
    if a=='label':label_mode()
    elif a=='train':train()
    elif a=='run':run()
    else:
      print('\n1. 라벨링  2. 학습  3. 실시간  0. 종료'); x=input('> ').strip(); {'1':label_mode,'2':train,'3':run}.get(x,lambda:None)()
if __name__=='__main__':main()
