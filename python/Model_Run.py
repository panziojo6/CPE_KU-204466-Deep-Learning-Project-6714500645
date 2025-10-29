import os
import math
import random
import numpy as np
import pandas as pd
from osgeo import gdal
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import multiprocessing


matplotlib.use("Agg")
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
torch.set_num_threads(multiprocessing.cpu_count())
torch.set_num_interop_threads(multiprocessing.cpu_count())
torch.manual_seed(42); np.random.seed(42); random.seed(42)

base="../DATA"
lulc_path=os.path.join(base,"47QMU.tif")
savi_dir=os.path.join(base,"MAX_SAVI")
mndwi_dir=os.path.join(base,"MAX_MNDWI")

lulc=gdal.Open(lulc_path).ReadAsArray().astype(np.int32)

def load_series(d,name):
    arrs=[]; months=[]
    files=[f for f in sorted(os.listdir(d)) if f.startswith(name) and f.endswith(".tif")]
    for f in files:
        p=os.path.join(d,f)
        ds=gdal.Open(p)
        if ds is None: 
            continue
        a=ds.ReadAsArray().astype(np.float32)
        a[a<=-9000]=np.nan
        arrs.append(a)
        tag=f.split("_")[1]
        months.append(tag[4:6])
    if len(arrs)==0:
        return None, []
    return np.stack(arrs,axis=-1), months

SAVI, savi_months = load_series(savi_dir,"SAVI")
MNDWI, mndwi_months = load_series(mndwi_dir,"MNDWI")
assert SAVI is not None and MNDWI is not None, "Missing SAVI/MNDWI stacks."
common_months = sorted(list(set(savi_months) & set(mndwi_months)))
assert len(common_months)>0, "No overlapping months."

SAVI = np.stack([SAVI[:,:,savi_months.index(m)] for m in common_months],axis=-1)
MNDWI = np.stack([MNDWI[:,:,mndwi_months.index(m)] for m in common_months],axis=-1)

X=np.concatenate([SAVI, MNDWI],axis=-1)
mask=(lulc>0)
X=X[mask]
y=(lulc[mask]>6000).astype(int)
good=np.all(np.isfinite(X),axis=1)
X=X[good]; y=y[good]

idx_forest=np.where(y==1)[0]
idx_other=np.where(y==0)[0]
n=min(len(idx_forest), len(idx_other))
sel_forest=np.random.choice(idx_forest, n, replace=False)
sel_other=np.random.choice(idx_other, n, replace=False)
sel_idx=np.concatenate([sel_forest, sel_other])
np.random.shuffle(sel_idx)
X=X[sel_idx]; y=y[sel_idx]

T=len(common_months)
X = X.reshape(-1, T, 2)
feat_mean = np.nanmean(X, axis=(0,1))
feat_std  = np.nanstd(X, axis=(0,1)) + 1e-6
X = (X - feat_mean) / feat_std

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42, shuffle=True)

Xtr=torch.tensor(X_train, dtype=torch.float32)
ytr=torch.tensor(y_train, dtype=torch.long)
Xva=torch.tensor(X_val, dtype=torch.float32)
yva=torch.tensor(y_val, dtype=torch.long)

train_ds=TensorDataset(Xtr,ytr)
val_ds=TensorDataset(Xva,yva)

num_workers=os.cpu_count()
train_dl=DataLoader(train_ds,batch_size=512,shuffle=True,num_workers=num_workers,pin_memory=False)
val_dl=DataLoader(val_ds,batch_size=512,shuffle=False,num_workers=num_workers,pin_memory=False)

class LSTMCls(nn.Module):
    def __init__(self, hid=64):
        super().__init__()
        self.lstm=nn.LSTM(2,hid,num_layers=3,dropout=0.25,batch_first=True)
        self.fc=nn.Linear(hid,2)
    def forward(self,x):
        _,(h,_) = self.lstm(x)
        return self.fc(h[-1])

class BiLSTMAtt(nn.Module):
    def __init__(self, hid=64):
        super().__init__()
        self.bilstm=nn.LSTM(2,hid,num_layers=3,dropout=0.25,batch_first=True,bidirectional=True)
        self.att=nn.Linear(hid*2,1)
        self.fc=nn.Linear(hid*2,2)
    def forward(self,x):
        o,_=self.bilstm(x)
        w=torch.softmax(self.att(o).squeeze(-1),dim=1).unsqueeze(-1)
        ctx=(o*w).sum(dim=1)
        return self.fc(ctx)

class PosEnc(nn.Module):
    def __init__(self,d_model,max_len=128):
        super().__init__()
        pe=torch.zeros(max_len,d_model)
        pos=torch.arange(0,max_len).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div)
        pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self,x):
        return x+self.pe[:,:x.size(1)]

class TransEnc(nn.Module):
    def __init__(self,d_model=64,nhead=8,dim_ff=128,nlayer=3):
        super().__init__()
        self.proj=nn.Linear(2,d_model)
        self.pe=PosEnc(d_model, max_len=128)
        enc_layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=dim_ff,batch_first=True)
        self.enc=nn.TransformerEncoder(enc_layer,num_layers=nlayer)
        self.fc=nn.Linear(d_model,2)
    def forward(self,x):
        z=self.proj(x)
        z=self.pe(z)
        z=self.enc(z)
        z=z.mean(dim=1)
        return self.fc(z)

def train_eval(model, tag, epochs=50, lr=1e-3, patience=8):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    crit=nn.CrossEntropyLoss()
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=0.5,patience=3,verbose=False)
    tr_loss=[]; va_loss=[]; best=float("inf"); wait=0
    for e in range(epochs):
        model.train(); tl=0.0
        for xb,yb in train_dl:
            opt.zero_grad()
            out=model(xb)
            loss=crit(out,yb)
            loss.backward()
            opt.step()
            tl+=loss.item()
        tr=tl/len(train_dl); tr_loss.append(tr)
        model.eval(); vl=0.0
        with torch.no_grad():
            for xb,yb in val_dl:
                vl+=crit(model(xb),yb).item()
        va=vl/len(val_dl); va_loss.append(va)
        sch.step(va)
        print(f"[{tag}] Epoch {e+1}/{epochs} train_loss={tr:.4f} val_loss={va:.4f} lr={opt.param_groups[0]['lr']:.2e}")
        if va<best:
            best=va; wait=0
            torch.save(model.state_dict(),f"model_{tag}_best.pt")
        else:
            wait+=1
            if wait>=patience:
                print(f"[{tag}] Early stop at epoch {e+1}")
                break
    plt.figure(figsize=(6,4))
    plt.plot(tr_loss,label="train"); plt.plot(va_loss,label="val")
    plt.legend(); plt.tight_layout(); plt.savefig(f"loss_curve_{tag}.png")
    best_state=torch.load(f"model_{tag}_best.pt",map_location="cpu")
    model.load_state_dict(best_state)
    model.eval(); y_true=[]; y_prob=[]
    with torch.no_grad():
        for xb,yb in val_dl:
            p=model(xb)
            y_true.extend(yb.cpu().numpy())
            y_prob.extend(torch.softmax(p,dim=1)[:,1].cpu().numpy())
    y_true=np.array(y_true); y_prob=np.array(y_prob); y_pred=(y_prob>0.5).astype(int)
    acc=accuracy_score(y_true,y_pred)
    pre=precision_score(y_true,y_pred)
    rec=recall_score(y_true,y_pred)
    f1=f1_score(y_true,y_pred)
    auc=roc_auc_score(y_true,y_prob)
    cm=confusion_matrix(y_true,y_pred)
    plt.figure(figsize=(4,4))
    plt.imshow(cm,cmap="Blues")
    plt.xticks([0,1],["Other","Forest"])
    plt.yticks([0,1],["Other","Forest"])
    for i in range(2):
        for j in range(2):
            plt.text(j,i,cm[i,j],ha="center",va="center",color="black")
    plt.tight_layout(); plt.savefig(f"confusion_matrix_{tag}.png")
    fpr,tpr,_=roc_curve(y_true,y_prob)
    plt.figure(figsize=(4,4))
    plt.plot(fpr,tpr); plt.plot([0,1],[0,1],"--")
    plt.title(str(auc))
    plt.tight_layout(); plt.savefig(f"roc_curve_{tag}.png")
    pd.DataFrame([[acc,pre,rec,f1,auc]],columns=["acc","precision","recall","f1","auc"]).to_csv(f"metrics_{tag}.csv",index=False)
    torch.save(model.state_dict(),f"model_{tag}.pt")

model1=LSTMCls(hid=64)
train_eval(model1,"lstm",epochs=50,lr=1e-3,patience=10)

model2=BiLSTMAtt(hid=64)
train_eval(model2,"bilstm_att",epochs=50,lr=1e-3,patience=10)

model3=TransEnc(d_model=64,nhead=8,dim_ff=128,nlayer=3)
train_eval(model3,"transformer",epochs=60,lr=8e-4,patience=10)
