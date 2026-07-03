"""Overlay one hardware spike vs one simulation spike (Vout)."""
import os, struct
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from brian2 import *

BASE="experiment_data_compare"

def load_trace(base,binf,ch):
    with open(os.path.join(base,binf),"rb") as f:
        assert f.read(2)==b"AG"; f.read(2)
        _,nw=struct.unpack("2i",f.read(8)); tgt=ch-1
        ta=va=None
        for i in range(nw):
            hb=f.read(140); pts=struct.unpack_from("i",hb,12)[0]
            nb=struct.unpack_from("i",hb,8)[0]
            xi=struct.unpack_from("d",hb,32)[0]; xo=struct.unpack_from("d",hb,40)[0]
            for _ in range(nb):
                _,bt,_,bs=struct.unpack("ihhi",f.read(12)); dat=f.read(bs)
                if i==tgt and bt==1:
                    ta=xo+np.arange(pts)*xi; va=np.frombuffer(dat,dtype=np.float32)
    return ta,va

t,v=load_trace(BASE,"O4scope_4.bin",1)
dt_hw=t[1]-t[0]
dist=max(1,int(0.005/dt_hw))
pk,_=find_peaks(v,height=0.2,distance=dist)

# pick two clean consecutive spikes from a steady ~165Hz region (first tonic plateau)
# choose peaks whose amplitude is near the median (avoid clipped/partial)
amps=v[pk]
med=np.median(amps)
# find a peak in mid-trace with amp close to median and a neighbour spike after it
cand=[p for p in pk if abs(v[p]-med)<0.15]
p0=cand[len(cand)//2]
# window a single spike: from ~0.6ms before to ~2ms after peak
w_pre=int(0.0006/dt_hw); w_post=int(0.0025/dt_hw)
seg_t=(t[p0-w_pre:p0+w_post]-t[p0])*1e3   # ms, aligned to peak
seg_v=v[p0-w_pre:p0+w_post]
print(f"HW spike: peak {v[p0]:.3f} V, samples in window = {len(seg_v)} (dt={dt_hw*1e6:.0f}µs)")

# local firing rate around this spike -> infer current for the sim
loc=pk[(pk>p0-int(0.05/dt_hw))&(pk<p0+int(0.05/dt_hw))]
f_loc=1/np.mean(np.diff(t[loc])) if len(loc)>2 else np.nan
print(f"local rate ~ {f_loc:.0f} Hz")

# ── simulation at matched current (from fit: f=165Hz -> ~67.9µA) ──
d=np.load(f"{BASE}/_fit_curve.npz")
Rm_hi=float(d['Rm_hi']);Rm_lo=float(d['Rm_lo']);Vth_v=float(d['Vth'])
Ra_v=float(d['Ra']);Cm_v=float(d['Cm']);I_hold_v=float(d['I_hold'])
# choose I so that model rate == local hardware rate
Ig=d['Igrid'];fg=d['fgrid']
I_match=float(np.interp(f_loc,fg[:np.argmax(fg)],Ig[:np.argmax(fg)])) if not np.isnan(f_loc) else 67.9e-6
print(f"matched sim current = {I_match*1e6:.1f} µA")

prefs.codegen.target='numpy'; defaultclock.dt=1*us
eqs='''
dVm/dt=(I_in-Vm/(Rm_S+Ra))/Cm:volt
Rm_S=(1-s)*Rm_hi+s*Rm_lo:ohm
I_M=Vm/(Rm_S+Ra):amp
Vout=Vm*Ra/(Rm_S+Ra):volt
I_in:amp
s:1
'''
G=NeuronGroup(1,eqs,threshold='Vm>Vth and s<0.5',reset='s=1',
    events={'reopen':'I_M<I_hold and s>0.5'},method='euler',
    namespace=dict(Cm=Cm_v*farad,Ra=Ra_v*ohm,Rm_hi=Rm_hi*ohm,Rm_lo=Rm_lo*ohm,
                   Vth=Vth_v*volt,I_hold=I_hold_v*amp))
G.run_on_event('reopen','s=0'); G.Vm=0*volt;G.s=0;G.I_in=I_match*amp
stm=StateMonitor(G,'Vout',record=True,dt=1*us); spm=SpikeMonitor(G)
run(200*ms)
tv=np.array(stm.t/ms); vo=np.array(stm.Vout[0]/volt)
# take the 2nd spike (steady state), align peak to 0
st=float(spm.t[1]/ms)
m=(tv>st-0.6)&(tv<st+2.5)
sp_t=tv[m]; sp_v=vo[m]
pk_i=np.argmax(sp_v); sp_t=sp_t-sp_t[pk_i]
print(f"SIM spike: peak {sp_v.max():.3f} V")

# downsample sim to scope rate for fair comparison
ds_t=np.arange(sp_t[0],sp_t[-1],dt_hw*1e3)
ds_v=np.interp(ds_t,sp_t,sp_v)

# ── plot ──
fig,axes=plt.subplots(1,2,figsize=(14,5.5))
for ax,title in zip(axes,['Spike shape overlay','Two consecutive spikes']):
    ax.set_xlabel('t − t_peak (ms)'); ax.set_ylabel('Vout (V)')
    ax.grid(alpha=0.3); ax.set_title(title,fontweight='bold')

ax=axes[0]
ax.plot(sp_t,sp_v,'-',color='C0',lw=2,label='Simulation (dt=1µs)')
ax.plot(ds_t,ds_v,'s--',color='C0',ms=6,mfc='white',alpha=0.7,
        label=f'Simulation @ scope rate (10kHz)')
ax.plot(seg_t,seg_v,'D-',color='C3',ms=7,lw=1.5,label='Hardware (10kHz scope)')
ax.legend(fontsize=9); ax.set_xlim(-0.6,2.5)

# right: two consecutive HW spikes vs two sim spikes
ax=axes[1]
p1=cand[len(cand)//2]; 
# two consecutive hw peaks
nxt=pk[pk>p1][0]
w0=int(0.001/dt_hw); w1=int(0.001/dt_hw)
hw_m=slice(p1-w0, nxt+w1)
hw_tt=(t[hw_m]-t[p1])*1e3; hw_vv=v[hw_m]
ax.plot(hw_tt,hw_vv,'D-',color='C3',ms=5,lw=1.3,label='Hardware (2 spikes)')
# two sim spikes
st0=float(spm.t[1]/ms); st1=float(spm.t[2]/ms)
mm=(tv>st0-1)&(tv<st1+1); s2t=tv[mm]-st0; s2v=vo[mm]
ax.plot(s2t,s2v,'-',color='C0',lw=1.8,label='Simulation (2 spikes)')
ax.legend(fontsize=9)
ax.set_xlim(-1, (st1-st0)+1.5)

fig.suptitle(f'MSN spike-shape comparison  |  I_in≈{I_match*1e6:.1f}µA, '
             f'f≈{f_loc:.0f}Hz  |  Rm_lo={Rm_lo:.0f}Ω→τ_close={Cm_v*(Rm_lo+Ra_v)*1e6:.0f}µs',
             fontweight='bold')
fig.tight_layout()
fig.savefig(f"{BASE}/spike_shape_compare.png",dpi=130,bbox_inches='tight')
print("saved -> spike_shape_compare.png")

# numbers
def fwhm(tt,vv):
    pkv=vv.max(); half=pkv/2; pi=np.argmax(vv)
    l=pi
    while l>0 and vv[l]>half:l-=1
    r=pi
    while r<len(vv)-1 and vv[r]>half:r+=1
    return tt[r]-tt[l]
print(f"\nHW peak   {seg_v.max():.3f} V   FWHM {fwhm(seg_t,seg_v):.3f} ms")
print(f"SIM peak  {sp_v.max():.3f} V   FWHM {fwhm(sp_t,sp_v):.3f} ms")
