"""Fit MSN simulation spike shape to the high-res (40kHz) capture spike_fit.bin."""
import os, struct
import numpy as np
from scipy.optimize import least_squares
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.signal import find_peaks

BASE="experiment_data_compare"
def load(base,binf):
    out={}
    with open(os.path.join(base,binf),"rb") as f:
        assert f.read(2)==b"AG"; f.read(2)
        _,nw=struct.unpack("2i",f.read(8))
        for i in range(nw):
            hb=f.read(140); pts=struct.unpack_from("i",hb,12)[0]
            nb=struct.unpack_from("i",hb,8)[0]
            xi=struct.unpack_from("d",hb,32)[0]; xo=struct.unpack_from("d",hb,40)[0]
            ta=va=None
            for _ in range(nb):
                _,bt,_,bs=struct.unpack("ihhi",f.read(12)); dat=f.read(bs)
                if bt==1: ta=xo+np.arange(pts)*xi; va=np.frombuffer(dat,dtype=np.float32)
            out[i+1]=(ta,va)
    return out
d=load(BASE,"spike_fit.bin")

# fixed hardware + F-I-fit params
Cm,Ra=0.1e-6,2200.0
Rm_hi=56400.0; I_hold=96e-6

def avg_spike(t,v,dt):
    pk,_=find_peaks(v,height=0.5,distance=int(0.002/dt))
    pre=int(0.10e-3/dt); post=int(1.4e-3/dt)
    segs=[v[p-pre:p+post] for p in pk if p-pre>=0 and p+post<len(v)]
    segs=[s for s in segs if len(s)==pre+post]
    A=np.array(segs); m=A.mean(0)
    tt=(np.arange(-pre,post))*dt
    return tt, m, len(segs), pk

def model_vout(tt, Vth, Rm_lo, I_in):
    """Analytic Vout: instant rise to peak at t=0, then closed-phase RC decay,
    then (after reopen) open-phase charging from V_hold."""
    tc=Cm*(Rm_lo+Ra); to=Cm*(Rm_hi+Ra)
    Vc=I_in*(Rm_lo+Ra); Vo=I_in*(Rm_hi+Ra)
    Vh=I_hold*(Rm_lo+Ra)                         # Vm at reopen
    # closed-phase Vm decay from Vth toward Vc; reopen when Vm=Vh
    t_sp=tc*np.log((Vth-Vc)/(Vh-Vc))
    out=np.zeros_like(tt)
    for i,t in enumerate(tt):
        if t<0:
            out[i]=Vh*Ra/(Rm_hi+Ra)              # pre-spike baseline (charging, ~Vh open)
        elif t<=t_sp:
            Vm=Vc+(Vth-Vc)*np.exp(-t/tc)
            out[i]=Vm*Ra/(Rm_lo+Ra)
        else:
            Vm=Vo+(Vh-Vo)*np.exp(-(t-t_sp)/to)   # open-phase recharge
            out[i]=Vm*Ra/(Rm_hi+Ra)
    return out

results={}
fig,axes=plt.subplots(1,2,figsize=(14,5.5))
for ax,ch in zip(axes,[1,2]):
    t,v=d[ch]; dt=t[1]-t[0]
    tt,mspk,nseg,pk=avg_spike(t,v,dt)
    rate=1/np.mean(np.diff(t[pk]))
    # infer I_in from rate via working-range fit curve
    I_in0=np.interp(rate,[49,117,165,195,212],[43.6e-6,52.5e-6,67.9e-6,80.1e-6,91.5e-6])
    # fit Vth, Rm_lo (I_in fixed from rate)
    def resid(p): return model_vout(tt,p[0],p[1],I_in0)-mspk
    r=least_squares(resid,[2.3,300.],bounds=([1.5,10.],[3.5,3000.]))
    Vth,Rm_lo=r.x
    fit=model_vout(tt,Vth,Rm_lo,I_in0)
    tc=Cm*(Rm_lo+Ra)
    # measured vs model peak/FWHM
    def fwhm(tt,vv):
        pkv=vv.max();h=pkv/2;pi=np.argmax(vv);l=pi
        while l>0 and vv[l]>h:l-=1
        rr=pi
        while rr<len(vv)-1 and vv[rr]>h:rr+=1
        return tt[rr]-tt[l]
    results[ch]=dict(Vth=Vth,Rm_lo=Rm_lo,tc=tc,I_in=I_in0,rate=rate,
                     hw_peak=mspk.max(),sim_peak=fit.max(),
                     hw_fwhm=fwhm(tt,mspk),sim_fwhm=fwhm(tt,fit),nseg=nseg)
    ax.plot(tt*1e3,mspk,'D-',color='C3',ms=4,lw=1.2,label=f'Hardware CH{ch} (avg of {nseg})')
    ax.plot(tt*1e3,fit,'-',color='C0',lw=2,label='Simulation (fitted)')
    ax.set_xlabel('t − t_peak (ms)'); ax.set_ylabel('Vout (V)'); ax.grid(alpha=0.3)
    ax.set_title(f'CH{ch}: {rate:.0f}Hz  |  Vth={Vth:.2f}V  Rm_lo={Rm_lo:.0f}Ω  '
                 f'τ_close={tc*1e6:.0f}µs',fontweight='bold',fontsize=10)
    ax.legend(fontsize=9)
fig.suptitle('MSN spike-shape fit to high-resolution capture (40 kHz)',fontweight='bold')
fig.tight_layout(); fig.savefig(f"{BASE}/spike_hires_fit.png",dpi=130,bbox_inches='tight')

print("="*60)
print("  HIGH-RES SPIKE FIT (Ra=2.2kΩ, Cm=0.1µF, Rm_hi=56.4kΩ fixed)")
print("="*60)
for ch,r in results.items():
    print(f"CH{ch} ({r['rate']:.0f}Hz, I≈{r['I_in']*1e6:.0f}µA, avg {r['nseg']} spikes):")
    print(f"   Vth   = {r['Vth']:.3f} V     Rm_lo = {r['Rm_lo']:.0f} Ω  (τ_close={r['tc']*1e6:.0f}µs)")
    print(f"   peak  hw {r['hw_peak']:.3f} V  vs sim {r['sim_peak']:.3f} V  ({(r['sim_peak']/r['hw_peak']-1)*100:+.1f}%)")
    print(f"   FWHM  hw {r['hw_fwhm']*1e3:.3f} ms vs sim {r['sim_fwhm']*1e3:.3f} ms")
print("\nsaved -> spike_hires_fit.png")
