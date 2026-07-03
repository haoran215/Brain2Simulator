"""Validate the fitted params with a real Brian2 sim and plot vs hardware."""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from brian2 import *

d = np.load('experiment_data_compare/_fit_curve.npz')
Rm_hi=float(d['Rm_hi']); Rm_lo=float(d['Rm_lo']); Vth_v=float(d['Vth'])
Ra_v=float(d['Ra']); Cm_v=float(d['Cm']); I_hold_v=float(d['I_hold'])
I_min_v=float(d['I_min'])
pts_I=d['pts_I']; pts_f=d['pts_f']; I_min_hw=float(d['I_min_hw'])

# ── Brian2 sweep ──────────────────────────────────────────────────
prefs.codegen.target = 'numpy'
defaultclock.dt = 1*us
Isweep = np.linspace(I_min_v*1.02, I_hold_v*0.999, 45)
N = len(Isweep)

eqs='''
dVm/dt = (I_in - Vm/(Rm_S+Ra))/Cm : volt
Rm_S = (1-s)*Rm_hi + s*Rm_lo       : ohm
I_M  = Vm/(Rm_S+Ra)                : amp
I_in : amp
s    : 1
'''
G=NeuronGroup(N,eqs,threshold='Vm>Vth and s<0.5',reset='s=1',
    events={'reopen':'I_M<I_hold and s>0.5'},method='euler',
    namespace=dict(Cm=Cm_v*farad,Ra=Ra_v*ohm,Rm_hi=Rm_hi*ohm,
                   Rm_lo=Rm_lo*ohm,Vth=Vth_v*volt,I_hold=I_hold_v*amp))
G.run_on_event('reopen','s=0')
G.Vm=0*volt; G.s=0; G.I_in=Isweep*amp
spm=SpikeMonitor(G)
Trun=1.0*second
run(Trun)
counts=np.array([np.sum(spm.i==k) for k in range(N)])
f_sim=counts/float(Trun)

# analytical curve
Ig=d['Igrid']; fg=d['fgrid']

# ── plot ──────────────────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(9,6))
ax.plot(Ig*1e6, fg,'-',color='C0',lw=2,label='MSN model (analytical, fitted)')
ax.plot(Isweep*1e6, f_sim,'o',color='C0',ms=5,mfc='white',label='MSN model (Brian2 sim)')
ax.plot(pts_I*1e6, pts_f,'D',color='C3',ms=11,label='Hardware anchor points',zorder=5)
ax.plot([I_min_hw*1e6, I_hold_v*1e6],[0,0],'s',color='C3',ms=9,zorder=5)
ax.annotate(f'I_min={I_min_hw*1e6:.1f}µA',(I_min_hw*1e6,0),textcoords='offset points',
            xytext=(-5,10),ha='right',fontsize=9,color='C3')
ax.annotate(f'I_max=I_hold={I_hold_v*1e6:.1f}µA',(I_hold_v*1e6,0),textcoords='offset points',
            xytext=(5,10),ha='left',fontsize=9,color='C3')
for I,f in zip(pts_I,pts_f):
    ax.annotate(f'{f:.0f}Hz@{I*1e6:.1f}µA',(I*1e6,f),textcoords='offset points',
                xytext=(8,-4),fontsize=9,color='C3')
ax.set_xlabel('Injected current  I_in  (µA)',fontsize=11)
ax.set_ylabel('Firing rate  f  (Hz)',fontsize=11)
ax.set_title('MSN simulation fitted to hardware neuron\n'
             f'Ra={Ra_v/1e3:.1f}kΩ, Cm={Cm_v*1e9:.0f}nF fixed | '
             f'fit: Rm_hi={Rm_hi/1e3:.1f}kΩ, Rm_lo={Rm_lo:.0f}Ω, Vth={Vth_v:.2f}V',
             fontsize=11,fontweight='bold')
ax.grid(alpha=0.3); ax.legend(fontsize=10,loc='upper left')
ax.set_ylim(bottom=-8)
fig.tight_layout()
fig.savefig('experiment_data_compare/msn_fit_vs_hardware.png',dpi=130,bbox_inches='tight')
print("saved -> experiment_data_compare/msn_fit_vs_hardware.png")
print("\nBrian2 vs analytical at anchor currents:")
for I,ft in zip(pts_I,pts_f):
    k=int(np.argmin(np.abs(Isweep-I)))
    print(f"  I~{Isweep[k]*1e6:5.1f}µA: sim {f_sim[k]:6.1f}Hz  analytic {np.interp(Isweep[k],Ig,fg):6.1f}Hz  hw {ft:.0f}Hz")
print(f"\nsim peak rate {f_sim.max():.1f} Hz at {Isweep[np.argmax(f_sim)]*1e6:.1f} µA")
