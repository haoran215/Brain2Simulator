"""Fit MSN F-I over the working range only (exclude depol-block corner)."""
import numpy as np, pandas as pd
from scipy.optimize import least_squares
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

df = pd.read_excel('experiment_data_compare/IFcurve.xlsx')
I_hw = df['Input current(uA)'].values*1e-6
f_hw = df['Frequency(hz)'].values

Cm, Ra = 0.1e-6, 2200.0
# working range: firing points below the plateau/cliff (< 93 µA)
fit_mask = (f_hw > 0) & (I_hw < 93e-6)
Ii, ff = I_hw[fit_mask], f_hw[fit_mask]

def frate(I, Rm_hi, Rm_lo, Vth, I_hold):
    tau_o=Cm*(Rm_hi+Ra); tau_c=Cm*(Rm_lo+Ra)
    Vh=I_hold*(Rm_lo+Ra); Vo=I*(Rm_hi+Ra); Vc=I*(Rm_lo+Ra)
    with np.errstate(all='ignore'):
        T=tau_o*np.log((Vo-Vh)/(Vo-Vth))+tau_c*np.log((Vth-Vc)/(Vh-Vc))
        f=np.where(T>0,1/T,0.0)
    return np.where((I>Vth/(Rm_hi+Ra))&(I<I_hold),f,0.0)

r=least_squares(lambda p: frate(Ii,*p)-ff, [52e3,1000.,2.3,103e-6],
                bounds=([5e3,1.,0.5,95e-6],[5e5,5e3,5.,115e-6]),
                x_scale=[1e4,1e3,1,1e-6])
Rm_hi,Rm_lo,Vth,I_hold=r.x
Imin=Vth/(Rm_hi+Ra); tau_c=Cm*(Rm_lo+Ra)
fmod=frate(Ii,*r.x); rmse=np.sqrt(np.mean((fmod-ff)**2))
print("="*56)
print("  WORKING-RANGE FIT (I < 93 µA), Ra & Cm fixed")
print("="*56)
print(f"  Rm_hi  = {Rm_hi/1e3:7.2f} kΩ")
print(f"  Rm_lo  = {Rm_lo:7.1f} Ω   (bound 5000 — not pinned now)")
print(f"  Vth    = {Vth:7.3f} V")
print(f"  I_hold = {I_hold*1e6:7.2f} µA")
print(f"  I_min  = {Imin*1e6:.2f} µA   tau_open={Cm*(Rm_hi+Ra)*1e3:.2f}ms  tau_close={tau_c*1e6:.0f}µs")
print(f"  spike-width check: tau_close={tau_c*1e6:.0f}µs vs measured FWHM~300µs")
print(f"  RMSE over {fit_mask.sum()} fitted points = {rmse:.2f} Hz")
print("-"*56)
for I,fh,fm in zip(Ii*1e6,ff,fmod):
    print(f"   {I:6.1f}µA  hw {fh:6.1f}  model {fm:6.1f}  ({fm-fh:+5.1f})")
np.savez('experiment_data_compare/_fit_working.npz',
         Rm_hi=Rm_hi,Rm_lo=Rm_lo,Vth=Vth,I_hold=I_hold,Imin=Imin,Ra=Ra,Cm=Cm)

# plot
Ig=np.linspace(Imin*1.001,I_hold*0.9999,600); fg=frate(Ig,*r.x)
fig,ax=plt.subplots(figsize=(9.5,6))
ax.plot(Ig*1e6,fg,'-',color='C0',lw=2,label='MSN model (fitted to working range)')
ax.plot(I_hw[fit_mask]*1e6,f_hw[fit_mask],'D',color='C3',ms=8,label='Hardware — fitted',zorder=5)
excl=(f_hw>0)&(I_hw>=93e-6)
ax.plot(I_hw[excl]*1e6,f_hw[excl],'d',color='gray',ms=8,mfc='none',
        label='Hardware — depol-block (excluded)',zorder=5)
ax.plot(I_hw[f_hw==0]*1e6,f_hw[f_hw==0],'x',color='gray',ms=7)
ax.axvspan(93,105,color='gray',alpha=0.08)
ax.text(99,20,'depol-block corner\n(device-specific)',ha='center',fontsize=9,color='gray')
ax.set_xlabel('Input current (µA)'); ax.set_ylabel('Firing rate (Hz)')
ax.set_title('MSN simulation vs hardware — working-range fit\n'
   f'Rm_hi={Rm_hi/1e3:.1f}kΩ Rm_lo={Rm_lo:.0f}Ω Vth={Vth:.2f}V | RMSE={rmse:.1f}Hz over 13 pts',
   fontsize=11,fontweight='bold')
ax.grid(alpha=0.3); ax.legend(fontsize=9,loc='lower right'); ax.set_ylim(bottom=-8)
fig.tight_layout(); fig.savefig('experiment_data_compare/msn_working_range_fit.png',dpi=130,bbox_inches='tight')
print("\nsaved -> msn_working_range_fit.png")
