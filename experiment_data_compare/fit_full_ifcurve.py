"""Fit MSN F-I to the FULL measured curve (IFcurve.xlsx). Ra,Cm fixed."""
import numpy as np, pandas as pd
from scipy.optimize import least_squares
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

df = pd.read_excel('experiment_data_compare/IFcurve.xlsx')
I_hw = df['Input current(uA)'].values*1e-6
f_hw = df['Frequency(hz)'].values
mask = f_hw > 0                     # fit only firing points
Ii, ff = I_hw[mask], f_hw[mask]

Cm, Ra = 0.1e-6, 2200.0            # fixed to hardware

def frate(I, Rm_hi, Rm_lo, Vth, I_hold):
    tau_o=Cm*(Rm_hi+Ra); tau_c=Cm*(Rm_lo+Ra)
    Vh=I_hold*(Rm_lo+Ra); Vo=I*(Rm_hi+Ra); Vc=I*(Rm_lo+Ra)
    with np.errstate(all='ignore'):
        Tc=tau_o*np.log((Vo-Vh)/(Vo-Vth)); Td=tau_c*np.log((Vth-Vc)/(Vh-Vc))
        T=Tc+Td; f=np.where(T>0,1/T,0.0)
    Imin=Vth/(Rm_hi+Ra)
    return np.where((I>Imin)&(I<I_hold),f,0.0)

def resid(p):
    return frate(Ii,*p) - ff

# free: Rm_hi, Rm_lo, Vth, I_hold
p0=[52e3, 1000., 2.3, 102e-6]
lb=[5e3, 1., 0.5, 100e-6]; ub=[5e5, 5e3, 5., 110e-6]
r=least_squares(resid,p0,bounds=(lb,ub),x_scale=[1e4,1e3,1,1e-6])
Rm_hi,Rm_lo,Vth,I_hold=r.x
Imin=Vth/(Rm_hi+Ra)
fmod=frate(Ii,*r.x)
rmse=np.sqrt(np.mean((fmod-ff)**2))
print("="*58)
print("  FULL-CURVE FIT (Ra=2.2kΩ, Cm=0.1µF fixed)")
print("="*58)
print(f"  Rm_hi  = {Rm_hi/1e3:7.2f} kΩ")
print(f"  Rm_lo  = {Rm_lo:7.1f} Ω")
print(f"  Vth    = {Vth:7.3f} V")
print(f"  I_hold = {I_hold*1e6:7.2f} µA   (was assumed 99.9)")
print(f"  -> I_min = {Imin*1e6:.2f} µA")
print(f"  -> tau_open {Cm*(Rm_hi+Ra)*1e3:.2f} ms, tau_close {Cm*(Rm_lo+Ra)*1e6:.0f} µs")
print(f"  RMSE over {mask.sum()} firing points = {rmse:.1f} Hz")
print("-"*58)
print("   I(µA)  hw(Hz)  model(Hz)   err")
for I,fh,fm in zip(Ii*1e6,ff,fmod):
    print(f"  {I:6.1f} {fh:7.1f} {fm:9.1f} {fm-fh:+7.1f}")
np.savez('experiment_data_compare/_fit_full.npz',
         Rm_hi=Rm_hi,Rm_lo=Rm_lo,Vth=Vth,I_hold=I_hold,Imin=Imin,Ra=Ra,Cm=Cm)

# ── plot full curve ──
Ig=np.linspace(Imin*1.001,I_hold*0.9999,600); fg=frate(Ig,*r.x)
fig,ax=plt.subplots(figsize=(9.5,6))
ax.plot(Ig*1e6,fg,'-',color='C0',lw=2,label='MSN model (fitted, analytical)')
ax.plot(I_hw*1e6,f_hw,'D',color='C3',ms=8,label='Hardware (IFcurve.xlsx)',zorder=5)
ax.axvline(Imin*1e6,ls=':',color='gray',lw=1); ax.axvline(I_hold*1e6,ls=':',color='gray',lw=1)
ax.text(Imin*1e6,ax.get_ylim()[1]*0.02,f' I_min={Imin*1e6:.1f}µA',fontsize=9,color='gray')
ax.text(I_hold*1e6,ax.get_ylim()[1]*0.02,f'I_hold={I_hold*1e6:.1f}µA ',fontsize=9,color='gray',ha='right')
ax.set_xlabel('Input current (µA)'); ax.set_ylabel('Firing rate (Hz)')
ax.set_title('MSN simulation vs hardware — full F–I curve\n'
   f'fit: Rm_hi={Rm_hi/1e3:.1f}kΩ Rm_lo={Rm_lo:.0f}Ω Vth={Vth:.2f}V I_hold={I_hold*1e6:.1f}µA | RMSE={rmse:.1f}Hz',
   fontsize=11,fontweight='bold')
ax.grid(alpha=0.3); ax.legend(fontsize=10); ax.set_ylim(bottom=-8)
fig.tight_layout(); fig.savefig('experiment_data_compare/msn_full_ifcurve_fit.png',dpi=130,bbox_inches='tight')
print("\nsaved -> msn_full_ifcurve_fit.png")
