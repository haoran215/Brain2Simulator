"""
Fit the MSN analytical F-I curve to the measured hardware neuron.

Fixed to hardware:   Ra = 2.2 kΩ,  Cm = 0.1 µF,  I_hold = I_max = 99.9 µA
Free (fitted):       Rm_hi, Rm_lo, Vth
Hardware anchors:    I_min = 43.6 µA (f->0),  f(67.9µA)=165,  f(97.7µA)=211,
                     I_max = 99.9 µA (f->0)
"""
import numpy as np
from scipy.optimize import least_squares

# ── fixed hardware ────────────────────────────────────────────────
Cm     = 0.1e-6      # F
Ra     = 2200.0      # Ω
I_hold = 99.9e-6     # A  (= I_max)

# ── hardware anchor points ────────────────────────────────────────
I_min_hw = 43.6e-6
pts_I = np.array([67.9e-6, 97.7e-6])
pts_f = np.array([165.0,   211.0])

def firing_rate(I, Rm_hi, Rm_lo, Vth):
    """Analytical MSN firing rate f(I) [Hz]. Vectorised over I."""
    tau_o = Cm*(Rm_hi+Ra)
    tau_c = Cm*(Rm_lo+Ra)
    V_hold = I_hold*(Rm_lo+Ra)
    Vinf_o = I*(Rm_hi+Ra)      # open steady state
    Vinf_c = I*(Rm_lo+Ra)      # closed steady state
    # guard
    with np.errstate(invalid='ignore', divide='ignore'):
        T_charge = tau_o*np.log((Vinf_o - V_hold)/(Vinf_o - Vth))
        T_disch  = tau_c*np.log((Vth - Vinf_c)/(V_hold - Vinf_c))
    T = T_charge + T_disch
    f = np.where(T>0, 1.0/T, 0.0)
    # outside operating window -> 0
    I_min = Vth/(Rm_hi+Ra)
    f = np.where((I>I_min)&(I<I_hold), f, 0.0)
    return f

def residuals(p):
    Rm_hi, Rm_lo, Vth = p
    r_rate = firing_rate(pts_I, Rm_hi, Rm_lo, Vth) - pts_f
    I_min_model = Vth/(Rm_hi+Ra)
    r_imin = 5.0*(I_min_model - I_min_hw)/1e-6   # weight rheobase, in µA
    return np.concatenate([r_rate, [r_imin]])

p0 = [40e3, 500.0, 1.8]
lb = [5e3,  1.0,   0.5]
ub = [5e5,  5000., 5.0]
res = least_squares(residuals, p0, bounds=(lb, ub))
Rm_hi, Rm_lo, Vth = res.x

I_min_model = Vth/(Rm_hi+Ra)
tau_o = Cm*(Rm_hi+Ra); tau_c = Cm*(Rm_lo+Ra)
print("="*60)
print("  FITTED MSN PARAMETERS (Ra, Cm, I_hold fixed to hardware)")
print("="*60)
print(f"  Rm_hi  = {Rm_hi/1e3:8.2f} kΩ")
print(f"  Rm_lo  = {Rm_lo:8.1f} Ω")
print(f"  Vth    = {Vth:8.3f} V")
print(f"  --- fixed ---")
print(f"  Ra     = {Ra:8.0f} Ω   Cm = {Cm*1e9:.0f} nF   I_hold = {I_hold*1e6:.1f} µA")
print(f"  --- derived ---")
print(f"  I_min  = {I_min_model*1e6:6.2f} µA  (target 43.6)")
print(f"  tau_open  = {tau_o*1e3:.3f} ms")
print(f"  tau_close = {tau_c*1e6:.1f} µs")
print("-"*60)
print("  fit quality at anchor points:")
for I,ftgt in zip(pts_I, pts_f):
    fm = firing_rate(np.array([I]), Rm_hi, Rm_lo, Vth)[0]
    print(f"    I={I*1e6:5.1f} µA : model {fm:6.1f} Hz | hardware {ftgt:5.0f} Hz | err {fm-ftgt:+5.1f}")
print("="*60)

# save curve
Igrid = np.linspace(I_min_model*1.001, I_hold*0.9999, 400)
fgrid = firing_rate(Igrid, Rm_hi, Rm_lo, Vth)
np.savez('experiment_data_compare/_fit_curve.npz',
         Igrid=Igrid, fgrid=fgrid, Rm_hi=Rm_hi, Rm_lo=Rm_lo, Vth=Vth,
         I_min=I_min_model, I_hold=I_hold, Ra=Ra, Cm=Cm,
         pts_I=pts_I, pts_f=pts_f, I_min_hw=I_min_hw)
print("peak model rate:", round(fgrid.max(),1), "Hz at I =",
      round(Igrid[np.argmax(fgrid)]*1e6,1), "µA")
