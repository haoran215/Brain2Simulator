"""
neuron.py  —  Memristor aLIF Neuron Module
==========================================
Provides NeuronParams (solves Cm and t_ref from I-F targets) and
build_neurons() which returns a Brian2 NeuronGroup ready to use.

Equations implemented (paper Eqs. 9-12):
  Eq.(9)  Cm dVm/dt = -Vm/(Rm_S+Ra) + I_syn + I_0
  Eq.(10) Vpost     = Vm * Ra / (Rm_S + Ra)
  Eq.(11) tau_s1 dIs1/dt = -Is1    [+Iw*delta via Synapse.on_pre]
  Eq.(12) tau_s2 dIs2/dt = -Is2 + Is1
"""

import json
import numpy as np
from brian2 import *


class NeuronParams:
    """
    Holds all neuron parameters. Cm and t_ref are SOLVED analytically
    from the I-F targets (I_min, f_min) and (I_max, f_max) so that:
        f(I_min) = f_min Hz
        f(I_max) = f_max Hz  (depolarisation-block onset)

    Usage
    -----
    p = NeuronParams.from_json('config.json')
    p = NeuronParams(Ra_ohm=2200, Rm_hi_ohm=100000, ...)
    p.summary()
    """

    def __init__(self,
                 Ra_ohm      : float = 2200.0,
                 Rm_hi_ohm   : float = 100000.0,
                 Rm_lo_ohm   : float = 100.0,
                 Vthresh_V   : float = 4.0,
                 Vreset_V    : float = 0.0,
                 I_0_uA      : float = 15.0,
                 I_min_uA    : float = 40.0,
                 I_max_uA    : float = 100.0,
                 f_min_Hz    : float = 70.0,
                 f_max_Hz    : float = 200.0):

        # Circuit params
        self.Ra_ohm    = Ra_ohm
        self.Rm_hi_ohm = Rm_hi_ohm
        self.Rm_lo_ohm = Rm_lo_ohm
        self.Vthresh_V = Vthresh_V
        self.Vreset_V  = Vreset_V
        self.I_0_uA    = I_0_uA

        # I-F targets
        self.I_min_uA  = I_min_uA
        self.I_max_uA  = I_max_uA
        self.f_min_Hz  = f_min_Hz
        self.f_max_Hz  = f_max_Hz

        # Solved
        self.R_tot_ohm = Ra_ohm + Rm_hi_ohm
        self._solve()

    # ------------------------------------------------------------------
    def _solve(self):
        """Solve Cm and t_ref from the two I-F calibration points."""
        R   = self.R_tot_ohm
        Vth = self.Vthresh_V
        I0  = self.I_0_uA  * 1e-6
        Im  = self.I_min_uA * 1e-6
        Ix  = self.I_max_uA * 1e-6
        fm  = self.f_min_Hz
        fx  = self.f_max_Hz

        Vm_min = (Im + I0) * R
        Vm_max = (Ix + I0) * R

        if Vm_min <= Vth:
            raise ValueError(
                f"Vm_ss(I_min + I_0) = {Vm_min:.3f} V  ≤  Vthresh = {Vth} V.\n"
                f"Increase I_min or decrease I_0 / Rm_hi / Ra.")
        if Vm_max <= Vth:
            raise ValueError(
                f"Vm_ss(I_max + I_0) = {Vm_max:.3f} V  ≤  Vthresh = {Vth} V.")

        log_min = np.log(1.0 - Vth / Vm_min)   # negative
        log_max = np.log(1.0 - Vth / Vm_max)   # negative, smaller magnitude

        # From (t_cross_min - t_cross_max) = 1/f_min - 1/f_max  →  solve tau_m
        tau_m = (1.0/fm - 1.0/fx) / (log_max - log_min)
        tc_min = -tau_m * log_min
        t_ref  = 1.0/fm - tc_min

        if t_ref <= 0:
            raise ValueError(
                f"Solved t_ref = {t_ref*1e3:.3f} ms ≤ 0.\n"
                f"The two I-F targets are physically incompatible with "
                f"this circuit topology.\n"
                f"Try: increase I_max, decrease f_max, or change Rm_hi/Ra.")

        self.tau_m_ms  = tau_m * 1e3
        self.Cm_nF     = (tau_m / self.R_tot_ohm) * 1e9
        self.t_ref_ms  = t_ref * 1e3

        # Verify
        self._f_check_min = 1.0 / (tc_min + t_ref)
        tc_max = -tau_m * log_max
        self._f_check_max = 1.0 / (tc_max + t_ref)

    # ------------------------------------------------------------------
    def f_at(self, I_syn_uA: float) -> float:
        """Predicted firing rate at a given synaptic input current (µA)."""
        R   = self.R_tot_ohm
        Vth = self.Vthresh_V
        I0  = self.I_0_uA * 1e-6
        I   = I_syn_uA * 1e-6
        tau_m = self.tau_m_ms * 1e-3
        t_ref = self.t_ref_ms * 1e-3

        Vm_ss = (I + I0) * R
        if Vm_ss <= Vth:
            return 0.0
        if I_syn_uA > self.I_max_uA:
            return 0.0  # depolarisation block
        tc = -tau_m * np.log(1.0 - Vth / Vm_ss)
        return 1.0 / (tc + t_ref)

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> 'NeuronParams':
        keys = ['Ra_ohm','Rm_hi_ohm','Rm_lo_ohm','Vthresh_V','Vreset_V',
                'I_0_uA','I_min_uA','I_max_uA','f_min_Hz','f_max_Hz']
        return cls(**{k: d[k] for k in keys if k in d})

    @classmethod
    def from_json(cls, path: str) -> 'NeuronParams':
        with open(path) as f:
            cfg = json.load(f)
        return cls.from_dict(cfg['neuron'])

    # ------------------------------------------------------------------
    def summary(self):
        print("=" * 52)
        print("  NeuronParams")
        print("=" * 52)
        print(f"  Ra        = {self.Ra_ohm/1e3:.2f} kΩ")
        print(f"  Rm_hi     = {self.Rm_hi_ohm/1e3:.0f} kΩ   Rm_lo = {self.Rm_lo_ohm} Ω")
        print(f"  R_total   = {self.R_tot_ohm/1e3:.2f} kΩ")
        print(f"  Vthresh   = {self.Vthresh_V} V    Vreset = {self.Vreset_V} V")
        print(f"  I_0       = {self.I_0_uA} µA  (tonic bias)")
        print(f"  ── Solved from I-F targets ──────────")
        print(f"  tau_m     = {self.tau_m_ms:.3f} ms")
        print(f"  Cm        = {self.Cm_nF:.2f} nF")
        print(f"  t_ref     = {self.t_ref_ms:.3f} ms  →  f_ceil = {1e3/self.t_ref_ms:.0f} Hz")
        print(f"  f({self.I_min_uA:.0f}µA) = {self._f_check_min:.1f} Hz  (target {self.f_min_Hz} Hz)")
        print(f"  f({self.I_max_uA:.0f}µA) = {self._f_check_max:.1f} Hz  (target {self.f_max_Hz} Hz)")
        print("=" * 52)


# ══════════════════════════════════════════════════════════════════════════════

def build_neurons(N         : int,
                  np_       : NeuronParams,
                  tau_s1_ms : float,
                  tau_s2_ms : float,
                  mode      : str  = 'Is2',
                  name      : str  = 'neurons') -> NeuronGroup:
    """
    Build and return a Brian2 NeuronGroup implementing Eqs. 9-12.

    Parameters
    ----------
    N         : number of neurons
    np_       : NeuronParams instance
    tau_s1_ms : 1st synaptic time constant (ms)  — from SynapseParams
    tau_s2_ms : 2nd synaptic time constant (ms)
    mode      : 'Is1' (AMPA/fast) or 'Is2' (NMDA/slow)
    name      : Brian2 object name (must be unique per start_scope)

    Returns
    -------
    NeuronGroup with state variables: Vm, Is1_exc, Is2_exc, Is1_inh, Is2_inh,
                                       Vpost, Rm_S
    """
    # All shared parameters passed via namespace
    ns = {
        'Cm'    : np_.Cm_nF    * 1e-9 * farad,
        'Ra'    : np_.Ra_ohm          * ohm,
        'I_0'   : np_.I_0_uA   * 1e-6 * amp,
        'tau_s1': tau_s1_ms    * 1e-3 * second,
        'tau_s2': tau_s2_ms    * 1e-3 * second,
    }
    Vth_str = f'{np_.Vthresh_V:.6f}*volt'
    Vr_str  = f'{np_.Vreset_V:.6f}*volt'
    t_ref   = np_.t_ref_ms * 1e-3 * second
    Rm_hi   = np_.Rm_hi_ohm * ohm

    if mode == 'Is1':
        eqs = '''
        dVm/dt      = (-Vm/(Rm_S+Ra) + Is1_exc - Is1_inh + I_0) / Cm : volt
        dIs1_exc/dt = -Is1_exc / tau_s1                                : amp
        dIs2_exc/dt = (-Is2_exc + Is1_exc) / tau_s2                    : amp
        dIs1_inh/dt = -Is1_inh / tau_s1                                : amp
        dIs2_inh/dt = (-Is2_inh + Is1_inh) / tau_s2                    : amp
        Vpost       = Vm * Ra / (Rm_S + Ra)                            : volt
        Rm_S        : ohm
        '''
    elif mode == 'Is2':
        eqs = '''
        dVm/dt      = (-Vm/(Rm_S+Ra) + Is2_exc - Is2_inh + I_0) / Cm : volt
        dIs1_exc/dt = -Is1_exc / tau_s1                                : amp
        dIs2_exc/dt = (-Is2_exc + Is1_exc) / tau_s2                    : amp
        dIs1_inh/dt = -Is1_inh / tau_s1                                : amp
        dIs2_inh/dt = (-Is2_inh + Is1_inh) / tau_s2                    : amp
        Vpost       = Vm * Ra / (Rm_S + Ra)                            : volt
        Rm_S        : ohm
        '''
    else:
        raise ValueError(f"mode must be 'Is1' or 'Is2', got '{mode}'")

    ng = NeuronGroup(N, model=eqs,
                     threshold=f'Vm > {Vth_str}',
                     reset=f'Vm = {Vr_str}',
                     refractory=t_ref,
                     method='euler',
                     namespace=ns,
                     name=name)

    ng.Vm      = np_.Vreset_V  * volt
    ng.Rm_S    = Rm_hi
    ng.Is1_exc = 0 * amp;  ng.Is2_exc = 0 * amp
    ng.Is1_inh = 0 * amp;  ng.Is2_inh = 0 * amp

    return ng