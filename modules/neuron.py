"""
neuron.py  —  NeuronPopulation
================================
Wraps a Brian2 NeuronGroup (memristor aLIF, Eqs. 9-12) with:
  • Automatic Cm / t_ref solver from user I-F targets
  • Explicit tau_m (not a comment — a computed parameter)
  • Spike and state monitors
  • Support for 'Is1' (AMPA-like) and 'Is2' (NMDA-like) drive modes

Usage
-----
    from modules.neuron import NeuronPopulation
    pop = NeuronPopulation('reservoir', n=4, params=cfg['neuron_defaults'])
    # pop.group   -> brian2 NeuronGroup
    # pop.sp_mon  -> SpikeMonitor
    # pop.st_mon  -> StateMonitor
    # pop.objects -> list of all brian2 objects (add to Network)
"""

import numpy as np
from brian2 import (NeuronGroup, SpikeMonitor, StateMonitor,
                    nF, ohm, kohm, volt, ms, second, uA, amp)


# ─────────────────────────────────────────────────────────────────────────────
def solve_neuron_params(cfg: dict) -> dict:
    """
    Solve Cm and t_ref analytically so that:
        f(I_min) = f_min_Hz   and   f(I_max) = f_max_Hz

    Returns a copy of cfg enriched with solved fields:
        tau_m_ms, Cm_nF, t_ref_ms, f_cross_Hz
    """
    Ra      = cfg['Ra_ohm']
    Rm_hi   = cfg['Rm_hi_ohm']
    Vth     = cfg['Vthresh_V']
    I_min   = cfg['I_min_uA'] * 1e-6    # A
    I_max   = cfg['I_max_uA'] * 1e-6    # A
    f_min   = cfg['f_min_Hz']
    f_max   = cfg['f_max_Hz']
    I_0     = cfg.get('I_0_uA', 0.0) * 1e-6

    R_tot   = Ra + Rm_hi

    Vm_min  = (I_min + I_0) * R_tot
    Vm_max  = (I_max + I_0) * R_tot

    assert Vm_min > Vth, (
        f"Vm_ss(I_min+I_0)={Vm_min:.3f}V <= Vthresh={Vth}V — "
        f"neuron cannot fire at I_min. Reduce I_0 or increase I_min.")
    assert Vm_max > Vth, (
        f"Vm_ss(I_max+I_0)={Vm_max:.3f}V <= Vthresh={Vth}V — "
        f"neuron cannot fire at I_max.")

    log_min = np.log(1 - Vth / Vm_min)   # < 0
    log_max = np.log(1 - Vth / Vm_max)   # < 0

    tau_m   = (1/f_min - 1/f_max) / (-(log_min - log_max))   # seconds
    tc_min  = -tau_m * log_min
    t_ref   = 1/f_min - tc_min

    assert t_ref > 0, (
        f"Solved t_ref={t_ref*1e3:.3f}ms < 0 — targets are physically "
        f"impossible with these components. Try reducing f_max or R_tot.")

    Cm = tau_m / R_tot

    solved = dict(cfg)
    solved['tau_m_ms']   = tau_m * 1e3
    solved['Cm_nF']      = Cm  * 1e9
    solved['t_ref_ms']   = t_ref * 1e3
    solved['f_ceil_Hz']  = 1.0 / t_ref
    return solved


# ─────────────────────────────────────────────────────────────────────────────
# Brian2 equation templates  (referenced via explicit namespace)

_EQS_IS2 = '''
dVm/dt      = (-Vm/(Rm_S + Ra) + Is2_exc - Is2_inh + I_0) / Cm : volt
dIs1_exc/dt = -Is1_exc / tau_s1                                  : amp
dIs2_exc/dt = (-Is2_exc + Is1_exc) / tau_s2                      : amp
dIs1_inh/dt = -Is1_inh / tau_s1                                  : amp
dIs2_inh/dt = (-Is2_inh + Is1_inh) / tau_s2                      : amp
Vpost       = Vm * Ra / (Rm_S + Ra)                              : volt
Rm_S        : ohm
'''

_EQS_IS1 = '''
dVm/dt      = (-Vm/(Rm_S + Ra) + Is1_exc - Is1_inh + I_0) / Cm : volt
dIs1_exc/dt = -Is1_exc / tau_s1                                  : amp
dIs2_exc/dt = (-Is2_exc + Is1_exc) / tau_s2                      : amp
dIs1_inh/dt = -Is1_inh / tau_s1                                  : amp
dIs2_inh/dt = (-Is2_inh + Is1_inh) / tau_s2                      : amp
Vpost       = Vm * Ra / (Rm_S + Ra)                              : volt
Rm_S        : ohm
'''


# ─────────────────────────────────────────────────────────────────────────────
class NeuronPopulation:
    """
    One population of memristor aLIF neurons.

    Parameters
    ----------
    name : str
        Unique population identifier (used as Brian2 name prefix).
    n : int
        Number of neurons.
    params : dict
        Must contain all keys from default_network.json → "neuron_defaults".
        Cm_nF and t_ref_ms are solved automatically if absent.
    record_state : list[str]
        Which state variables to record (default: Vm, Is2_exc, Is2_inh).
    """

    def __init__(self, name: str, n: int, params: dict,
                 record_state: list = None):
        self.name   = name
        self.n      = n
        self.params = solve_neuron_params(params)
        self._build(record_state or ['Vm', 'Is1_exc', 'Is1_inh',
                                      'Is2_exc', 'Is2_inh'])

    # ------------------------------------------------------------------
    def _build(self, record_vars):
        p   = self.params
        nm  = self.name

        # --- Brian2 Quantities (passed as explicit namespace) ----------
        Cm      = p['Cm_nF']    * nF
        Ra      = p['Ra_ohm']   * ohm
        Rm_hi   = p['Rm_hi_ohm']* ohm
        tau_s1  = p['tau_s1_ms']* ms
        tau_s2  = p['tau_s2_ms']* ms
        I_0     = p.get('I_0_uA', 0.0) * uA
        Vthresh = p['Vthresh_V']* volt
        Vreset  = 0.0           * volt
        t_ref   = p['t_ref_ms'] * ms

        ns = dict(Cm=Cm, Ra=Ra, tau_s1=tau_s1,
                  tau_s2=tau_s2, I_0=I_0)

        mode = p.get('network_mode', 'Is2')
        eqs  = _EQS_IS2 if mode == 'Is2' else _EQS_IS1

        # --- NeuronGroup ----------------------------------------------
        self.group = NeuronGroup(
            self.n, eqs,
            threshold = 'Vm > Vthresh',
            reset     = 'Vm = Vreset',
            refractory= t_ref,
            namespace = ns,
            method    = 'euler',
            name      = f'{nm}_group'
        )
        # Threshold / reset need Vthresh, Vreset in namespace too
        self.group.namespace.update(Vthresh=Vthresh, Vreset=Vreset)

        # --- Initial conditions ---------------------------------------
        self.group.Vm      = Vreset
        self.group.Rm_S    = Rm_hi
        self.group.Is1_exc = 0 * uA
        self.group.Is2_exc = 0 * uA
        self.group.Is1_inh = 0 * uA
        self.group.Is2_inh = 0 * uA

        # --- Monitors -------------------------------------------------
        self.sp_mon = SpikeMonitor(self.group,  name=f'{nm}_spikes')
        self.st_mon = StateMonitor(self.group, record_vars,
                                   record=True, dt=0.5*ms,
                                   name=f'{nm}_states')

        self.objects = [self.group, self.sp_mon, self.st_mon]

    # ------------------------------------------------------------------
    def summary(self) -> None:
        p = self.params
        print(f"  NeuronPopulation '{self.name}'  n={self.n}")
        print(f"    tau_m  = {p['tau_m_ms']:.3f} ms  "
              f"Cm = {p['Cm_nF']:.2f} nF  "
              f"t_ref = {p['t_ref_ms']:.3f} ms")
        print(f"    f_min  = {p['f_min_Hz']:.0f} Hz @ {p['I_min_uA']:.0f} µA  "
              f"f_max = {p['f_max_Hz']:.0f} Hz @ {p['I_max_uA']:.0f} µA")
        print(f"    I_0    = {p.get('I_0_uA', 0):.1f} µA  "
              f"mode = {p.get('network_mode','Is2')}")
