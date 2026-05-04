"""
synapse.py  —  Synaptic Module
==============================
Provides SynapseParams (Rs, Cs, Iw, mode) and build_synapse() which
returns a Brian2 Synapses object implementing the Iw·δ(t_spike) kick
into the Is1 stage (Eq. 11).

Connection rules
----------------
  'all_to_all'   — every pre neuron connects to every post neuron
  'one_to_one'   — pre[i] → post[i]   (requires len(pre) == len(post))
  'random_p'     — each pair connects with probability p
  'custom'       — pass i_indices and j_indices explicitly
"""

import json
import numpy as np
from brian2 import *


class SynapseParams:
    """
    Holds all synapse parameters.

    Usage
    -----
    sp = SynapseParams.from_json('config.json')
    sp = SynapseParams(Rs_ohm=10000, Cs_nF=1000, Iw_exc_uA=20, Iw_inh_uA=30)
    """

    def __init__(self,
                 Rs_ohm    : float = 10000.0,
                 Cs_nF     : float = 1000.0,
                 Iw_exc_uA : float = 20.0,
                 Iw_inh_uA : float = 30.0,
                 mode      : str   = 'Is2'):

        self.Rs_ohm    = Rs_ohm
        self.Cs_nF     = Cs_nF
        self.Iw_exc_uA = Iw_exc_uA
        self.Iw_inh_uA = Iw_inh_uA
        self.mode      = mode

        # Derived
        self.tau_s1_ms = Rs_ohm * Cs_nF * 1e-9 * 1e3   # Rs(Ω)*Cs(F) → ms
        self.tau_s2_ms = self.tau_s1_ms                  # equal by default
        self.crossover_Hz = 1e3 / self.tau_s1_ms         # 1/tau_s in Hz

    @classmethod
    def from_dict(cls, d: dict) -> 'SynapseParams':
        keys = ['Rs_ohm','Cs_nF','Iw_exc_uA','Iw_inh_uA','mode']
        return cls(**{k: d[k] for k in keys if k in d})

    @classmethod
    def from_json(cls, path: str) -> 'SynapseParams':
        with open(path) as f:
            cfg = json.load(f)
        return cls.from_dict(cfg['synapse'])

    def summary(self):
        print("=" * 52)
        print("  SynapseParams")
        print("=" * 52)
        print(f"  Rs        = {self.Rs_ohm/1e3:.0f} kΩ")
        print(f"  Cs        = {self.Cs_nF:.0f} nF")
        print(f"  tau_s1    = Rs*Cs = {self.tau_s1_ms:.1f} ms")
        print(f"  tau_s2    = {self.tau_s2_ms:.1f} ms")
        print(f"  crossover = 1/tau_s = {self.crossover_Hz:.0f} Hz")
        print(f"  Iw_exc    = {self.Iw_exc_uA:.1f} µA   Iw_inh = {self.Iw_inh_uA:.1f} µA")
        print(f"  mode      = {self.mode}")
        print("=" * 52)


# ══════════════════════════════════════════════════════════════════════════════

def build_synapse(source        : NeuronGroup,
                  target        : NeuronGroup,
                  sp            : SynapseParams,
                  syn_type      : str   = 'exc',
                  rule          : str   = 'all_to_all',
                  p             : float = 0.3,
                  weight_scale  : float = 1.0,
                  i_idx         = None,
                  j_idx         = None,
                  name          : str   = 'syn') -> Synapses:
    """
    Build and return a Brian2 Synapses object.

    Parameters
    ----------
    source       : pre-synaptic NeuronGroup (or PoissonGroup)
    target       : post-synaptic NeuronGroup built by build_neurons()
    sp           : SynapseParams
    syn_type     : 'exc' — adds to Is1_exc; 'inh' — adds to Is1_inh
    rule         : 'all_to_all' | 'one_to_one' | 'random_p' | 'custom'
    p            : connection probability for 'random_p'
    weight_scale : multiplier on Iw_exc or Iw_inh
    i_idx, j_idx : arrays of pre/post indices for 'custom' rule
    name         : Brian2 object name (must be unique per start_scope)

    Returns
    -------
    Synapses object (already connected, not yet added to Network)
    """
    if syn_type == 'exc':
        Iw = sp.Iw_exc_uA * 1e-6 * weight_scale * amp
        on_pre = f'Is1_exc_post += {float(Iw/amp):.6e}*amp'
    elif syn_type == 'inh':
        Iw = sp.Iw_inh_uA * 1e-6 * weight_scale * amp
        on_pre = f'Is1_inh_post += {float(Iw/amp):.6e}*amp'
    else:
        raise ValueError(f"syn_type must be 'exc' or 'inh', got '{syn_type}'")

    syn = Synapses(source, target, on_pre=on_pre, name=name)

    if rule == 'all_to_all':
        syn.connect()
    elif rule == 'one_to_one':
        assert len(source) == len(target), \
            "one_to_one requires equal population sizes"
        syn.connect(j='i')
    elif rule == 'random_p':
        syn.connect(p=p)
    elif rule == 'custom':
        assert i_idx is not None and j_idx is not None, \
            "custom rule requires i_idx and j_idx arrays"
        syn.connect(i=i_idx, j=j_idx)
    else:
        raise ValueError(f"Unknown rule '{rule}'. "
                         f"Use: all_to_all, one_to_one, random_p, custom")

    return syn


def build_stdp_synapse(source       : NeuronGroup,
                       target       : NeuronGroup,
                       sp           : SynapseParams,
                       stdp_params  : dict,
                       syn_type     : str   = 'exc',
                       rule         : str   = 'all_to_all',
                       p            : float = 0.3,
                       w_init       : float = 1.0,
                       name         : str   = 'stdp_syn') -> Synapses:
    """
    Build a Synapses object with STDP weight update rule.

    STDP rule:
        on_pre:  Apre += A_plus;  w = clip(w + Apost, w_min, w_max)
        on_post: Apost += -A_minus; w = clip(w + Apre, w_min, w_max)

    Parameters
    ----------
    stdp_params : dict with keys A_plus, A_minus, tau_plus_ms, tau_minus_ms,
                  w_min, w_max
    w_init      : initial weight (multiplier on Iw)
    """
    A_plus    = stdp_params['A_plus']
    A_minus   = stdp_params['A_minus']
    tau_plus  = stdp_params['tau_plus_ms']  * 1e-3 * second
    tau_minus = stdp_params['tau_minus_ms'] * 1e-3 * second
    w_min     = stdp_params.get('w_min', 0.0)
    w_max     = stdp_params.get('w_max', 2.0)

    Iw_base = (sp.Iw_exc_uA if syn_type == 'exc' else sp.Iw_inh_uA) * 1e-6

    stdp_eqs = '''
    w       : 1
    dApre/dt  = -Apre  / tau_plus  : 1 (event-driven)
    dApost/dt = -Apost / tau_minus : 1 (event-driven)
    '''

    if syn_type == 'exc':
        on_pre  = (f'Is1_exc_post += w * {Iw_base:.6e}*amp\n'
                   f'Apre += {A_plus}\n'
                   f'w = clip(w + Apost, {w_min}, {w_max})')
        on_post = (f'Apost -= {A_minus}\n'
                   f'w = clip(w + Apre, {w_min}, {w_max})')
    else:
        on_pre  = (f'Is1_inh_post += w * {Iw_base:.6e}*amp\n'
                   f'Apre += {A_plus}\n'
                   f'w = clip(w + Apost, {w_min}, {w_max})')
        on_post = (f'Apost -= {A_minus}\n'
                   f'w = clip(w + Apre, {w_min}, {w_max})')

    syn = Synapses(source, target,
                   model=stdp_eqs,
                   on_pre=on_pre,
                   on_post=on_post,
                   namespace={'tau_plus': tau_plus, 'tau_minus': tau_minus},
                   name=name)

    if rule == 'all_to_all':
        syn.connect()
    elif rule == 'one_to_one':
        syn.connect(j='i')
    elif rule == 'random_p':
        syn.connect(p=p)

    syn.w = w_init
    return syn