"""
synapse.py  —  SynapseConnection
==================================
Wraps Brian2 Synapses implementing Eqs. 11-12:
    tau_s1 · dIs1/dt = -Is1  [+ Iw·δ(t_spike) via on_pre]
    tau_s2 · dIs2/dt = -Is2 + Is1

Spike weight Iw (current delta) is always injected into Is1.
Which stage (Is1 or Is2) drives Vm is controlled in NeuronPopulation.

Connectivity options (JSON "connectivity" field)
-------------------------------------------------
  "all_to_all"  — every pre connects to every post
  "one_to_one"  — pre[i] → post[i]  (requires same population size)
  "random"      — each pair connects with probability p_connect
  "fixed_in"    — each post neuron receives exactly k_in connections
"""

import numpy as np
from brian2 import Synapses, uA, ms


class SynapseConnection:
    """
    One directed connection between two NeuronPopulations (or PoissonGroups).

    Parameters
    ----------
    name : str
        Unique identifier.
    pre  : brian2 Group
        Pre-synaptic population (.group attribute of NeuronPopulation).
    post : brian2 NeuronGroup
        Post-synaptic population (.group attribute of NeuronPopulation).
    syn_cfg : dict
        Keys: synapse_type ('exc'|'inh'), connectivity, Iw_uA,
              p_connect (for 'random'), k_in (for 'fixed_in'),
              allow_self (bool, default False).
    """

    def __init__(self, name: str, pre, post, syn_cfg: dict):
        self.name    = name
        self.syn_cfg = syn_cfg
        self._build(pre, post, syn_cfg)

    # ------------------------------------------------------------------
    def _build(self, pre, post, cfg):
        stype    = cfg.get('synapse_type', 'exc')   # 'exc' or 'inh'
        Iw       = cfg.get('Iw_uA', 20.0) * uA
        allow_self = cfg.get('allow_self', False)

        # Weight injected into Is1_exc or Is1_inh depending on type
        if stype == 'exc':
            on_pre = 'Is1_exc_post += Iw_syn'
        elif stype == 'inh':
            on_pre = 'Is1_inh_post += Iw_syn'
        else:
            raise ValueError(f"synapse_type must be 'exc' or 'inh', got '{stype}'")

        ns = {'Iw_syn': Iw}

        self.synapses = Synapses(
            pre, post,
            on_pre    = on_pre,
            namespace = ns,
            name      = f'{self.name}_syn'
        )

        # --- Connectivity --------------------------------------------
        conn = cfg.get('connectivity', 'all_to_all')

        if conn == 'all_to_all':
            if allow_self:
                self.synapses.connect()
            else:
                self.synapses.connect(condition='i != j'
                                       if pre is post else 'True')

        elif conn == 'one_to_one':
            self.synapses.connect(j='i')

        elif conn == 'random':
            p = cfg.get('p_connect', 0.5)
            if pre is post and not allow_self:
                self.synapses.connect(condition='i != j', p=p)
            else:
                self.synapses.connect(p=p)

        elif conn == 'fixed_in':
            k    = cfg.get('k_in', 3)
            n_pre  = len(pre)
            n_post = len(post)
            rng    = np.random.default_rng(cfg.get('seed', 0))
            sources, targets = [], []
            for j in range(n_post):
                candidates = [i for i in range(n_pre)
                               if allow_self or pre is not post or i != j]
                chosen = rng.choice(candidates,
                                     size=min(k, len(candidates)),
                                     replace=False)
                sources.extend(chosen)
                targets.extend([j] * len(chosen))
            self.synapses.connect(i=sources, j=targets)

        else:
            raise ValueError(f"Unknown connectivity '{conn}'. "
                             f"Use: all_to_all, one_to_one, random, fixed_in")

        self.objects = [self.synapses]

    # ------------------------------------------------------------------
    def n_synapses(self) -> int:
        return len(self.synapses)

    def summary(self) -> None:
        cfg = self.syn_cfg
        print(f"  SynapseConnection '{self.name}'  "
              f"type={cfg.get('synapse_type','exc')}  "
              f"conn={cfg.get('connectivity','all_to_all')}  "
              f"Iw={cfg.get('Iw_uA',20):.0f}µA  "
              f"n_syn={self.n_synapses()}")
