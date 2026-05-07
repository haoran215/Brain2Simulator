"""
network.py  —  SNNNetwork
==========================
Reads a JSON config and assembles a Brian2 Network:
    PoissonGroup inputs → aLIF reservoir populations → monitors

Main class: SNNNetwork
  SNNNetwork(config_path)    → builds immediately from JSON
  .set_poisson_rate(id, hz)  → change input rate between trials
  .run(duration_s)           → advance simulation (cumulative time)
  .get_spikes(id)            → (spike_i, spike_t_seconds) since t=0
  .populations               → dict[str, NeuronPopulation] (alif only)
  .net                       → Brian2 Network object (for .t access)
  .summary()                 → print network overview
"""

import json
import numpy as np
from brian2 import (
    start_scope, seed, defaultclock,
    Network, PoissonGroup, SpikeMonitor,
    Hz, ms, second,
)

from .neuron  import NeuronPopulation
from .synapse import SynapseConnection


def _strip_comments(d: dict) -> dict:
    """Remove JSON comment keys (any key starting with '_') from a dict."""
    return {k: v for k, v in d.items() if not k.startswith('_')}


class SNNNetwork:
    """
    Full SNN assembled from a JSON config file.

    Attributes
    ----------
    populations : dict[str, NeuronPopulation]
        aLIF populations, keyed by their id string.
        Example: snn.populations['reservoir'].n

    net : brian2.Network (property)
        The assembled Brian2 Network. Use snn.net.t to read current time.
    """

    def __init__(self, config_path: str):
        with open(config_path) as fh:
            cfg = json.load(fh)

        self._sim_cfg         = cfg['simulation']
        self._neuron_defaults = _strip_comments(cfg['neuron_defaults'])
        self._pop_cfg_list    = cfg['populations']
        self._conn_cfg_list   = cfg['connections']

        # Populated by build()
        self.populations:       dict = {}
        self._poisson_groups:   dict = {}
        self._poisson_monitors: dict = {}
        self._connections:      dict = {}
        self._network                = None

        self.build()

    # ──────────────────────────────────────────────────────────── build ───────
    def build(self, seed_val: int = None) -> None:
        """
        (Re)build all Brian2 objects and assemble the Network.
        Called automatically by __init__. Call again to change the RNG seed.
        """
        if seed_val is None:
            seed_val = self._sim_cfg.get('seed', 42)

        start_scope()
        seed(seed_val)
        defaultclock.dt = self._sim_cfg['dt_ms'] * ms

        # Re-initialise registries (safe for repeated build() calls)
        self.populations       = {}
        self._poisson_groups   = {}
        self._poisson_monitors = {}
        self._connections      = {}
        all_objects            = []

        # ── Pass 1: create all populations ───────────────────────────────────
        for pop_cfg in self._pop_cfg_list:
            pop_id   = pop_cfg['id']
            pop_type = pop_cfg['type']
            n        = pop_cfg['n']

            if pop_type == 'poisson':
                rate_hz = pop_cfg.get('rate_Hz', 0.0)
                pg  = PoissonGroup(n, rates=rate_hz * Hz,
                                   name=f'{pop_id}_poisson')
                mon = SpikeMonitor(pg, name=f'{pop_id}_spikes')
                self._poisson_groups[pop_id]   = pg
                self._poisson_monitors[pop_id] = mon
                all_objects.extend([pg, mon])

            elif pop_type == 'alif':
                params = dict(self._neuron_defaults)
                params.update(_strip_comments(
                    pop_cfg.get('neuron_overrides', {})))
                pop = NeuronPopulation(pop_id, n=n, params=params)
                self.populations[pop_id] = pop
                all_objects.extend(pop.objects)

            else:
                raise ValueError(
                    f"Unknown population type '{pop_type}' for '{pop_id}'. "
                    f"Use 'poisson' or 'alif'.")

        # ── Pass 2: create all connections (after all populations exist) ──────
        for conn_cfg in self._conn_cfg_list:
            conn_id  = conn_cfg['id']
            pre_grp  = self._get_brian2_group(conn_cfg['pre'])
            post_grp = self._get_brian2_group(conn_cfg['post'])
            syn_cfg  = _strip_comments(
                {k: v for k, v in conn_cfg.items()
                 if k not in ('id', 'pre', 'post')})
            conn = SynapseConnection(conn_id, pre_grp, post_grp, syn_cfg)
            self._connections[conn_id] = conn
            all_objects.extend(conn.objects)

        self._network = Network(*all_objects)

    # ───────────────────────────────────────── _get_brian2_group (private) ────
    def _get_brian2_group(self, pop_id: str):
        """Return the raw Brian2 group for any population id."""
        if pop_id in self.populations:
            return self.populations[pop_id].group   # NeuronGroup
        if pop_id in self._poisson_groups:
            return self._poisson_groups[pop_id]     # PoissonGroup
        raise KeyError(
            f"Population '{pop_id}' not found. "
            f"Available: {list(self.populations) + list(self._poisson_groups)}")

    # ────────────────────────────────────────────────── set_poisson_rate ──────
    def set_poisson_rate(self, pop_id: str, rate_hz: float) -> None:
        """
        Set the firing rate of a Poisson input population.

        Call this before each trial to apply direction-tuned input rates.
        Takes effect at the next simulation timestep.
        """
        if pop_id not in self._poisson_groups:
            raise KeyError(
                f"'{pop_id}' is not a Poisson population. "
                f"Poisson groups: {list(self._poisson_groups)}")
        self._poisson_groups[pop_id].rates = rate_hz * Hz

    # ──────────────────────────────────────────────────────────── run ─────────
    def run(self, duration_s: float) -> None:
        """
        Advance the simulation by duration_s seconds.

        Simulation time is cumulative across calls. SpikeMonitors accumulate
        all spikes since t=0 — use get_spikes() and filter by time window.
        """
        if self._network is None:
            raise RuntimeError("Network not built. Call build() first.")
        self._network.run(duration_s * second)

    # ──────────────────────────────────────────────────────── get_spikes ──────
    def get_spikes(self, pop_id: str):
        """
        Return all spike data accumulated since t=0 for a population.

        Returns
        -------
        spike_i : np.ndarray  — neuron indices
        spike_t : np.ndarray  — spike times in seconds
        """
        if pop_id in self.populations:
            mon = self.populations[pop_id].sp_mon
        elif pop_id in self._poisson_monitors:
            mon = self._poisson_monitors[pop_id]
        else:
            raise KeyError(
                f"No population '{pop_id}' found. "
                f"Available: {list(self.populations) + list(self._poisson_groups)}")
        return np.array(mon.i), np.array(mon.t / second)

    # ──────────────────────────────────────────────────────── net (property) ──
    @property
    def net(self) -> Network:
        """Brian2 Network — use snn.net.t to read the current simulation time."""
        if self._network is None:
            raise RuntimeError("Network not built. Call build() first.")
        return self._network

    # ──────────────────────────────────────────────────────────── summary ─────
    def summary(self) -> None:
        """Print a human-readable overview of all populations and connections."""
        print("=" * 60)
        print("  SNNNetwork Summary")
        print("=" * 60)
        print(f"  dt = {self._sim_cfg['dt_ms']} ms  |  "
              f"seed = {self._sim_cfg.get('seed', 42)}")
        print()
        print("  Populations:")
        for pop_id, pg in self._poisson_groups.items():
            r = float(pg.rates[0] / Hz)
            print(f"    [poisson] '{pop_id}'  n={len(pg)}  "
                  f"default_rate={r:.1f} Hz")
        for pop in self.populations.values():
            pop.summary()
        print()
        print("  Connections:")
        for conn in self._connections.values():
            conn.summary()
        total = sum(c.n_synapses() for c in self._connections.values())
        print(f"  Total synapses: {total}")
        print("=" * 60)
