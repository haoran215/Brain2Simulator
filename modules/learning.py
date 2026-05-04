"""
learning.py  —  Learning Modules
===================================
Two learning strategies, auto-selected by regime:

  RATE regime   (f > 1/tau_s)  →  ReservoirReadout
    Fixed network weights. Spike counts / Is2 DC level from reservoir
    neurons form a feature vector. Trained offline with logistic regression.
    Suitable for classification, regression, temporal pattern recognition.

  TEMPORAL regime (f < 1/tau_s)  →  build_stdp_synapses
    Spike-Timing-Dependent Plasticity implemented directly in Brian2.
    Δw = A_plus  * exp(-Δt / tau_plus)   if pre before post  (+LTP)
    Δw = A_minus * exp(-Δt / tau_minus)  if post before pre  (-LTD)
"""

import numpy as np
from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import confusion_matrix, accuracy_score

from brian2 import Synapses, ms, uA


# ─────────────────────────────────────────────────────────────────────────────
# RATE REGIME  —  Reservoir Readout
# ─────────────────────────────────────────────────────────────────────────────
class ReservoirReadout:
    """
    Offline linear readout trained on reservoir spike counts.

    Workflow
    --------
    1. Run SNNNetwork for each class stimulus.
    2. Call extract_spike_counts() for each trial → feature vector.
    3. Call train(X, y) on the training set.
    4. Call predict(X) on the test set.

    Feature vector
    --------------
    For a reservoir of N neurons observed over window [t_start, t_end]:
        x = [count_0, count_1, ..., count_{N-1}]   (spike counts)
    Optionally augmented with mean Is2 (DC current) if state_monitor passed.
    """

    def __init__(self, n_classes: int, C: float = 1.0, max_iter: int = 1000):
        self.n_classes = n_classes
        self.scaler    = StandardScaler()
        self.clf       = LogisticRegression(C=C, max_iter=max_iter,
                                             solver='lbfgs')
        self._fitted   = False

    # ------------------------------------------------------------------
    @staticmethod
    def extract_spike_counts(spike_i: np.ndarray, spike_t_s: np.ndarray,
                              n_neurons: int,
                              t_start_s: float, t_end_s: float) -> np.ndarray:
        """
        Spike count feature vector for one trial window.

        Parameters
        ----------
        spike_i   : neuron index array from SpikeMonitor
        spike_t_s : spike time array (seconds)
        n_neurons : number of neurons in population
        t_start_s : window start (seconds)
        t_end_s   : window end   (seconds)

        Returns
        -------
        counts : shape (n_neurons,)  — spike count per neuron
        """
        mask   = (spike_t_s >= t_start_s) & (spike_t_s < t_end_s)
        counts = np.zeros(n_neurons, dtype=np.float32)
        for idx in spike_i[mask]:
            counts[idx] += 1
        return counts

    # ------------------------------------------------------------------
    @staticmethod
    def extract_mean_is2(st_mon, t_start_s: float,
                          t_end_s: float) -> np.ndarray:
        """
        Mean Is2_exc - Is2_inh (net synaptic current) per neuron over window.
        Useful as a complementary feature to spike counts.
        """
        from brian2 import uA as _uA
        t    = st_mon.t / 1.0   # already in seconds if using /second
        mask = (t >= t_start_s) & (t < t_end_s)
        exc  = st_mon.Is2_exc[:, mask] / _uA   # shape [N, T]
        inh  = st_mon.Is2_inh[:, mask] / _uA
        return np.mean(exc - inh, axis=1)

    # ------------------------------------------------------------------
    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Fit the readout classifier.

        Parameters
        ----------
        X : (n_trials, n_features)  — feature matrix
        y : (n_trials,)             — class labels (0 … n_classes-1)

        Returns
        -------
        train_accuracy : float
        """
        X_sc = self.scaler.fit_transform(X)
        self.clf.fit(X_sc, y)
        self._fitted = True
        return self.clf.score(X_sc, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._fitted, "Call train() first."
        return self.clf.predict(self.scaler.transform(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        assert self._fitted, "Call train() first."
        return accuracy_score(y, self.predict(X))

    def confusion_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return confusion_matrix(y, self.predict(X))


# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL REGIME  —  STDP Synapses
# ─────────────────────────────────────────────────────────────────────────────
def build_stdp_synapses(name: str, pre_group, post_group,
                         stdp_cfg: dict):
    """
    Create Brian2 STDP synapses.

    Parameters
    ----------
    name       : unique name string
    pre_group  : Brian2 NeuronGroup (pre-synaptic)
    post_group : Brian2 NeuronGroup (post-synaptic)
    stdp_cfg   : dict with keys:
        Iw_init_uA   : initial synaptic weight (µA)
        Iw_min_uA    : minimum weight (hard lower bound, µA)
        Iw_max_uA    : maximum weight (hard upper bound, µA)
        A_plus       : LTP amplitude  (µA)
        A_minus      : LTD amplitude  (µA)
        tau_plus_ms  : LTP time constant (ms)
        tau_minus_ms : LTD time constant (ms)
        connectivity : 'all_to_all' | 'random'
        p_connect    : probability (for 'random')
        synapse_type : 'exc' (only exc supported for STDP here)

    Returns
    -------
    synapses : Brian2 Synapses object
    """
    Iw_init  = stdp_cfg.get('Iw_init_uA',   20.0) * uA
    Iw_min   = stdp_cfg.get('Iw_min_uA',     0.0) * uA
    Iw_max   = stdp_cfg.get('Iw_max_uA',   100.0) * uA
    A_plus   = stdp_cfg.get('A_plus',         0.01) * uA
    A_minus  = stdp_cfg.get('A_minus',        0.012) * uA
    tau_p    = stdp_cfg.get('tau_plus_ms',   20.0) * ms
    tau_m_   = stdp_cfg.get('tau_minus_ms',  20.0) * ms

    stdp_eqs = '''
    Iw      : amp                      # synaptic weight
    dApre/dt  = -Apre  / tau_p  : amp (event-driven)
    dApost/dt = -Apost / tau_m_ : amp (event-driven)
    '''

    on_pre = '''
    Is1_exc_post += Iw
    Apre  += A_plus
    Iw    = clip(Iw + Apost, Iw_min, Iw_max)
    '''

    on_post = '''
    Apost -= A_minus
    Iw     = clip(Iw + Apre, Iw_min, Iw_max)
    '''

    ns = dict(tau_p=tau_p, tau_m_=tau_m_,
              A_plus=A_plus, A_minus=A_minus,
              Iw_min=Iw_min, Iw_max=Iw_max)

    syn = Synapses(pre_group, post_group,
                   model=stdp_eqs,
                   on_pre=on_pre, on_post=on_post,
                   namespace=ns,
                   name=f'{name}_stdp')

    conn = stdp_cfg.get('connectivity', 'all_to_all')
    if conn == 'all_to_all':
        syn.connect(condition='i != j'
                     if pre_group is post_group else 'True')
    elif conn == 'random':
        p = stdp_cfg.get('p_connect', 0.5)
        syn.connect(condition='i != j'
                     if pre_group is post_group else 'True', p=p)

    syn.Iw = Iw_init
    return syn
