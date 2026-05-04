"""
learning.py  —  Learning / Readout Module
==========================================
Provides readout classifiers for reservoir computing and utilities
for evaluating SNN classification performance.

ReservoirReadout
  Wraps sklearn classifiers to work on Is2 feature matrices from
  SNNNetwork.get_features().  Supports logistic regression, linear SVM,
  and ridge regression.

STDPModule is implemented directly in synapse.py via build_stdp_synapse().
"""

import numpy as np
import json
from sklearn.linear_model       import LogisticRegression, RidgeClassifier
from sklearn.svm                 import LinearSVC
from sklearn.preprocessing       import StandardScaler
from sklearn.model_selection     import cross_val_score, StratifiedKFold
from sklearn.metrics             import confusion_matrix, classification_report


class ReservoirReadout:
    """
    Linear readout trained on reservoir Is2 states.

    Usage
    -----
    readout = ReservoirReadout.from_json('config.json')
    readout.fit(X_train, y_train)
    y_pred = readout.predict(X_test)
    readout.report(X_test, y_test, class_names=['0°','45°',...])
    """

    def __init__(self,
                 classifier : str   = 'logistic',
                 C          : float = 1.0,
                 max_iter   : int   = 1000):
        self.classifier_type = classifier
        self.C        = C
        self.scaler   = StandardScaler()

        if classifier == 'logistic':
            self.clf = LogisticRegression(C=C, max_iter=max_iter,
                                          multi_class='multinomial',
                                          solver='lbfgs')
        elif classifier == 'linear_svm':
            self.clf = LinearSVC(C=C, max_iter=max_iter)
        elif classifier == 'ridge':
            self.clf = RidgeClassifier(alpha=1.0/C)
        else:
            raise ValueError(f"Unknown classifier '{classifier}'. "
                             f"Use: logistic, linear_svm, ridge")

        self._fitted = False

    @classmethod
    def from_json(cls, path: str) -> 'ReservoirReadout':
        with open(path) as f:
            cfg = json.load(f)
        p = cfg['learning']['reservoir_readout']
        return cls(classifier = p.get('classifier', 'logistic'),
                   C          = p.get('C', 1.0))

    # ──────────────────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the readout.

        Parameters
        ----------
        X : (n_trials, n_reservoir)  — mean Is2 per neuron per trial
        y : (n_trials,)              — class labels
        """
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._fitted, "Call fit() first"
        return self.clf.predict(self.scaler.transform(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        assert self._fitted, "Call fit() first"
        return self.clf.score(self.scaler.transform(X), y)

    def cross_val(self, X: np.ndarray, y: np.ndarray,
                  n_folds: int = 5) -> np.ndarray:
        """k-fold cross-validation accuracy scores."""
        X_scaled = self.scaler.fit_transform(X)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        return cross_val_score(self.clf, X_scaled, y, cv=cv)

    def confusion(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return confusion_matrix(y, self.predict(X))

    def report(self, X: np.ndarray, y: np.ndarray,
               class_names: list = None) -> str:
        y_pred = self.predict(X)
        print(classification_report(y, y_pred, target_names=class_names))
        acc = np.mean(y_pred == y)
        chance = 1.0 / len(np.unique(y))
        print(f"  Accuracy : {acc*100:.1f}%   Chance: {chance*100:.1f}%   "
              f"Gain: {acc/chance:.1f}×")
        return y_pred


def plot_confusion(cm        : np.ndarray,
                   class_names: list,
                   title     : str  = 'Confusion Matrix',
                   ax               = None):
    """Plot a confusion matrix with percentage annotations."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Fraction')

    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm[i, j]
            pct = cm_norm[i, j]
            color = 'white' if pct > 0.5 else 'black'
            ax.text(j, i, f'{val}\n({pct*100:.0f}%)',
                    ha='center', va='center', fontsize=8, color=color)

    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title, fontweight='bold')
    return ax