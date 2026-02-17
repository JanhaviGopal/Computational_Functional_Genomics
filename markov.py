import numpy as np
import pandas as pd
import itertools
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class MarkovModel:
    """
    Markov model of order (m-1)
    P(x_t | x_{t-m+1} ... x_{t-1})

    Laplace smoothing:
    P = (count + alpha) / (denom + 4*alpha)
    """

    def __init__(self, m, alpha=1.0):
        self.m = m
        self.alpha = alpha
        self.alphabet = ["A", "C", "G", "T"]
        self.model = {}

    def _contexts(self):
        return ["".join(p) for p in itertools.product(self.alphabet, repeat=self.m - 1)]

    def train(self, sequences):
        counts = defaultdict(int)

        # Count m-mers
        for seq in sequences:
            for i in range(len(seq) - self.m + 1):
                counts[seq[i:i+self.m]] += 1

        # Estimate conditional probabilities
        for ctx in self._contexts():
            denom = sum(counts.get(ctx + b, 0) for b in self.alphabet)
            denom += 4 * self.alpha

            for b in self.alphabet:
                num = counts.get(ctx + b, 0) + self.alpha
                self.model[ctx + b] = num / denom

    def log_likelihood(self, seq):
        """
        log P(S | model)
        = sum log P(m-mer)
        """
        ll = 0.0
        for i in range(len(seq) - self.m + 1):
            ll += np.log(self.model.get(seq[i:i+self.m], 1e-12))
        return ll

def score_sequence(seq, model_unbound, model_bound):
    """
    Score(S) = log P(S|Unbound) − log P(S|Bound)
    """
    return (
        model_unbound.log_likelihood(seq)
        - model_bound.log_likelihood(seq)
    )

def cross_validate_markov(df, m, k=5):
    """
    df: dataframe with columns ['sequence', 'label']
    m : m-mer length
    k : number of CV folds
    """

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    all_labels = []
    all_scores = []

    for train_idx, test_idx in kf.split(df):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]

        bound_train = train[train.label == 1].sequence.tolist()
        unbound_train = train[train.label == 0].sequence.tolist()

        model_b = MarkovModel(m)
        model_u = MarkovModel(m)

        model_b.train(bound_train)
        model_u.train(unbound_train)

        for _, row in test.iterrows():
            score = score_sequence(row.sequence, model_u, model_b)
            all_scores.append(score)
            all_labels.append(row.label)

    return np.array(all_labels), np.array(all_scores)

def plot_roc(y_true, scores, title):
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

    return roc_auc

def plot_score_histogram(labels, scores, title):
    plt.figure()
    plt.hist(scores[labels == 1], bins=50, alpha=0.6, label="Bound", color="red")
    plt.hist(scores[labels == 0], bins=50, alpha=0.6, label="Unbound", color="blue")
    plt.xlabel("Markov Likelihood-Ratio Score")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def select_best_m(df, m_values, k=5):
    aucs = {}

    for m in m_values:
        labels, scores = cross_validate_markov(df, m, k)
        auc_val = auc(*roc_curve(labels, scores)[:2])
        aucs[m] = auc_val
        print(f"m={m} → AUC={auc_val:.4f}")

    best_m = max(aucs, key=aucs.get)
    return best_m, aucs
