"""Electrode Importance Analysis — Author: Adham Aboulkheir | University of Essex | PhD Research"""
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass

@dataclass
class ElectrodeImportanceResult:
    electrode_ids: np.ndarray; importance_scores: np.ndarray; top_k_indices: np.ndarray; method: str
    def top_k_report(self, k=10):
        lines = [f"Top {k} Important Electrodes ({self.method})", "="*40]
        for rank, idx in enumerate(self.top_k_indices[:k], 1):
            lines.append(f"  Rank {rank:2d}: Electrode_{idx+1:3d}  score={self.importance_scores[idx]:.4f}")
        return "\n".join(lines)

class ElectrodeImportanceAnalyser:
    def __init__(self, method="permutation", n_selected=20):
        self.method = method; self.n_selected = n_selected; self.scaler = MinMaxScaler()
    def compute(self, X, y):
        X_s = self.scaler.fit_transform(X); n = X.shape[1]
        if self.method == "permutation":
            model = GradientBoostingClassifier(n_estimators=50, random_state=42)
            model.fit(X_s, y)
            result = permutation_importance(model, X_s, y, n_repeats=5, random_state=42)
            scores = result.importances_mean
        elif self.method == "mutual_info":
            from sklearn.feature_selection import mutual_info_classif
            scores = mutual_info_classif(X_s, y, random_state=42)
        else:
            scores = np.array([abs(np.corrcoef(X_s[:,i], y)[0,1]) for i in range(n)])
            scores = np.nan_to_num(scores)
        top_k = np.argsort(scores)[::-1][:self.n_selected]
        return ElectrodeImportanceResult(np.arange(n), scores, top_k, self.method)

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    np.random.seed(42)
    X, y = make_classification(n_samples=2000, n_features=142, n_informative=20, weights=[0.6,0.4], random_state=42)
    analyser = ElectrodeImportanceAnalyser(method="permutation", n_selected=20)
    result = analyser.compute(X, y)
    print(result.top_k_report(10))
