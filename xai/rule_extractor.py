"""
Fuzzy Rule Extractor for Biocomputer Explainability
Author: Adham Aboulkheir | University of Essex | PhD Research
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class FuzzyRule:
    antecedents: Dict[int, str]   # {electrode_id: term}
    consequent: str               # "SPIKE" or "NO-SPIKE"
    confidence: float
    support: float
    weight: float

    def to_natural_language(self, electrode_names: Dict[int, str] = None) -> str:
        """Convert rule to human-readable IF-THEN format."""
        conditions = []
        for eid, term in self.antecedents.items():
            name = electrode_names.get(eid, f"Electrode_{eid+1}") if electrode_names else f"Electrode_{eid+1}"
            conditions.append(f"{name} is {term}")
        return (f"IF {' AND '.join(conditions)} "
                f"THEN {self.consequent} "
                f"(Conf: {self.confidence:.1%}, Support: {self.support:.1%})")


class FuzzyRuleExtractor:
    """
    Extract interpretable fuzzy IF-THEN rules from MEA data.
    Rules explain which electrode firing patterns lead to spike/no-spike classification.
    """

    TERMS = ["LOW", "MEDIUM", "HIGH"]

    def __init__(self, min_confidence: float = 0.7, min_support: float = 0.05,
                 max_antecedents: int = 3):
        self.min_confidence = min_confidence
        self.min_support = min_support
        self.max_antecedents = max_antecedents
        self.rules_: List[FuzzyRule] = []

    def _fuzzify(self, x: float) -> Dict[str, float]:
        """Fuzzify a normalised value into LOW/MEDIUM/HIGH memberships."""
        low    = max(0, min(1, 1 - 2 * x))
        medium = max(0, min(2 * x, 2 - 2 * x))
        high   = max(0, min(1, 2 * x - 1))
        return {"LOW": low, "MEDIUM": medium, "HIGH": high}

    def extract_rules(self, X: np.ndarray, y: np.ndarray,
                       selected_electrodes: List[int]) -> List[FuzzyRule]:
        """
        Extract fuzzy rules from training data.

        Parameters
        ----------
        X                  : (n_samples, n_electrodes) normalised firing rates
        y                  : (n_samples,) class labels (0=NO-SPIKE, 1=SPIKE)
        selected_electrodes: list of electrode indices to consider
        """
        from sklearn.preprocessing import MinMaxScaler
        X_norm = MinMaxScaler().fit_transform(X)

        rules = []
        n_electrodes = len(selected_electrodes)

        for cl, label in [(0, "NO-SPIKE"), (1, "SPIKE")]:
            Xc = X_norm[y == cl]
            if len(Xc) == 0:
                continue

            # Generate candidate rules from pairs of electrodes
            for i in range(min(n_electrodes, 8)):
                for j in range(i + 1, min(n_electrodes, 8)):
                    eid_i = selected_electrodes[i]
                    eid_j = selected_electrodes[j]

                    for ti in self.TERMS:
                        for tj in self.TERMS:
                            # Compute support
                            mf_i = self._fuzzify(Xc[:, eid_i].mean())
                            mf_j = self._fuzzify(Xc[:, eid_j].mean())
                            support = min(mf_i[ti], mf_j[tj])

                            if support < self.min_support:
                                continue

                            # Compute confidence
                            all_mf_i = self._fuzzify(X_norm[:, eid_i].mean())
                            all_mf_j = self._fuzzify(X_norm[:, eid_j].mean())
                            total = min(all_mf_i[ti], all_mf_j[tj])
                            confidence = support / (total + 1e-9)

                            if confidence >= self.min_confidence:
                                rules.append(FuzzyRule(
                                    antecedents={eid_i: ti, eid_j: tj},
                                    consequent=label,
                                    confidence=min(confidence, 1.0),
                                    support=support,
                                    weight=confidence * support
                                ))

        rules.sort(key=lambda r: -r.weight)
        self.rules_ = rules[:200]
        return self.rules_

    def get_top_rules(self, n: int = 10, consequent: str = None) -> List[FuzzyRule]:
        """Get top-n rules, optionally filtered by consequent class."""
        rules = self.rules_
        if consequent:
            rules = [r for r in rules if r.consequent == consequent]
        return rules[:n]

    def rules_summary(self, n: int = 5) -> str:
        """Generate a human-readable summary of the top rules."""
        lines = [f"Top {n} Fuzzy Rules (of {len(self.rules_)} total)", "=" * 60]
        for i, rule in enumerate(self.rules_[:n], 1):
            lines.append(f"Rule {i:2d}: {rule.to_natural_language()}")
        return "\n".join(lines)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    print("Fuzzy Rule Extractor Demo")
    print("=" * 50)

    np.random.seed(42)
    X, y = make_classification(n_samples=2000, n_features=142, n_informative=20,
                                weights=[0.6, 0.4], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Select top 20 electrodes
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import MinMaxScaler
    X_norm = MinMaxScaler().fit_transform(X_train)
    mi_scores = mutual_info_classif(X_norm, y_train, random_state=42)
    selected = list(np.argsort(mi_scores)[::-1][:20])

    extractor = FuzzyRuleExtractor(min_confidence=0.7, min_support=0.05)
    rules = extractor.extract_rules(X_train, y_train, selected)

    print(f"Extracted {len(rules)} rules")
    print()
    print(extractor.rules_summary(n=5))

    print("\nSPIKE rules:", len(extractor.get_top_rules(consequent="SPIKE")))
    print("NO-SPIKE rules:", len(extractor.get_top_rules(consequent="NO-SPIKE")))
