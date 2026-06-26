"""
Counterfactual Explanation Generator for Biocomputers
Author: Adham Aboulkheir | University of Essex | PhD Research
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CounterfactualExplanation:
    original_sample: np.ndarray
    counterfactual: np.ndarray
    original_prediction: str
    counterfactual_prediction: str
    changed_electrodes: List[int]
    change_magnitude: float
    sparsity: float  # fraction of electrodes changed


class CounterfactualGenerator:
    """
    Generate counterfactual explanations for MEA spike predictions.

    A counterfactual answers: 'What minimal change to the electrode firing
    pattern would flip the prediction from SPIKE to NO-SPIKE (or vice versa)?'
    """

    def __init__(self, model, n_iterations: int = 100, step_size: float = 0.05,
                 sparsity_weight: float = 0.1):
        self.model = model
        self.n_iterations = n_iterations
        self.step_size = step_size
        self.sparsity_weight = sparsity_weight

    def generate(self, sample: np.ndarray,
                  target_class: Optional[int] = None) -> CounterfactualExplanation:
        """
        Generate a counterfactual explanation for a single sample.

        Parameters
        ----------
        sample       : (n_features,) input sample
        target_class : desired class (None = flip current prediction)
        """
        original_pred = int(self.model.predict(sample.reshape(1, -1))[0])
        target = 1 - original_pred if target_class is None else target_class

        cf = sample.copy().astype(float)
        best_cf = cf.copy()
        best_dist = np.inf

        for iteration in range(self.n_iterations):
            # Gradient-free perturbation
            noise = np.random.normal(0, self.step_size, cf.shape)
            candidate = cf + noise

            # Check if prediction flipped
            pred = int(self.model.predict(candidate.reshape(1, -1))[0])

            if pred == target:
                dist = np.linalg.norm(candidate - sample)
                if dist < best_dist:
                    best_dist = dist
                    best_cf = candidate.copy()

            # Gradient approximation: move toward target
            cf += noise * self.step_size * 0.1

        # Identify changed electrodes
        diff = np.abs(best_cf - sample)
        threshold = np.percentile(diff, 80)
        changed_electrodes = list(np.where(diff > threshold)[0])

        pred_names = {0: "NO-SPIKE", 1: "SPIKE"}

        return CounterfactualExplanation(
            original_sample=sample,
            counterfactual=best_cf,
            original_prediction=pred_names[original_pred],
            counterfactual_prediction=pred_names[target],
            changed_electrodes=changed_electrodes,
            change_magnitude=float(best_dist),
            sparsity=len(changed_electrodes) / len(sample)
        )

    def explain_batch(self, samples: np.ndarray) -> List[CounterfactualExplanation]:
        """Generate counterfactuals for a batch of samples."""
        return [self.generate(s) for s in samples]


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    print("Counterfactual Explanation Demo")
    print("=" * 50)

    np.random.seed(42)
    X, y = make_classification(n_samples=1000, n_features=30, n_informative=10,
                                weights=[0.6, 0.4], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = GradientBoostingClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    generator = CounterfactualGenerator(model, n_iterations=50, step_size=0.1)

    print("\nGenerating counterfactuals for 3 test samples:")
    for i in range(3):
        sample = X_test[i]
        cf = generator.generate(sample)
        print(f"\nSample {i+1}:")
        print(f"  Original:        {cf.original_prediction}")
        print(f"  Counterfactual:  {cf.counterfactual_prediction}")
        print(f"  Changed electrodes: {len(cf.changed_electrodes)} of {len(sample)}")
        print(f"  Change magnitude:   {cf.change_magnitude:.4f}")
        print(f"  Sparsity:           {cf.sparsity:.1%}")
        print(f"  Key electrodes: {cf.changed_electrodes[:5]}")
