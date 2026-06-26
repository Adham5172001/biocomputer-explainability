"""Biocomputer Explainability Demo — Author: Adham Aboulkheir | University of Essex"""
import numpy as np, matplotlib, os, sys
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from xai.electrode_importance import ElectrodeImportanceAnalyser
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def main():
    print("Biocomputer Explainability Framework Demo")
    os.makedirs("outputs", exist_ok=True)
    np.random.seed(42)
    X, y = make_classification(n_samples=3000, n_features=142, n_informative=20, n_redundant=10, weights=[0.6,0.4], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    analyser = ElectrodeImportanceAnalyser(method="permutation", n_selected=20)
    result = analyser.compute(X_train, y_train)
    print(result.top_k_report(10))
    n_e = X.shape[1]
    positions = np.array([[i*200, j*200] for i in range(12) for j in range(12)])[:n_e]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#0d1117")
    for ax in axes: ax.set_facecolor("#161b22")
    top20 = result.top_k_indices[:20]; scores = result.importance_scores[top20]; labels = [f"E{i+1}" for i in top20]
    colors = ["#f4a261" if i < 3 else "#00c9b1" for i in range(20)]
    axes[0].barh(labels[::-1], scores[::-1], color=colors[::-1], alpha=0.85)
    axes[0].set_title("Top 20 Electrode Importance", color="white"); axes[0].set_xlabel("Importance Score", color="white"); axes[0].tick_params(colors="white"); axes[0].grid(axis="x", alpha=0.3, color="#21262d")
    all_scores = result.importance_scores; norm = (all_scores - all_scores.min()) / (all_scores.max() - all_scores.min() + 1e-9)
    sc = axes[1].scatter(positions[:,0], positions[:,1], c=norm, cmap="plasma", s=60, alpha=0.9)
    plt.colorbar(sc, ax=axes[1], label="Importance (normalised)")
    top_pos = positions[top20[:5]]
    axes[1].scatter(top_pos[:,0], top_pos[:,1], s=200, facecolors="none", edgecolors="#f4a261", linewidths=2.5, label="Top 5")
    axes[1].set_title("Electrode Importance Spatial Map", color="white"); axes[1].set_xlabel("X (um)", color="white"); axes[1].set_ylabel("Y (um)", color="white"); axes[1].legend(facecolor="#161b22", labelcolor="white", fontsize=8); axes[1].tick_params(colors="white")
    plt.tight_layout()
    plt.savefig("outputs/biocomputer_explainability_results.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("  Saved: outputs/biocomputer_explainability_results.png")

if __name__ == "__main__":
    main()
