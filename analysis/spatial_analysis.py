"""
Spatial Analysis of Electrode Importance
Author: Adham Aboulkheir | University of Essex | PhD Research
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SpatialCluster:
    cluster_id: int
    centre: Tuple[float, float]
    electrode_ids: List[int]
    mean_importance: float
    radius: float


def compute_spatial_autocorrelation(importance_scores: np.ndarray,
                                     positions: np.ndarray) -> float:
    """
    Compute Moran's I spatial autocorrelation for electrode importance.
    Positive I = spatially clustered, Negative I = dispersed.
    """
    n = len(importance_scores)
    mean_imp = importance_scores.mean()
    deviations = importance_scores - mean_imp

    # Compute spatial weight matrix (inverse distance)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(positions[i] - positions[j])
                W[i, j] = 1.0 / (dist + 1e-9)

    # Row-normalise
    row_sums = W.sum(axis=1, keepdims=True)
    W = W / (row_sums + 1e-9)

    # Moran's I
    numerator = n * np.sum(W * np.outer(deviations, deviations))
    denominator = np.sum(W) * np.sum(deviations ** 2)

    return float(numerator / (denominator + 1e-9))


def find_hub_electrodes(importance_scores: np.ndarray,
                         positions: np.ndarray,
                         threshold_percentile: float = 80) -> List[int]:
    """
    Identify hub electrodes — those with both high importance and
    spatial centrality (close to other important electrodes).
    """
    threshold = np.percentile(importance_scores, threshold_percentile)
    high_importance = np.where(importance_scores > threshold)[0]

    if len(high_importance) == 0:
        return []

    # Compute centrality within high-importance group
    hub_positions = positions[high_importance]
    centralities = []
    for i, pos in enumerate(hub_positions):
        dists = np.linalg.norm(hub_positions - pos, axis=1)
        centrality = 1.0 / (dists[dists > 0].mean() + 1e-9)
        centralities.append(centrality)

    # Return electrodes sorted by combined importance + centrality
    combined = importance_scores[high_importance] * np.array(centralities)
    sorted_idx = np.argsort(combined)[::-1]
    return list(high_importance[sorted_idx])


if __name__ == "__main__":
    print("Spatial Analysis Demo")
    print("=" * 40)

    np.random.seed(42)
    n_electrodes = 64
    positions = np.array([[i * 200, j * 200] for i in range(8) for j in range(8)])
    importance = np.random.exponential(0.5, n_electrodes)
    importance[:10] *= 3  # Make first 10 electrodes more important

    morans_i = compute_spatial_autocorrelation(importance, positions)
    print(f"Moran's I: {morans_i:.4f} ({'clustered' if morans_i > 0 else 'dispersed'})")

    hubs = find_hub_electrodes(importance, positions, threshold_percentile=80)
    print(f"Hub electrodes: {[f'E{h+1}' for h in hubs[:5]]}")
    print(f"Mean importance of hubs: {importance[hubs[:5]].mean():.4f}")
    print(f"Mean importance overall: {importance.mean():.4f}")
