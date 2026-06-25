# Biocomputer Explainability Framework

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![PhD](https://img.shields.io/badge/University%20of%20Essex-PhD%20Research-purple)](https://essex.ac.uk)
[![Conference](https://img.shields.io/badge/WCCI%20FUZZ--IEEE-2026-red)](https://attend.ieee.org/wcci-2026/)

A framework for applying Explainable AI (XAI) techniques to living neural biocomputers. Provides tools for interpreting what biological neural networks are computing, which electrodes are functionally important, and what rules govern their spike-based communication.

## Why Explainability Matters for Biocomputers

Living neural biocomputers are fundamentally different from silicon AI — they are biological systems that we do not fully understand. Explainability is not just a nice-to-have; it is scientifically essential:

1. **Scientific discovery**: Reveals *which* electrodes and *what* patterns drive computation
2. **Clinical trust**: Enables regulatory approval for medical applications (FDA/EU AI Act)
3. **Debugging**: When the system fails, we can identify the exact electrode and rule responsible
4. **Knowledge transfer**: Rules from one neural culture can be validated against another
5. **Closed-loop control**: Explainable rules enable the biocomputer to adjust its own stimulation

## Framework Components

### 1. Electrode Importance Analysis
```python
from biocomputer_xai import ElectrodeImportanceAnalyser

analyser = ElectrodeImportanceAnalyser()
importance_scores = analyser.compute(
    recordings=mea_data,
    method="annigma"  # or "shap", "permutation"
)
analyser.plot_spatial_map(importance_scores, chip_layout)
```

### 2. Rule Extraction & Visualisation
```python
from biocomputer_xai import FuzzyRuleExtractor

extractor = FuzzyRuleExtractor(n_terms=3)
rules = extractor.extract(X_train, y_train)

# Human-readable output
for rule in rules.top_k(10):
    print(rule.to_natural_language())
# "When electrode 13 shows medium activity AND electrode 17 shows medium activity,
#  the network is likely generating a spike (confidence: 100%)"
```

### 3. Counterfactual Explanations
```python
from biocomputer_xai import CounterfactualExplainer

explainer = CounterfactualExplainer(classifier)
cf = explainer.explain(
    sample=firing_rates,
    target_class="NO_SPIKE"
)
print(f"To change prediction: reduce electrode_17 activity by {cf.delta:.2f}")
```

### 4. Network Topology Analysis
```python
from biocomputer_xai import NetworkTopologyAnalyser

topo = NetworkTopologyAnalyser()
hubs = topo.find_hub_electrodes(connectivity_matrix)
communities = topo.detect_communities(connectivity_matrix)
```

## Installation

```bash
git clone https://github.com/Adham5172001/biocomputer-explainability.git
cd biocomputer-explainability
pip install -r requirements.txt
```

## Key Findings from PhD Research

- **Electrode 17** appears in 47% of spike-detection rules across all 6 chips tested, identifying it as a critical spatial hub
- Rules are **transferable** across biologically diverse chips with <3% accuracy loss
- The fuzzy rule base reveals **spatial clustering** of important electrodes, consistent with known cortical column organisation
- Explainable rules enable **closed-loop stimulation** — the system can identify when to stimulate based on current network state

## Related Work

- [mea-spike-detection-fuzzy](https://github.com/Adham5172001/mea-spike-detection-fuzzy) — The classifier this framework was built to explain
- Associated paper: *"A Fuzzy-Based Approach for Interpretable Spike Detection in Living Neural Biocomputers"*, WCCI FUZZ-IEEE 2026

## License

MIT License
