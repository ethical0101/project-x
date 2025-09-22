# Project X: Step-by-Step Research & Presentation Documentation

## Overview
Project X implements frequent pattern mining and concept lattice analysis on transaction data. The workflow is designed for reproducible research and clear presentation.

---

## 1. Environment Setup
- **Python Version:** 3.13+
- **Virtual Environment:**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate
  pip install -r requirements.txt
  ```
- **Dependencies:**
  - mlxtend, pandas, networkx, matplotlib, seaborn, numpy, scikit-learn, jupyterlab, ipywidgets, pytest

---

## 2. Data Preparation
- **Input Files:**
  - `data/transactions.json`: Raw transaction data
  - `data/normalization.json`: Item normalization map
- **Configuration:**
  - `config.json`: Set parameters like `min_support`, `min_confidence`, `rare_item_threshold`

---

## 3. Data Cleaning (`notebooks/1_cleaning.ipynb`)
- **Purpose:** Cleans and normalizes transaction data, removes rare items.
- **How to Run:**
  - Open notebook and run all cells sequentially.
- **Outputs:**
  - `output/cleaned_transactions.json`, `output/rare_items.json`, `output/transaction_lists.json`, `output/cleaned_transactions.csv`
- **Research Notes:**
  - Ensures data quality and consistency for mining.

---

## 4. Frequent Pattern Mining (`notebooks/2_mining.ipynb`)
- **Purpose:** Finds frequent itemsets and association rules using Apriori and FP-Growth.
- **How to Run:**
  - Open notebook and run all cells sequentially.
- **Outputs:**
  - `output/apriori_itemsets.csv`, `output/fpgrowth_itemsets.csv`, `output/apriori_rules.csv`, `output/fpgrowth_rules.csv`
- **Research Notes:**
  - Compare algorithm performance and rule quality.
  - Visualize support, confidence, lift, and other metrics.

---

## 5. Concept Lattice Analysis (`notebooks/3_concept_analysis.ipynb`)
- **Purpose:** Builds formal context and concept lattice, analyzes implications.
- **How to Run:**
  - Open notebook and run all cells sequentially.
- **Outputs:**
  - `output/formal_context.csv`, `output/formal_concepts.json`, `output/attribute_implications.json`
- **Research Notes:**
  - Connects discrete math theory to association rule mining.
  - Visualizes concept hierarchy and dependencies.

---

## 6. Lattice Construction & Visualization (`notebooks/3_lattice.ipynb`)
- **Purpose:** Advanced lattice construction and visualization.
- **How to Run:**
  - Open notebook and run all cells sequentially.
- **Outputs:**
  - `figures/lattice.png`, other lattice-related figures
- **Research Notes:**
  - Explore structure and relationships in mined data.

---

## 7. Advanced Visualizations (`notebooks/4_additional_visualizations.ipynb`)
- **Purpose:** Generates 10+ advanced visualizations for rules, itemsets, and lattice.
- **How to Run:**
  - Open notebook and run all cells sequentially.
- **Outputs:**
  - Various images in `figures/` (network graphs, heatmaps, 3D plots, radar charts, etc.)
- **Presentation Notes:**
  - Use these for deeper insights and impactful presentations.

---

## 8. Evaluation & Comparison (`notebooks/4_evaluation.ipynb`)
- **Purpose:** Compares traditional algorithms and concept lattice approach, evaluates rule quality.
- **How to Run:**
  - Open notebook and run all cells sequentially.
- **Outputs:**
  - Evaluation metrics, comparison charts, summary tables
- **Research Notes:**
  - Discuss strengths, weaknesses, and practical implications.

---

## 9. Output Files & Visualizations
- **Data:** All processed data in `output/`
- **Figures:** All visualizations in `figures/`

---

## 10. Presentation & Research Tips
- **For Research:**
  - Document parameter choices and their impact.
  - Compare results across algorithms and visualizations.
  - Discuss theoretical connections and practical findings.
- **For Presentation:**
  - Use visualizations to tell a story.
  - Highlight key findings and insights.
  - Structure slides according to workflow steps.

---

## 11. Troubleshooting
- If a notebook cell fails, check for missing dependencies or data files.
- Ensure all previous steps are completed before running dependent notebooks.
- Review error messages for guidance.

---

## 12. References & Further Reading
- See README.md for background, references, and acknowledgments.
- Explore formal concept analysis and association rule mining literature for deeper understanding.

---

## 13. License & Contribution
- MIT License (see LICENSE)
- Contributions welcome via pull requests.

---

*Prepared for research and presentation purposes. For questions or improvements, contact the project maintainer.*
