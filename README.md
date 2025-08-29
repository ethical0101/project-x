# Project X: Frequent Pattern Mining with Concept Lattice Analysis

This repository implements a comprehensive Frequent Pattern Mining project with a strong foundation in Discrete Mathematics, specifically leveraging concept lattices for advanced analysis.

## Overview

This project provides tools for analyzing transaction data using frequent pattern mining techniques and concept lattice theory. The pipeline includes data cleaning, transaction encoding, itemset mining using both Apriori and FP-Growth algorithms, concept lattice construction, and rule extraction.

## Features

- Data cleaning and normalization
- Transaction encoding
- Frequent itemset mining (Apriori and FP-Growth algorithms)
- Closed itemset computation and concept lattice construction
- Association rule generation and evaluation
- Comprehensive visualizations of itemsets, rules, and lattice structure
- Sensitivity analysis and algorithm comparison

## Getting Started

### Prerequisites

This project requires Python 3.8+ and the following libraries:
- mlxtend
- pandas
- networkx
- matplotlib
- seaborn
- numpy

### Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Project

1. Place your transaction data in `data/transactions.json` with the format:
```json
[
  {"transaction_id": 1, "items": ["apple", "milk", "bread"]},
  {"transaction_id": 2, "items": ["rice", "oil"]}
]
```

2. Create a normalization map in `data/normalization.json` for handling synonyms and variants:
```json
{
  "apple": "apple",
  "Apple": "apple",
  "apples": "apple"
}
```

3. Run the notebooks in the following order:

   a. **1_cleaning.ipynb**: Cleans and normalizes the raw transaction data
   b. **2_mining.ipynb**: Performs frequent pattern mining with Apriori and FP-Growth
   c. **3_concept_analysis.ipynb**: Builds concept lattice and analyzes formal concepts
   d. **4_additional_visualizations.ipynb**: Creates additional visualizations (10+ different types)

4. All outputs will be stored in the `output/` directory and visualizations in the `figures/` directory.

### Configuration

Parameters for the mining algorithms and visualizations can be adjusted in `config.json`:
- `min_support`: Minimum support threshold for itemsets (default: 0.2)
- `min_confidence`: Minimum confidence threshold for rules (default: 0.6)
- `min_lift`: Minimum lift threshold for rules (default: 1.0)
- `rare_item_threshold`: Threshold for filtering rare items (default: 0.05)

You can experiment with these parameters to see how they affect the results:
- Lower `min_support` will generate more itemsets but may be less reliable
- Higher `min_confidence` will produce stronger rules but fewer of them
- Higher `min_lift` will focus on more interesting relationships

## Project Structure

```
.
├── config.json           # Configuration parameters
├── data/
│   ├── transactions.json # Input transaction data
│   └── normalization.json # Normalization map
├── figures/              # Generated visualizations
├── notebooks/            # Jupyter notebooks
│   ├── 1_cleaning.ipynb  # Data cleaning and preprocessing
│   ├── 2_mining.ipynb    # Frequent pattern mining algorithms
│   ├── 3_concept_analysis.ipynb # Concept lattice analysis
│   └── 4_additional_visualizations.ipynb # Extended visualizations
├── output/               # Generated output data
│   ├── cleaned_transactions.csv # Cleaned transaction data
│   ├── apriori_itemsets.csv # Frequent itemsets from Apriori
│   ├── fpgrowth_itemsets.csv # Frequent itemsets from FP-Growth
│   ├── apriori_rules.csv # Association rules from Apriori
│   ├── fpgrowth_rules.csv # Association rules from FP-Growth
│   └── formal_concepts.json # Formal concepts from lattice analysis
├── README.md
└── requirements.txt
```

## Detailed Notebook Descriptions

### 1. Data Cleaning (1_cleaning.ipynb)
- Loading transaction data from JSON
- Normalizing item names using mapping
- Removing rare items
- Analyzing transaction patterns
- Saving cleaned data for mining

### 2. Mining (2_mining.ipynb)
- Implementing Apriori and FP-Growth algorithms
- Discovering frequent itemsets with configurable support threshold
- Generating association rules with confidence and lift metrics
- Comparing algorithm performance
- Visualizing support and confidence distributions

### 3. Concept Analysis (3_concept_analysis.ipynb)
- Building formal context from transaction data
- Computing the concept lattice structure
- Visualizing the concept hierarchy
- Analyzing implications and dependencies
- Connecting lattice theory to association rules

### 4. Additional Visualizations (4_additional_visualizations.ipynb)
- Creating 10+ different visualization types for deeper analysis
- Network graphs showing rule relationships
- Heat maps of item associations
- 3D visualizations of rule metrics
- Distribution charts of antecedents and consequents
- Advanced visualizations like radar charts and treemaps

## Discrete Math Foundations

This project demonstrates the connection between frequent pattern mining and formal concept analysis:

1. **Closure Operator**: Computing closed itemsets via the Galois connection between items and transactions
2. **Concept Lattice**: Building the Hasse diagram of the concept lattice using closed itemsets
3. **Minimal Generators**: Finding minimal generators for each closed itemset
4. **Concise Rules**: Extracting non-redundant association rules from the lattice structure

## Results and Visualizations

The pipeline produces:

### Data Files
- Cleaned transaction data in CSV format
- Frequent itemsets from Apriori and FP-Growth algorithms
- Association rules with various metrics (support, confidence, lift, etc.)
- Formal concepts and lattice structure

### Visualizations
At least 10 different types of visualizations are generated, including:

1. **Association Rule Network** - A network graph showing relationships between items
2. **Lift Matrix Heatmap** - A heatmap showing lift values between different items
3. **3D Rule Visualization** - A 3D scatter plot of support, confidence, and lift
4. **Bubble Chart** - A bubble chart showing rule metrics with sized bubbles
5. **Parallel Coordinates Plot** - A visualization showing how metrics relate across rules
6. **Radar Chart** - A spider chart comparing different metrics for top rules
7. **Antecedent Pie Chart** - A pie chart showing distribution of antecedent items
8. **Confidence Treemap** - A treemap visualization of rule confidence
9. **Support vs. Lift Scatter** - A scatter plot with multiple dimensions encoded
10. **Consequent Bar Chart** - A horizontal bar chart of consequent item distribution
11. **Concept Lattice Graph** - A hierarchical visualization of the concept lattice
12. **Algorithm Performance Comparison** - Charts comparing Apriori vs. FP-Growth performance

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Authors

- **Your Name** - *Initial work* - [YourGitHubUsername](https://github.com/YourGitHubUsername)

## Acknowledgments

- Hat tip to anyone whose code was used
- Inspiration from formal concept analysis and association rule mining literature
- Special thanks to the maintainers of mlxtend, networkx, and other libraries used
