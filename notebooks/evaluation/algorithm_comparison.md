# Algorithm Comparison Table

## Performance

| Metric | Apriori | Fpgrowth | Closed |
| --- | --- | --- | --- |
| Itemset Mining Time (s) | 0.002471 | 0.001681 | N/A |
| Rule Generation Time (s) | 0.001416 | 0.001848 | N/A |
| Total Time (s) | 0.003888 | 0.003529 | N/A |

## Output Size

| Metric | Apriori | Fpgrowth | Closed |
| --- | --- | --- | --- |
| Number of Itemsets | 26 | 26 | 6.00 |
| Number of Rules | 29 | 29 | 26 |

## Rule Quality

| Metric | Apriori | Fpgrowth | Closed |
| --- | --- | --- | --- |
| Average Support | 0.2138 | 0.2138 | 0.2000 |
| Average Confidence | 0.9885 | 0.9885 | 1.00 |
| Average Lift | 3.48 | 3.48 | 5.00 |
| Ratio of Rules with Lift > 1 | 1.00 | 1.00 | 1.00 |
| Ratio of Rules with Lift > 3 | 0.4828 | 0.4828 | 1.00 |

## Rule Structure

| Metric | Apriori | Fpgrowth | Closed |
| --- | --- | --- | --- |
| Average Antecedent Size | N/A | N/A | 1.46 |
| Average Consequent Size | N/A | N/A | 1.46 |
| Average Rule Size | N/A | N/A | 2.92 |

## Rule Set Properties

| Metric | Apriori | Fpgrowth | Closed |
| --- | --- | --- | --- |
| Redundancy Ratio | 0.5862 | 0.5862 | 0.4615 |
| Rule Diversity | 0.4970 | 0.4970 | 0.6905 |

## Strengths and Weaknesses

### Apriori

**Strengths:**
- Simple to understand and implement
- Works well for small to medium sized datasets
- Generates all possible frequent itemsets

**Weaknesses:**
- Slow performance on large datasets due to multiple database scans
- Generates many redundant rules
- Memory intensive for large itemsets

### FP-Growth

**Strengths:**
- More efficient than Apriori, especially for large datasets
- Only scans the database twice
- Uses a compact data structure (FP-tree)

**Weaknesses:**
- More complex implementation
- Still generates redundant rules
- FP-tree construction can be memory intensive

### Closed Itemsets / Concept Lattice

**Strengths:**
- Provides a more concise representation of patterns
- Captures hierarchical relationships between itemsets
- Reduces redundancy in rules
- Enables formal concept analysis

**Weaknesses:**
- More complex theoretical foundation
- Can be computationally intensive for lattice construction
- Less widely implemented in standard libraries

## Recommendations

- **For small datasets**: Any method works well, with Apriori being simplest to implement and understand
- **For medium to large datasets**: FP-Growth offers better performance
- **For concise, high-quality rules**: Closed itemset mining with concept lattices reduces redundancy
- **For exploratory data analysis**: Concept lattices provide rich insights into hierarchical relationships
- **For production systems with speed requirements**: FP-Growth or optimized implementations of closed itemset mining
