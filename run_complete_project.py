#!/usr/bin/env python3
"""
Project X: Complete Frequent Pattern Mining & Concept Lattice Analysis
All-in-One Script for Faculty Presentation

This script combines all notebooks and runs the complete workflow:
1. Data Cleaning & Normalization
2. Frequent Pattern Mining (Apriori & FP-Growth)
3. Concept Lattice Analysis
4. Advanced Visualizations
5. Performance Evaluation

Just run this file and everything will execute automatically!
"""

import os
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

print("="*80)
print("PROJECT X: FREQUENT PATTERN MINING & CONCEPT LATTICE ANALYSIS")
print("="*80)
print("Starting complete workflow...")
print()

# Create necessary directories
os.makedirs('output', exist_ok=True)
os.makedirs('figures', exist_ok=True)
os.makedirs('evaluation', exist_ok=True)

# ============================================================================
# STEP 1: DATA CLEANING & NORMALIZATION
# ============================================================================
print("STEP 1: DATA CLEANING & NORMALIZATION")
print("-" * 50)

def load_json_file(file_path: str) -> Any:
    """Load data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def save_json_file(data: Any, file_path: str) -> None:
    """Save data to a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def normalize_item_name(item: str, normalization_map: Dict[str, str]) -> str:
    """Normalize item name using the normalization map."""
    return normalization_map.get(item.lower(), item.lower())

def clean_transaction_data(transactions: List[Dict], normalization_map: Dict[str, str]) -> List[Dict]:
    """Clean and normalize transaction data."""
    cleaned_transactions = []

    for transaction in transactions:
        transaction_id = transaction.get('transaction_id', len(cleaned_transactions) + 1)
        items = transaction.get('items', [])

        # Normalize item names
        normalized_items = []
        for item in items:
            if isinstance(item, str) and item.strip():
                normalized_item = normalize_item_name(item.strip(), normalization_map)
                if normalized_item:
                    normalized_items.append(normalized_item)

        # Remove duplicates while preserving order
        unique_items = []
        seen = set()
        for item in normalized_items:
            if item not in seen:
                unique_items.append(item)
                seen.add(item)

        if unique_items:
            cleaned_transactions.append({
                'transaction_id': transaction_id,
                'items': unique_items
            })

    return cleaned_transactions

# Load or create sample data
transaction_file_path = 'data/transactions.json'
normalization_file_path = 'data/normalization.json'

# Load transactions
transactions = load_json_file(transaction_file_path)
if transactions is None:
    print("Creating sample transaction data...")
    transactions = [
        {"transaction_id": 1, "items": ["Apple", "Milk", "Bread"]},
        {"transaction_id": 2, "items": ["Rice", "Oil"]},
        {"transaction_id": 3, "items": ["milk", "Eggs", "Cheese"]},
        {"transaction_id": 4, "items": ["Bread", "Butter", "Milk"]},
        {"transaction_id": 5, "items": ["apple", "Banana", "Orange"]},
        {"transaction_id": 6, "items": ["Tea", "Sugar", "Milk"]},
        {"transaction_id": 7, "items": ["Bread", "Jam", "Butter"]},
        {"transaction_id": 8, "items": ["Apple", "Orange", "Banana"]},
        {"transaction_id": 9, "items": ["Rice", "Dal", "Oil"]},
        {"transaction_id": 10, "items": ["Milk", "Cheese", "Yogurt"]}
    ]
    save_json_file(transactions, transaction_file_path)

print(f"Loaded {len(transactions)} transactions")

# Load normalization map
normalization_map = load_json_file(normalization_file_path)
if normalization_map is None:
    print("Creating sample normalization map...")
    normalization_map = {
        "apple": "apple", "Apple": "apple", "apples": "apple",
        "milk": "milk", "Milk": "milk", "MILK": "milk",
        "bread": "bread", "Bread": "bread", "breads": "bread",
        "rice": "rice", "Rice": "rice", "RICE": "rice",
        "oil": "oil", "Oil": "oil", "cooking oil": "oil",
        "eggs": "eggs", "Eggs": "eggs", "egg": "eggs",
        "cheese": "cheese", "Cheese": "cheese",
        "butter": "butter", "Butter": "butter",
        "orange": "orange", "Orange": "orange", "oranges": "orange",
        "banana": "banana", "Banana": "banana", "bananas": "banana",
        "tea": "tea", "Tea": "tea", "TEA": "tea",
        "sugar": "sugar", "Sugar": "sugar",
        "jam": "jam", "Jam": "jam",
        "dal": "dal", "Dal": "dal", "lentils": "dal",
        "yogurt": "yogurt", "Yogurt": "yogurt", "curd": "yogurt"
    }
    save_json_file(normalization_map, normalization_file_path)

print(f"Loaded normalization map with {len(normalization_map)} entries")

# Clean transactions
cleaned_transactions = clean_transaction_data(transactions, normalization_map)
print(f"After cleaning: {len(cleaned_transactions)} transactions")

# Calculate item frequencies
all_items = []
for transaction in cleaned_transactions:
    all_items.extend(transaction['items'])

item_counts = defaultdict(int)
for item in all_items:
    item_counts[item] += 1

total_transactions = len(cleaned_transactions)
item_frequencies = {item: count / total_transactions for item, count in item_counts.items()}

print(f"Number of unique items: {len(item_frequencies)}")

# Filter rare items (frequency < 5%)
rare_threshold = 0.05
rare_items = [item for item, freq in item_frequencies.items() if freq < rare_threshold]
common_items = [item for item, freq in item_frequencies.items() if freq >= rare_threshold]

print(f"Rare items (frequency < {rare_threshold}): {len(rare_items)}")
print(f"Common items: {len(common_items)}")

# Remove rare items from transactions
filtered_transactions = []
for transaction in cleaned_transactions:
    filtered_items = [item for item in transaction['items'] if item in common_items]
    if filtered_items:
        filtered_transactions.append({
            'transaction_id': transaction['transaction_id'],
            'items': filtered_items
        })

print(f"After filtering rare items: {len(filtered_transactions)} transactions")

# Save cleaned data
transaction_lists = [t['items'] for t in filtered_transactions]
save_json_file(filtered_transactions, 'output/cleaned_transactions.json')
save_json_file(rare_items, 'output/rare_items.json')
save_json_file(transaction_lists, 'output/transaction_lists.json')

# Create CSV format for analysis
rows = []
for transaction in filtered_transactions:
    for item in transaction['items']:
        rows.append({'transaction_id': transaction['transaction_id'], 'item': item})

df_transactions = pd.DataFrame(rows)
df_transactions.to_csv('output/cleaned_transactions.csv', index=False)

print("✓ Data cleaning completed!")
print(f"  - Saved to: output/cleaned_transactions.json, output/transaction_lists.json")
print(f"  - CSV format: output/cleaned_transactions.csv")
print()

# ============================================================================
# STEP 2: FREQUENT PATTERN MINING
# ============================================================================
print("STEP 2: FREQUENT PATTERN MINING")
print("-" * 50)

# Configuration
config = {
    'min_support': 0.2,
    'min_confidence': 0.6,
    'min_lift': 1.0
}

min_support = config['min_support']
min_confidence = config['min_confidence']

print(f"Mining parameters:")
print(f"  Min Support: {min_support}")
print(f"  Min Confidence: {min_confidence}")

# Encode transactions
te = TransactionEncoder()
te_ary = te.fit_transform(transaction_lists)
encoded_df = pd.DataFrame(te_ary, columns=te.columns_)

print(f"Encoded {len(transaction_lists)} transactions with {len(te.columns_)} unique items")
encoded_df.to_csv('output/encoded_transactions.csv', index=False)

def frozenset_to_list(df: pd.DataFrame) -> pd.DataFrame:
    """Convert frozenset columns in a DataFrame to lists."""
    result = df.copy()
    for col in result.columns:
        if result[col].apply(lambda x: isinstance(x, frozenset)).any():
            result[col] = result[col].apply(lambda x: list(x) if isinstance(x, frozenset) else x)
    return result

# APRIORI ALGORITHM
print("\nRunning Apriori algorithm...")
start_time = time.time()
apriori_itemsets = apriori(encoded_df, min_support=min_support, use_colnames=True)
apriori_time = time.time() - start_time

print(f"Apriori completed in {apriori_time:.4f} seconds")
print(f"Found {len(apriori_itemsets)} frequent itemsets")

apriori_itemsets_list = frozenset_to_list(apriori_itemsets)
apriori_itemsets_list.to_csv('output/apriori_itemsets.csv', index=False)

# Generate association rules from Apriori
if len(apriori_itemsets) > 0:
    apriori_rules = association_rules(apriori_itemsets, metric="confidence", min_threshold=min_confidence)
    apriori_rules_list = frozenset_to_list(apriori_rules)
    print(f"Generated {len(apriori_rules_list)} association rules from Apriori")
    apriori_rules_list.to_csv('output/apriori_rules.csv', index=False)

# FP-GROWTH ALGORITHM
print("\nRunning FP-Growth algorithm...")
start_time = time.time()
fpgrowth_itemsets = fpgrowth(encoded_df, min_support=min_support, use_colnames=True)
fpgrowth_time = time.time() - start_time

print(f"FP-Growth completed in {fpgrowth_time:.4f} seconds")
print(f"Found {len(fpgrowth_itemsets)} frequent itemsets")

fpgrowth_itemsets_list = frozenset_to_list(fpgrowth_itemsets)
fpgrowth_itemsets_list.to_csv('output/fpgrowth_itemsets.csv', index=False)

# Generate association rules from FP-Growth
if len(fpgrowth_itemsets) > 0:
    fpgrowth_rules = association_rules(fpgrowth_itemsets, metric="confidence", min_threshold=min_confidence)
    fpgrowth_rules_list = frozenset_to_list(fpgrowth_rules)
    print(f"Generated {len(fpgrowth_rules_list)} association rules from FP-Growth")
    fpgrowth_rules_list.to_csv('output/fpgrowth_rules.csv', index=False)

# Performance comparison
print(f"\nAlgorithm Comparison:")
print(f"  Apriori: {len(apriori_itemsets)} itemsets, {apriori_time:.4f}s")
print(f"  FP-Growth: {len(fpgrowth_itemsets)} itemsets, {fpgrowth_time:.4f}s")
print(f"  Speedup: {apriori_time/fpgrowth_time:.2f}x" if fpgrowth_time > 0 else "  Speedup: N/A")

print("✓ Frequent pattern mining completed!")
print()

# ============================================================================
# STEP 3: CONCEPT LATTICE ANALYSIS
# ============================================================================
print("STEP 3: CONCEPT LATTICE ANALYSIS")
print("-" * 50)

def create_formal_context(itemsets: List[List[str]]) -> pd.DataFrame:
    """Create a formal context from itemsets."""
    # Get all unique items
    all_items = set()
    for itemset in itemsets:
        all_items.update(itemset)

    all_items = sorted(list(all_items))

    # Create binary matrix
    context = []
    for itemset in itemsets:
        row = [item in itemset for item in all_items]
        context.append(row)

    return pd.DataFrame(context, columns=all_items)

def generate_concepts(context: pd.DataFrame) -> List[Tuple[Set[int], Set[str]]]:
    """Generate formal concepts from a formal context."""
    concepts = []
    n_objects, n_attributes = context.shape

    # Get all possible attribute combinations
    attributes = list(context.columns)

    for i in range(len(attributes) + 1):
        from itertools import combinations
        for attr_combo in combinations(attributes, i):
            intent = set(attr_combo)

            # Find extent (objects that have all attributes in intent)
            if len(intent) == 0:
                extent = set(range(n_objects))
            else:
                extent = set(range(n_objects))
                for attr in intent:
                    objects_with_attr = set(context[context[attr] == True].index)
                    extent = extent.intersection(objects_with_attr)

            # Check if this is a valid concept (closed set)
            if len(extent) > 0:
                # Find all attributes shared by all objects in extent
                derived_intent = set()
                for attr in attributes:
                    if all(context.loc[obj, attr] for obj in extent):
                        derived_intent.add(attr)

                # If intent equals derived intent, it's a formal concept
                if intent == derived_intent:
                    concepts.append((extent, intent))

    return concepts

# Load frequent itemsets for concept analysis
itemsets_for_concepts = []
if len(fpgrowth_itemsets) > 0:
    for _, row in fpgrowth_itemsets.iterrows():
        itemset = list(row['itemsets'])
        itemsets_for_concepts.append(itemset)
else:
    # Use individual items if no frequent itemsets
    itemsets_for_concepts = [[item] for item in te.columns_]

print(f"Using {len(itemsets_for_concepts)} itemsets for concept analysis")

# Create formal context
context = create_formal_context(itemsets_for_concepts)
print(f"Created formal context with shape: {context.shape}")

context.to_csv('output/formal_context.csv', index=False)

# Generate formal concepts
concepts = generate_concepts(context)
print(f"Generated {len(concepts)} formal concepts")

# Save concepts
concepts_json = [
    {
        "extent": sorted(list(extent)),
        "intent": sorted(list(intent))
    }
    for extent, intent in concepts
]

save_json_file(concepts_json, 'output/formal_concepts.json')

print("✓ Concept lattice analysis completed!")
print()

# ============================================================================
# STEP 4: ADVANCED VISUALIZATIONS
# ============================================================================
print("STEP 4: ADVANCED VISUALIZATIONS")
print("-" * 50)

def create_visualizations():
    """Create various visualizations for the analysis."""

    # 1. Support Distribution
    if len(apriori_itemsets) > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(apriori_itemsets['support'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Support')
        plt.ylabel('Frequency')
        plt.title('Distribution of Itemset Support Values')
        plt.grid(True, alpha=0.3)
        plt.savefig('figures/support_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Support distribution plot saved")

    # 2. Rule Metrics Scatter Plot
    if len(apriori_rules_list) > 0:
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(apriori_rules_list['support'], apriori_rules_list['confidence'],
                            c=apriori_rules_list['lift'], cmap='viridis', s=50, alpha=0.7)
        plt.xlabel('Support')
        plt.ylabel('Confidence')
        plt.title('Association Rules: Support vs Confidence (colored by Lift)')
        plt.colorbar(scatter, label='Lift')
        plt.grid(True, alpha=0.3)
        plt.savefig('figures/rules_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Rules scatter plot saved")

    # 3. Item Frequency Bar Chart
    plt.figure(figsize=(12, 6))
    items = list(item_frequencies.keys())
    frequencies = list(item_frequencies.values())

    plt.bar(items, frequencies, color='lightcoral', alpha=0.8)
    plt.xlabel('Items')
    plt.ylabel('Frequency')
    plt.title('Item Frequency Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/item_frequency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Item frequency plot saved")

    # 4. Algorithm Performance Comparison
    algorithms = ['Apriori', 'FP-Growth']
    times = [apriori_time, fpgrowth_time]
    itemset_counts = [len(apriori_itemsets), len(fpgrowth_itemsets)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Execution times
    ax1.bar(algorithms, times, color=['skyblue', 'lightgreen'], alpha=0.8)
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Algorithm Execution Times')
    ax1.grid(True, alpha=0.3)

    # Itemset counts
    ax2.bar(algorithms, itemset_counts, color=['orange', 'pink'], alpha=0.8)
    ax2.set_ylabel('Number of Itemsets')
    ax2.set_title('Frequent Itemsets Found')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Algorithm comparison plot saved")

create_visualizations()
print("✓ Advanced visualizations completed!")
print()

# ============================================================================
# STEP 5: PERFORMANCE EVALUATION & SUMMARY
# ============================================================================
print("STEP 5: PERFORMANCE EVALUATION & SUMMARY")
print("-" * 50)

def generate_summary_report():
    """Generate a comprehensive summary report."""

    report = {
        "project_title": "Frequent Pattern Mining & Concept Lattice Analysis",
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_summary": {
            "total_transactions": len(filtered_transactions),
            "unique_items": len(te.columns_),
            "rare_items_filtered": len(rare_items),
            "sparsity": 1 - (sum(len(t) for t in transaction_lists) / (len(transaction_lists) * len(te.columns_)))
        },
        "mining_results": {
            "apriori": {
                "itemsets": len(apriori_itemsets),
                "rules": len(apriori_rules_list) if len(apriori_itemsets) > 0 else 0,
                "execution_time": apriori_time
            },
            "fpgrowth": {
                "itemsets": len(fpgrowth_itemsets),
                "rules": len(fpgrowth_rules_list) if len(fpgrowth_itemsets) > 0 else 0,
                "execution_time": fpgrowth_time
            }
        },
        "concept_analysis": {
            "formal_concepts": len(concepts),
            "context_size": f"{context.shape[0]}x{context.shape[1]}"
        },
        "configuration": config,
        "output_files": [
            "output/cleaned_transactions.json",
            "output/transaction_lists.json",
            "output/apriori_itemsets.csv",
            "output/fpgrowth_itemsets.csv",
            "output/apriori_rules.csv",
            "output/fpgrowth_rules.csv",
            "output/formal_context.csv",
            "output/formal_concepts.json"
        ],
        "visualizations": [
            "figures/support_distribution.png",
            "figures/rules_scatter.png",
            "figures/item_frequency.png",
            "figures/algorithm_comparison.png"
        ]
    }

    save_json_file(report, 'output/project_summary.json')

    # Print summary
    print("PROJECT SUMMARY:")
    print(f"  • {report['data_summary']['total_transactions']} transactions processed")
    print(f"  • {report['data_summary']['unique_items']} unique items identified")
    print(f"  • {report['mining_results']['apriori']['itemsets']} frequent itemsets (Apriori)")
    print(f"  • {report['mining_results']['fpgrowth']['itemsets']} frequent itemsets (FP-Growth)")
    print(f"  • {report['concept_analysis']['formal_concepts']} formal concepts generated")
    print(f"  • {len(report['visualizations'])} visualizations created")

    return report

summary = generate_summary_report()

# Performance metrics
print(f"\nPERFORMANCE METRICS:")
print(f"  • Apriori execution time: {apriori_time:.4f} seconds")
print(f"  • FP-Growth execution time: {fpgrowth_time:.4f} seconds")
print(f"  • Speed improvement: {apriori_time/fpgrowth_time:.2f}x" if fpgrowth_time > 0 else "  • Speed improvement: N/A")

print("✓ Performance evaluation completed!")
print()

# ============================================================================
# PRESENTATION SUMMARY
# ============================================================================
print("="*80)
print("PRESENTATION SUMMARY FOR FACULTY")
print("="*80)

print("""
🎯 PROJECT OVERVIEW:
   This project demonstrates Frequent Pattern Mining and Concept Lattice Analysis
   for discovering hidden patterns in transaction data.

📊 KEY RESULTS:
   ✓ Successfully processed {total_transactions} transactions
   ✓ Identified {unique_items} unique items
   ✓ Discovered {itemsets} frequent itemsets using both Apriori and FP-Growth
   ✓ Generated {rules} association rules with confidence ≥ {min_confidence}
   ✓ Built concept lattice with {concepts} formal concepts
   ✓ Created {visualizations} different visualizations

⚡ ALGORITHM PERFORMANCE:
   • Apriori Algorithm: {apriori_time:.4f} seconds
   • FP-Growth Algorithm: {fpgrowth_time:.4f} seconds
   • Performance Gain: {speedup:.2f}x faster with FP-Growth

🔍 RESEARCH CONTRIBUTIONS:
   1. Comparative analysis of mining algorithms
   2. Integration of concept lattice theory
   3. Comprehensive visualization framework
   4. Automated end-to-end pipeline

📁 OUTPUT FILES GENERATED:
   • All results saved in 'output/' directory
   • Visualizations saved in 'figures/' directory
   • Complete project summary in 'output/project_summary.json'

🚀 TECHNICAL HIGHLIGHTS:
   • Data cleaning and normalization
   • Multiple mining algorithms comparison
   • Formal concept analysis
   • Advanced data visualizations
   • Performance benchmarking
""".format(
    total_transactions=summary['data_summary']['total_transactions'],
    unique_items=summary['data_summary']['unique_items'],
    itemsets=summary['mining_results']['apriori']['itemsets'],
    rules=summary['mining_results']['apriori']['rules'],
    concepts=summary['concept_analysis']['formal_concepts'],
    visualizations=len(summary['visualizations']),
    min_confidence=config['min_confidence'],
    apriori_time=apriori_time,
    fpgrowth_time=fpgrowth_time,
    speedup=apriori_time/fpgrowth_time if fpgrowth_time > 0 else 1
))

print("="*80)
print("✅ PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
print("✅ All files generated and ready for presentation!")
print("="*80)

print(f"\n📍 Next Steps for Presentation:")
print(f"   1. Open 'figures/' folder to see all visualizations")
print(f"   2. Check 'output/' folder for all data files")
print(f"   3. Use 'output/project_summary.json' for detailed metrics")
print(f"   4. Show algorithm comparison charts for performance analysis")
print(f"   5. Demonstrate concept lattice results for theoretical depth")

# Final timing
total_time = time.time() - start_time if 'start_time' in locals() else 0
print(f"\n⏱️  Total execution time: {total_time:.2f} seconds")
print(f"🎉 Ready for your faculty presentation!")
