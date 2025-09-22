# Test script to verify evaluation notebook functionality
import sys
import os
sys.path.append('.')

# Add the project root to the path
import pandas as pd
import json
from pathlib import Path

def test_evaluation_notebook():
    """Test if the evaluation notebook can work with real data."""

    # Check if required files exist
    required_files = [
        'output/transaction_lists.json',
        'output/apriori_itemsets.csv',
        'output/fpgrowth_itemsets.csv',
        'config.json'
    ]

    print("=== File Existence Check ===")
    for file_path in required_files:
        exists = os.path.exists(file_path)
        print(f"{file_path}: {'✓' if exists else '✗'}")

        if exists and file_path.endswith('.json'):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        print(f"  Contains {len(data)} items")
                    elif isinstance(data, dict):
                        print(f"  Contains {len(data)} keys: {list(data.keys())}")
            except Exception as e:
                print(f"  Error reading {file_path}: {e}")

        elif exists and file_path.endswith('.csv'):
            try:
                df = pd.read_csv(file_path)
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {list(df.columns)}")
            except Exception as e:
                print(f"  Error reading {file_path}: {e}")

    print("\n=== Transaction Data Analysis ===")
    if os.path.exists('output/transaction_lists.json'):
        with open('output/transaction_lists.json', 'r') as f:
            transactions = json.load(f)
        print(f"Number of transactions: {len(transactions)}")
        print(f"Sample transactions: {transactions[:3]}")

        # Calculate unique items
        all_items = set()
        for transaction in transactions:
            all_items.update(transaction)
        print(f"Unique items: {len(all_items)}")
        print(f"Items: {sorted(list(all_items))}")

        # Calculate item frequencies
        item_freq = {}
        for transaction in transactions:
            for item in transaction:
                item_freq[item] = item_freq.get(item, 0) + 1

        print("\nItem frequencies:")
        for item, freq in sorted(item_freq.items(), key=lambda x: x[1], reverse=True):
            support = freq / len(transactions)
            print(f"  {item}: {freq}/{len(transactions)} = {support:.3f}")

    print("\n=== Rule Generation Test ===")
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder

        # Load and encode transactions
        with open('output/transaction_lists.json', 'r') as f:
            transactions = json.load(f)

        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

        print(f"Encoded data shape: {df_encoded.shape}")

        # Test different support thresholds
        for min_support in [0.2, 0.15, 0.1, 0.05]:
            frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
            print(f"Min support {min_support}: {len(frequent_itemsets)} frequent itemsets")

            if len(frequent_itemsets) > 1:
                # Try to generate rules
                for min_confidence in [0.1, 0.05, 0.01]:
                    try:
                        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
                        if len(rules) > 0:
                            print(f"  ✓ Generated {len(rules)} rules with confidence >= {min_confidence}")
                            print(f"    Sample rule: {rules.iloc[0]['antecedents']} -> {rules.iloc[0]['consequents']} (conf: {rules.iloc[0]['confidence']:.3f}, lift: {rules.iloc[0]['lift']:.3f})")
                            return True
                        else:
                            print(f"    No rules with confidence >= {min_confidence}")
                    except Exception as e:
                        print(f"    Error generating rules with confidence {min_confidence}: {e}")

        print("❌ Could not generate any association rules")
        return False

    except Exception as e:
        print(f"❌ Error in rule generation test: {e}")
        return False

if __name__ == "__main__":
    test_evaluation_notebook()
