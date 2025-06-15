import json
import os
import pandas as pd

# Paths to your trained RotatE model results
model_paths = {
    "RotatE-citations": "models/RotatE-citations/results.json",
    "RotatE-emb64-negs1": "models/RotatE-emb64-negs1/results.json",
    "RotatE-emb128-negs5": "models/RotatE-emb128-negs5/results.json",
}

records = []

for model_name, path in model_paths.items():
    try:
        with open(path, "r") as f:
            data = json.load(f)
            realistic = data["metrics"]["both"]["realistic"]
            records.append({
                "Model": model_name,
                "Hits@10": realistic["hits_at_10"],
                "Hits@3": realistic["hits_at_3"],
                "Hits@1": realistic["hits_at_1"],
                "MRR": realistic["inverse_harmonic_mean_rank"],
                "Harmonic Rank": realistic["harmonic_mean_rank"]
            })
    except Exception as e:
        print(f"Error loading {path}: {e}")

# Convert to DataFrame and sort by MRR
df = pd.DataFrame.from_records(records)
df_sorted = df.sort_values(by="MRR", ascending=False)
print("\nRotatE Variant Comparison:")
print(df_sorted.to_string(index=False))

# Save to CSV if needed
df_sorted.to_csv("rotate_comparison.csv", index=False)
print("\nComparison saved to 'rotate_comparison.csv'")
