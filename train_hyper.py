# tune_rotate.py
import os
import json
import pandas as pd
import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

# Fixed params
model_name = "RotatE"
learning_rate = 0.01
num_epochs = 15
random_seed = 2025
triples_path = "query-result.tsv"

# Auto-select device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device!r}")

# Load triples
tf = TriplesFactory.from_path(triples_path)
train, test = tf.split([0.8, 0.2], random_state=random_seed)

# Parameter combinations (excluding the already tested 128/1)
configs = [
    {"embedding_dim": 64,  "num_negs_per_pos": 1},
    {"embedding_dim": 128, "num_negs_per_pos": 5},
]

# Store results
records = []

for config in configs:
    emb_dim = config["embedding_dim"]
    negs = config["num_negs_per_pos"]
    print(f"\nTraining RotatE with emb_dim={emb_dim}, num_negs={negs} …")

    result = pipeline(
        training=train,
        testing=test,
        model=model_name,
        model_kwargs=dict(embedding_dim=emb_dim),
        negative_sampler_kwargs=dict(num_negs_per_pos=negs),
        optimizer_kwargs=dict(lr=learning_rate),
        training_kwargs=dict(num_epochs=num_epochs),
        random_seed=random_seed,
        device=device,
    )

    # Record for CSV
    records.append({
        "model": model_name,
        "embedding_dim": emb_dim,
        "num_negs": negs,
        "mrr": float(result.get_metric("mean_reciprocal_rank")),
        "hits@10": float(result.get_metric("hits_at_10")),
    })

    # Save model output
    model_dir = os.path.join("models", f"{model_name}-emb{emb_dim}-negs{negs}")
    os.makedirs(model_dir, exist_ok=True)
    result.save_to_directory(model_dir)

    # Save detailed metrics
    with open(os.path.join(model_dir, "evaluation_full.json"), "w") as f:
        json.dump(result.metric_results.to_dict(), f, indent=2)

# Save all summary results
df = pd.DataFrame.from_records(records)
df.to_csv("tuned_results.csv", index=False)
print("\nTuning complete. Results saved to `tuned_results.csv`.")
