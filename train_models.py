# run_models.py
import os
import json
import pandas as pd
import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

# Configuration
models1           = ["TransE", "DistMult"]
models2         = ["ComplEx", "RotatE"]
embedding_dim    = 128
num_negs_per_pos = 1
learning_rate    = 0.01
num_epochs       = 10
random_seed      = 2025

# Auto-select device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device!r}")

# Path to your triples file
triples_path = "query-result.tsv"

# Load your data
tf = TriplesFactory.from_path(triples_path)
train, test = tf.split([0.8, 0.2], random_state=random_seed)

# Run each model
records = []
for model_name in models1:
    print(f"\nTraining {model_name} (dim={embedding_dim}, negs={num_negs_per_pos}) …")
    result = pipeline(
        training=train,
        testing=test,
        model=model_name,
        model_kwargs=dict(embedding_dim=embedding_dim),
        negative_sampler_kwargs=dict(num_negs_per_pos=num_negs_per_pos),
        optimizer_kwargs=dict(lr=learning_rate),
        training_kwargs=dict(num_epochs=num_epochs),
        random_seed=random_seed,
        device=device,
    )

    #Record summary metrics for CSV
    records.append({
        "model":         model_name,
        "embedding_dim": embedding_dim,
        "num_negs":      num_negs_per_pos,
        "mrr":           float(result.get_metric("mean_reciprocal_rank")),
        "hits@10":       float(result.get_metric("hits_at_10")),
    })

    #Create a folder for this model
    model_dir = os.path.join("models", f"{model_name}-citations")
    os.makedirs(model_dir, exist_ok=True)

    #Save full PyKEEN result (weights, mappings, logs)
    result.save_to_directory(model_dir)
    print(f"Saved full model to `{model_dir}/`")

    #Dump the full metric_results dict
    evaluation = result.metric_results.to_dict()
    with open(os.path.join(model_dir, "evaluation_full.json"), "w") as f:
        json.dump(evaluation, f, indent=2)
    print(f"Saved full evaluation to `{model_dir}/evaluation_full.json`")



#Write out the summary CSV
df = pd.DataFrame.from_records(records)
out_file = "results.csv"
df.to_csv(out_file, index=False)
print(f"\nAll done — summary metrics in `{out_file}`")
