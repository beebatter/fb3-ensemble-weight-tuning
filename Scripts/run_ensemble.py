import yaml, pandas as pd, os

# 1. read best weights
best = yaml.safe_load(open("configs/best_weights.yaml"))

# 2. read test data and ensemble
models = list(best.keys())
targets = ['cohesion','syntax','vocabulary','phraseology','grammar','conventions']
dfs = [pd.read_csv(f"data/{m}_test.csv") for m in models]

final = dfs[0].copy()
final[targets] = sum(best[m] * df[targets] for m, df in zip(models, dfs))

os.makedirs("results", exist_ok=True)
final.to_csv("results/final_submission.csv", index=False)
print("Saved final submission to results/final_submission.csv")