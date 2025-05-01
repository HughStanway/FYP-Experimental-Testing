# pylint: skip-file

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from extract_accuracy_scores import get_accuracy_scores

filepath = "metrics/accuracy/GCJ2-4_py"
data = get_accuracy_scores(filepath)

sns.set_theme(style="ticks")
sns.set_context("notebook", font_scale=1.2) 

x_points = [10, 50, 100, 300]

def tidy_nested_data(data):
    records = []
    for db, models in data.items():
        for model, metrics in models.items():
            method = "jina-v2-base-code" if model == "ordis-jina-embeddings-v2-base-code" else model
            for metric, values in metrics.items():
                if metric == "MAP":  # Only store MAP values
                    for i, value in enumerate(values):
                        records.append({"x": x_points[i], "database": db, "method": method, "metric": metric, "value": value})
    return pd.DataFrame(records)

df = tidy_nested_data(data)

# Set order for methods
method_order = ["voyage-code-3", "jina-v2-base-code", "deepseek-r1:1.5B", "llama3.2"]
df = df.sort_values(by="method", key=lambda x: x.map({m: i for i, m in enumerate(method_order)}))

# Create 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
axes = axes.flatten()
db_names = ["chroma", "qdrant", "weaviate", "milvus"]

for i, db in enumerate(db_names):
    ax = axes[i]
    sns.lineplot(ax=ax, data=df[df["database"] == db], x="x", y="value", hue="method", style="method", markers=True)
    if db == "chroma":
        ax.set_title(f"{db.capitalize()}db")
    else:
        ax.set_title(f"{db.capitalize()}")
    ax.set_xlabel("Rank k")
    ax.set_ylabel("MAP")
    ax.grid(axis='y', linestyle='--')
    legend = ax.legend(loc="lower left")
    legend.get_title().set_fontsize(14)
    for text in legend.get_texts():
        text.set_fontsize(12)

# Set y-axis limits and ticks
for ax in axes:
    ax.set_ylim(0, 1.1)
    ax.set_yticks([i / 10 for i in range(12)])

plt.tight_layout()
plt.show()

