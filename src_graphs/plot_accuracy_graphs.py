# pylint: skip-file

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from extract_accuracy_scores import get_accuracy_scores

sns.set_theme(style="ticks")

filepath = "metrics/accuracy/GCJ2-4_cpp"
data = get_accuracy_scores(filepath)["qdrant"]

# Convert the nested dictionary to a tidy DataFrame
def tidy_nested_data(data):
    records = []
    for method, metrics in data.items():
        if method == "ordis-jina-embeddings-v2-base-code":
            method = "jina-v2"
        for metric, values in metrics.items():
            for x, value in enumerate(values):
                records.append({"x": x, "method": method, "metric": metric, "value": value})
    return pd.DataFrame(records)

df = tidy_nested_data(data)

# Create subplots for Precision and MAP
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot for Precision
sns.lineplot(ax=axes[0], data=df[df["metric"] == "Precision"], x="x", y="value", hue="method", style="method", markers=True, legend=False)
axes[0].set_title("Precision Against Rank k")
axes[0].set_xlabel("Rank k")
axes[0].set_ylabel("Precision")
axes[0].grid(axis='y', linestyle='--')

# Plot for MAP
sns.lineplot(ax=axes[1], data=df[df["metric"] == "MAP"], x="x", y="value", hue="method", style="method", markers=True, legend=True)
axes[1].set_title("Mean Average Precision Against Rank k")
axes[1].set_xlabel("Rank k")
axes[1].set_ylabel("MAP")
axes[1].grid(axis='y', linestyle='--')
axes[1].legend(title="Model", loc="lower left")

# Set y-axis limits and ticks to be between 0 and 1 with a step of 0.1
axes[0].set_ylim(0, 1.1)
axes[1].set_ylim(0, 1.1)
axes[0].set_yticks([i / 10 for i in range(12)])  # Y-ticks from 0 to 1 with a step of 0.1
axes[1].set_yticks([i / 10 for i in range(12)])  # Y-ticks from 0 to 1 with a step of 0.1


# Adjust layout and show the plot
plt.show()

