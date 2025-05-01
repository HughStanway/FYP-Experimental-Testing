# pylint: skip-file

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from extract_accuracy_scores import get_accuracy_scores

sns.set_theme(style="ticks")
sns.set_context("notebook", font_scale=1.2)

filepath = "metrics/accuracy/POJ"
data = get_accuracy_scores(filepath)['qdrant']

x_points = [10, 50, 100, 300]

# Convert the nested dictionary to a tidy DataFrame
def tidy_nested_data(data):
    records = []
    for method, metrics in data.items():
        if method == "ordis-jina-embeddings-v2-base-code":
            method = "jina-v2-base-code"
        for metric, values in metrics.items():
            if metric == "MAP":  # Only keep MAP metric
                counter = 0
                for x, value in enumerate(values):
                    records.append({"x": x_points[counter], "method": method, "metric": metric, "value": value})
                    counter += 1
    return pd.DataFrame(records)

df = tidy_nested_data(data)
method_order = ["voyage-code-3", "jina-v2", "deepseek-r1:1.5B", "llama3.2"]
df = df.sort_values(by="method", key=lambda x: x.map({m: i for i, m in enumerate(method_order)}))

# Create MAP plot
plt.figure(figsize=(6, 5))
sns.lineplot(data=df, x="x", y="value", hue="method", style="method", markers=True, legend=True)
plt.title("Mean Average Precision Against Rank k")
plt.xlabel("Rank k")
plt.ylabel("MAP")
plt.grid(axis='y', linestyle='--')
plt.ylim(0, 1.1)
plt.yticks([i / 10 for i in range(12)])  # Y-ticks from 0 to 1 with a step of 0.1

# Adjust legend
legend = plt.legend(loc="lower left")
legend.get_title().set_fontsize(14)
for text in legend.get_texts():
    text.set_fontsize(12)

plt.tight_layout()
plt.show()
