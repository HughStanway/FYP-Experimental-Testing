# pylint: skip-file

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme(style="whitegrid")
sns.set_context("notebook", font_scale=1.2) 

FILEPATH = "metrics/storage_tests/"
FILENAMES = ["storage_tests_qdrant-voyage-code-3.csv",
             "storage_tests_qdrant-ordis-jina-embeddings-v2-base-code.csv",
             "storage_tests_qdrant-llama3.2.csv",
             "storage_tests_qdrant-deepseek-r1:1.5B.csv"]
DIMENSIONS = [1024, 768, 3078, 1536]

temp = []
for i, filename in enumerate(FILENAMES):
    csv_filename = FILEPATH + filename
    df = pd.read_csv(csv_filename)
    temp.append(df['Storage Size'].max())
y_max = max(temp)


# Get data
x = []
y = []
averages = []
for i, filename in enumerate(FILENAMES):
    csv_filename = FILEPATH + filename
    df = pd.read_csv(csv_filename)
    x_tmp = df['Database Size']
    y_tmp = df['Storage Size']
    x.append(x_tmp)
    y.append(y_tmp)
    print(f"Average {i}: {(sum(y_tmp) / len(x_tmp)):.3e}")
    print(f"Average per dimension {DIMENSIONS[i]}: {((sum(y_tmp) / len(x_tmp)) / DIMENSIONS[i]):.3e}")

# Create a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
data = [(x[0], y[0], "Voyage-code-3"),
        (x[1], y[1], "Jina-v2-base-code"),
        (x[2], y[2], "Llama3.2"),
        (x[3], y[3], "DeepSeek-r1:1.5B")]

for ax, (x, y, title) in zip(axes.flatten(), data):
    df = pd.DataFrame({'x': x, 'y': y})
    line = sns.regplot(x='x', y='y', data=df, ax=ax, color='green', scatter_kws={'s': 50}, line_kws={'color': 'red'})
    
    # Compute the line of best fit
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs[0], coeffs[1]
    equation = f"y = {slope:.3e}x + {intercept:.3e}"
    print(f"{title}: {equation}")
    
    # Create a custom legend with a red line symbol
    legend_line = plt.Line2D([0], [0], color='red', lw=1, label=equation)
    ax.legend(handles=[legend_line], loc='upper left')
    ax.set_title(title)
    ax.set_xlabel("Database Size")
    ax.set_ylabel("Storage Size")
    ax.set_ylim(0, y_max)

plt.tight_layout()
plt.show()
