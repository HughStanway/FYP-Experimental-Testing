# pylint: skip-file

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme(style="whitegrid")
sns.set_context("notebook", font_scale=1.2) 

FILEPATH = ""
FILENAMES = ["metrics/execution_times/execution_times_qdrant_2025:01:02-voyage-code-3-query.csv",
             "metrics/execution_times/execution_times_qdrant_2024:12:26-ordis-jina-embeddings-v2-base-code-query.csv",
             "metrics/execution_times/execution_times_qdrant_2024:12:26-llama3.2-query.csv",
             "metrics/execution_times/execution_times_qdrant_2025:01:23-deepseek-r1:1.5B-query.csv"]

temp = []
for filename in FILENAMES:
    csv_filename = FILEPATH + filename
    df = pd.read_csv(csv_filename)
    temp.append(df['Execution Time (seconds)'].max())
y_max = max(temp)

# Get data
x = []
y = []
averages = []
for i, filename in enumerate(FILENAMES):
    csv_filename = FILEPATH + filename
    df = pd.read_csv(csv_filename)
    x_tmp = df['Database Size']
    y_tmp = df['Execution Time (seconds)']
    x.append(x_tmp)
    y.append(y_tmp)
    mean_val = (sum(y_tmp) / len(x_tmp)) * 1000  # Convert to milliseconds
    median_val = np.median(y_tmp) * 1000  # Convert to milliseconds
    print(f"Dataset {i}: Mean = {mean_val:.6f} ms, Median = {median_val:.6f} ms")
'''
# Create a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
data = [(x[0], y[0], "Voyage-code-3"),
        (x[1], y[1], "Jina-v2-base-code"),
        (x[2], y[2], "Llama3.2"),
        (x[3], y[3], "DeepSeek-r1:1.5B")]

for ax, (x, y, title) in zip(axes.flatten(), data):
    df = pd.DataFrame({'x': x, 'y': y})
    line = sns.regplot(x='x', y='y', data=df, ax=ax, scatter_kws={'s': 50}, line_kws={'color': 'red'})
    
    # Compute the line of best fit
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs[0], coeffs[1]
    equation = f"y = {slope:.3e}x + {intercept:.6f}"
    print(f"{title}: {equation}")
    
    # Create a custom legend with a red line symbol
    legend_line = plt.Line2D([0], [0], color='red', lw=2, label=equation)
    ax.legend(handles=[legend_line], loc='upper left')
    ax.set_title(title)
    ax.set_xlabel("Database Size")
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_ylim(0, y_max)

plt.tight_layout()
plt.show()

# Create a separate figure with all lines of best fit
plt.figure(figsize=(6, 5))  # Same size as a single plot in the 2x2 grid
line_styles = ['-', '--', '-.', ':']  # Different line styles for each model

for i, (x, y, title) in enumerate(data):
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs[0], coeffs[1]
    plt.plot(x, slope * x + intercept, label=f"{title}: y = {slope:.1f}x + {intercept:.6f}", 
             linestyle=line_styles[i], linewidth=3)

plt.xlabel("Database Size")
plt.ylabel("Peak Memory Usage (bytes)")
plt.legend()
plt.grid(True)
plt.ylim(0, 0.05)
plt.tight_layout()
plt.show()
'''