# pylint: skip-file

'''
This program generates a graph for a dataset of db_size against query 
execution time.
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define a list of filenames
FILENAMES = ["memory_usage_weaviate_2024:12:11-llama3.2-add.csv",
             "memory_usage_weaviate_2024:12:11-ordis-jina-embeddings-v2-base-code-add.csv"]

DESCRIPTIONS = ["Pre-embed and add to collection using llama2.3 embedding",
                "Pre-embed and add to collection using jina-embeddings-v2-base-code embedding"]
REMOVE_OUTLIERS = False

num_files = len(FILENAMES)
fig, axes = plt.subplots(1, num_files, figsize=(8 * num_files, 6), sharey=True)

for i, filename in enumerate(FILENAMES):
    csv_filename = f"metrics/memory_usage/{filename}"
    df = pd.read_csv(csv_filename)

    # Optinally filter out the outliers for execution time
    if REMOVE_OUTLIERS:
        mean_execution_time = df['Execution Time (seconds)'].mean()
        std_execution_time = df['Execution Time (seconds)'].std()

        df = df[(df['Execution Time (seconds)'] >= mean_execution_time - std_execution_time) & 
                (df['Execution Time (seconds)'] <= mean_execution_time + std_execution_time)]

    # Extract the data for plotting execution time
    x_execution = df['Database Size']
    y_execution = df['Memory Usage (MB)']
    # Plot the execution time points on the current subplot
    axes[i].scatter(x_execution, y_execution, color='green', label='Data Points')

    # Calculate the line of best fit for execution time
    slope_execution, intercept_execution = np.polyfit(x_execution, y_execution, 1)
    line_of_best_fit_execution = slope_execution * x_execution + intercept_execution

    # Plot the line of best fit for execution time
    axes[i].plot(x_execution, line_of_best_fit_execution, color='red', label=f'Line of Best Fit: y = {slope_execution}x + {intercept_execution}')

    # Adding labels and title for each subplot
    axes[i].set_xlabel('Database Size')
    if REMOVE_OUTLIERS:
        axes[i].set_title(f'File: {filename}')
    else:
        axes[i].set_title(f'File: {filename}')
    axes[i].text(0.5, -0.1, DESCRIPTIONS[i], ha='center', va='top', transform=axes[i].transAxes, wrap=True)

    # Add a legend and grid to each subplot
    axes[i].legend()
    axes[i].grid(True)

# Set a common ylabel for all subplots
axes[0].set_ylabel('Memory Usage (MB)')

# Show the plot with multiple subplots
plt.tight_layout()
plt.show()