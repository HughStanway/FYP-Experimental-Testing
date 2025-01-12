# pylint: skip-file

'''
This program generates a graph for a dataset of db_size against query 
execution time.
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define a list of filenames
FILENAMES = ["metrics/execution_times/execution_times_test.csv",
             "metrics/execution_times/execution_times_test2.csv"]

DESCRIPTIONS = ["jina-embeddings-v2-base-code embedding", "voyage-3-code embedding"]

num_files = len(FILENAMES)
fig, axes = plt.subplots(1, num_files, figsize=(8 * num_files, 6), sharey=True)

temp = []
for i, csv_filename in enumerate(FILENAMES):
    df = pd.read_csv(csv_filename)
    temp.append(df['Execution Time (seconds)'].max())
max_value = max(temp)

for i, csv_filename in enumerate(FILENAMES):
    filename = csv_filename.split('/')[-1]
    df = pd.read_csv(csv_filename)

    # Extract the data for plotting execution time
    x_execution = df['Database Size']
    y_execution = df['Execution Time (seconds)']

    axes[i].set_ylim(0, max_value)
    # Plot the execution time points on the current subplot
    axes[i].scatter(x_execution, y_execution, color='blue', label='Data Points')

    # Calculate the line of best fit for execution time
    slope_execution, intercept_execution = np.polyfit(x_execution, y_execution, 1)
    line_of_best_fit_execution = slope_execution * x_execution + intercept_execution

    # Plot the line of best fit for execution time
    axes[i].plot(x_execution, line_of_best_fit_execution, color='red', label=f'Line of Best Fit: y = {float(f"{slope_execution:.16f}")}x + {float(f"{intercept_execution:.8f}")}')

    # Adding labels and title for each subplot
    axes[i].set_xlabel('Database Size')
    axes[i].set_title(DESCRIPTIONS[i])
    #axes[i].text(0.5, -0.1, DESCRIPTIONS[i], ha='center', va='top', transform=axes[i].transAxes, wrap=True)

    # Add a legend and grid to each subplot
    axes[i].legend(loc='upper left')
    axes[i].grid(True)

# Set a common ylabel for all subplots
axes[0].set_ylabel('Execution Time (seconds)')

# Show the plot with multiple subplots
plt.tight_layout()
plt.show()