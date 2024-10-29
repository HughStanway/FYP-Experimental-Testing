'''
This program generates a graph for a dataset of db_size against query 
execution time.
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FILENAME = "execution_times_2024:10:27_17:44.csv"
REMOVE_OUTLIERS = False

# Read the CSV file into a pandas DataFrame
csv_filename = f"metrics/{FILENAME}"
df = pd.read_csv(csv_filename)

# Calculate the mean and standard deviation of the execution time
mean_execution_time = df['Execution Time (seconds)'].mean()
std_execution_time = df['Execution Time (seconds)'].std()

# Filter out the outliers for execution time
if REMOVE_OUTLIERS:
   df = df[(df['Execution Time (seconds)'] >= mean_execution_time - std_execution_time) & 
                                    (df['Execution Time (seconds)'] <= mean_execution_time + std_execution_time)]

# Extract the data for plotting execution time
x_execution = df['Database Size']
y_execution = df['Execution Time (seconds)']

# Plot the execution time points
plt.figure(figsize=(8, 6))
plt.scatter(x_execution, y_execution, color='blue', label='Data Points')

# Calculate the line of best fit for execution time
slope_execution, intercept_execution = np.polyfit(x_execution, y_execution, 1)
line_of_best_fit_execution = slope_execution * x_execution + intercept_execution

# Plot the line of best fit for execution time
plt.plot(x_execution, line_of_best_fit_execution, color='red', label='Line of Best Fit')

# Adding labels and title for execution time
plt.xlabel('Database Size')
plt.ylabel('Execution Time (seconds)')
if REMOVE_OUTLIERS:
    plt.title('Execution Time vs. Database Size (Outliers Removed)')
else:
    plt.title('Execution Time vs. Database Size (Outliers Not Removed)')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
