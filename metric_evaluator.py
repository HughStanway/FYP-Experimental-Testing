import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FILENAME = "execution_times_2024:10:17_18:11.csv"

# Read the CSV file into a pandas DataFrame
csv_filename = f"metrics/{FILENAME}"
df = pd.read_csv(csv_filename)

# Calculate the mean and standard deviation of the execution time
mean_execution_time = df['Execution Time (seconds)'].mean()
std_execution_time = df['Execution Time (seconds)'].std()

# Filter out the outliers for execution time
filtered_execution_time_df = df[(df['Execution Time (seconds)'] >= mean_execution_time - std_execution_time) & 
                                 (df['Execution Time (seconds)'] <= mean_execution_time + std_execution_time)]

# Extract the data for plotting execution time
x_execution = filtered_execution_time_df['Database Size']
y_execution = filtered_execution_time_df['Execution Time (seconds)']

# Plot the execution time points
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)  # Create a subplot for execution time
plt.scatter(x_execution, y_execution, color='blue', label='Data Points')

# Calculate the line of best fit for execution time
slope_execution, intercept_execution = np.polyfit(x_execution, y_execution, 1)
line_of_best_fit_execution = slope_execution * x_execution + intercept_execution

# Plot the line of best fit for execution time
plt.plot(x_execution, line_of_best_fit_execution, color='red', label='Line of Best Fit')

# Adding labels and title for execution time
plt.xlabel('Database Size')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs. Database Size (Outliers Removed)')
plt.legend()
plt.grid(True)

# Calculate the mean and standard deviation of the memory usage
mean_memory_usage = df['Memory Usage (MiB)'].mean()
std_memory_usage = df['Memory Usage (MiB)'].std()

# Filter out the outliers for memory usage
filtered_memory_df = df[(df['Memory Usage (MiB)'] >= mean_memory_usage - std_memory_usage) & 
                         (df['Memory Usage (MiB)'] <= mean_memory_usage + std_memory_usage)]

# Extract the data for plotting memory usage
x_memory = filtered_memory_df['Database Size']
y_memory = filtered_memory_df['Memory Usage (MiB)']

# Plot the memory usage points
plt.subplot(1, 2, 2)  # Create a subplot for memory usage
plt.scatter(x_memory, y_memory, color='green', label='Data Points')

# Calculate the line of best fit for memory usage
slope_memory, intercept_memory = np.polyfit(x_memory, y_memory, 1)
line_of_best_fit_memory = slope_memory * x_memory + intercept_memory

# Plot the line of best fit for memory usage
plt.plot(x_memory, line_of_best_fit_memory, color='orange', label='Line of Best Fit')

# Adding labels and title for memory usage
plt.xlabel('Database Size')
plt.ylabel('Memory Usage (MiB)')
plt.title('Memory Usage vs. Database Size (Outliers Removed)')
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()
