import csv
import matplotlib.pyplot as plt

# Initialize a dictionary to store data by k
k_values_data = {10: [], 50: [], 100: [], 300: []}

# Read the CSV file
with open('metrics/precision/mean_ap_at_k_ctx_window_weaviate.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        k = int(row['k'])  # Convert k to integer
        context_window_size = int(row['Context Window Size'])
        mean_average_precision = float(row['Mean Average Precision'])

        # Append to the appropriate k list
        if k in k_values_data:
            k_values_data[k].append((context_window_size, mean_average_precision))

# Sort each list by context window size in descending order
for k in k_values_data:
    k_values_data[k].sort(reverse=True, key=lambda x: x[0])

# Define custom colors for each line
colors = {10: 'blue', 50: 'yellow', 100: 'green', 300: 'red'}

# Plot line graph for all k values on the same graph
plt.figure(figsize=(12, 8))

for k, data in k_values_data.items():
    context_window_sizes = [item[0] for item in data]
    map_values = [item[1] for item in data]
    plt.plot(context_window_sizes, map_values, marker='o', color=colors[k], label=f'k = {k}')

plt.title('Mean Average Precision vs Context Window Size')
plt.xlabel('Context Window Size')
plt.ylabel('Mean Average Precision')
plt.legend(title="k Values")
plt.grid(True)
plt.tight_layout()
plt.show()
