import csv
import matplotlib.pyplot as plt

# Initialize a dictionary to store data by context window size for each embedding model
embedding_model_data = {
    "default":{},
    "ordis/jina-embeddings-v2-base-code": {},
    "llama3.2": {},
    "voyage-code-3": {},
}

# Read the CSV file
with open('metrics/precision/mean_ap_at_k_chroma.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        embedding_model = row['Embedding Model']
        k = int(row['k'])  # Convert k to integer
        context_window_size = int(row['context window size'])
        mean_average_precision = float(row['Mean Average Precision'])

        # Organize data by context window size
        if embedding_model not in embedding_model_data:
            embedding_model_data[embedding_model] = {}
        if context_window_size not in embedding_model_data[embedding_model]:
            embedding_model_data[embedding_model][context_window_size] = []

        embedding_model_data[embedding_model][context_window_size].append((k, mean_average_precision))

# Sort each list by k in ascending order
for model, context_sizes in embedding_model_data.items():
    for context_window_size, data in context_sizes.items():
        context_sizes[context_window_size].sort(key=lambda x: x[0])

# Define custom colors for each embedding model
model_colors = {
    "default": 'red',
    "ordis/jina-embeddings-v2-base-code": 'green',
    "llama3.2": 'blue',
    "voyage-code-3": 'orange',
}

# Plot line graphs for all embedding models and context window sizes
plt.figure(figsize=(12, 8))

for model, context_sizes in embedding_model_data.items():
    for context_window_size, data in context_sizes.items():
        k_values = [item[0] for item in data]
        map_values = [item[1] for item in data]
        plt.plot(k_values, map_values, marker='o', label=f'{model}', color=model_colors[model])

plt.title('Mean Average Precision vs k')
plt.xlabel('k')
plt.ylabel('Mean Average Precision')
plt.legend(title="Embedding Models")
plt.grid(True)
plt.tight_layout()
plt.show()
