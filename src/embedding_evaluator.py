import json
import matplotlib.pyplot as plt

file_path = 'metrics/embedding_time_results_20241110_123830.json'

try:
    with open(file_path, 'r') as json_file:
        results_data = json.load(json_file)
        print("JSON data successfully loaded.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' does not exist.")
except json.JSONDecodeError:
    print(f"Error: The file '{file_path}' is not a valid JSON file.")

counters = {
    "5":0,
    "10":0,
    "15":0,
    "20":0,
    "25":0,
    "30":0,
    "35":0,
    "40":0,
    "45":0,
    "50":0
}

num = 1
for text_file, chunk_results in results_data.items():
    for chunk_size, time_taken in chunk_results.items():
        print(f"[Iteration: {num}]  Chunk size {chunk_size}: {time_taken:.4f} seconds")
        counters[chunk_size] += time_taken
    num += 1

for counter, total_time in counters.items():
    tmp = counters[counter]
    counters[counter] = total_time / 1000

chunk_sizes = list(counters.keys())
times = list(counters.values())

plt.figure(figsize=(10, 6))
plt.bar(chunk_sizes, times, color='skyblue')
plt.xlabel('Chunk Size (lines)')
plt.ylabel('Average Embedding Time (seconds)')
plt.title('Average Embedding Time by Chunk Size')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()
