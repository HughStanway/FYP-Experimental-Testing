import timeit
import json
import sys
from datetime import datetime
from pathlib import Path
from ollama import Client 

COLLECTION_NAME = "embedding_metrics"
EMBED_MODEL = "llama3.2"
CHUNK_SIZES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50] 

def read_text_file(file_path):
    """Reads the entire text from the file."""
    with open(file_path, 'r') as f:
        return f.readlines()

def chunk_text_by_lines(lines, lines_per_chunk):
    """Chunks the text into groups of lines of specified size."""
    return [''.join(lines[i:i + lines_per_chunk]) for i in range(0, len(lines), lines_per_chunk)]

def get_total_embedding_time(chunks):
    """Calculates the embedding time for a list of text chunks."""
    start_time = timeit.default_timer()
    _ = ollama_client.embed(model=EMBED_MODEL, input=chunks)
    return timeit.default_timer() - start_time

def run_chunk_size_tests(file_path, chunk_sizes):
    """Tests different chunk sizes for embedding time."""
    results = {}
    text_lines = read_text_file(file_path)

    for chunk_size in chunk_sizes:
        print(f"\nTesting chunk size: {chunk_size} lines")
        
        # Split the text into chunks of current size
        chunks = chunk_text_by_lines(text_lines, chunk_size)

        # Measure embedding time for current chunk size
        embedding_time = get_total_embedding_time(chunks)
        results[chunk_size] = embedding_time
        
        print(f"Chunk size: {chunk_size} - Total embedding time: {embedding_time:.4f} seconds")

    return results

if __name__=="__main__":
    ollama_client = Client(host='http://localhost:11434')
    
    # The first argument is a folder in which all snippets are stored.
    # Traverse the folder recursively and add all files to the collection, with the file path as the ID.
    path=sys.argv[1]
    res = {}
    counter = 0
    directory = Path(path)
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix == '.txt':
            print(f"[Iteration]: {counter + 1}] - Adding document: {file_path}")
            try:
                res[str(file_path)] = run_chunk_size_tests(str(file_path), CHUNK_SIZES)
                counter += 1
            except Exception as e:
                print(f"Error: {e}")

            if counter >= 1000:
                break
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'metrics/embedding_time_results_{timestamp}.json'
    with open(output_file, 'w') as json_file:
        json.dump(res, json_file, indent=1)
    print(f"Results saved to {output_file}")