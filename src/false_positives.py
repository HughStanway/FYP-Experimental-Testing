# pylint: skip-file

'''
This program loops through every file in the directory and queries against
the pre-populated vector database. It queries for the 10 closest results
and checks:

    - The first result should be the query text itself.
    - The next 9 results should be from the same POJ question.
'''

import os
import sys
import csv
import chromadb
import time
import hashlib
import diskcache as dc
from pathlib import Path
from ollama import Client 

os.environ["TOKENIZERS_PARALLELISM"] = "false"
COLLECTION_NAME = "POJ_DATASET_ollama_embedding"
EMBED_MODEL = "llama3.2"

cache = dc.Cache('embedding_cache')
client = chromadb.HttpClient(host='localhost', port=8000)
collection = client.get_collection(name=COLLECTION_NAME)
ollama_client = Client(host='http://localhost:11434')

def compute_embedding(file_text, embed_model):
    key = f"{file_text}-{embed_model}"
    hashed_key = hashlib.sha256(key.encode()).hexdigest()
    if hashed_key in cache:
        return cache[hashed_key]
    embeddings = ollama_client.embed(model=embed_model, input=file_text)['embeddings']
    cache[hashed_key] = embeddings
    return embeddings

def check_false_positive(query_path, result_paths, result_distances):
    min_distance = result_distances[0]
    min_distance_indices = [i for i, distance in enumerate(result_distances) if distance == min_distance]
    
    for index in min_distance_indices:
        if result_paths[index] == query_path:
            return True # No false positive, the query path is among the closest matches
    
    return False # False positive detected, the closest matches do not include the query path

def query(file_text):
    try:
        results = collection.query(
            query_texts=[file_text],
            n_results=5,
        )
        return results

    except Exception as e:
        print(f"Error querying collection: {e}")

def query_using_embeddings(file_text):
    try:
        start_time = time.time()
        embeddings = compute_embedding(file_text, EMBED_MODEL)
        elapsed_time = time.time() - start_time
        print(f"Embedding time: {elapsed_time}")
        results = collection.query(
            query_embeddings=embeddings,
            n_results=5,
        )
        return results

    except Exception as e:
        print(f"Error querying collection: {e}")

def run():
    path=sys.argv[1]
    counter, misses = 1, 0
    directory = Path(path)
    print(f"Collection size: {collection.count()}")
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix == '.txt':
            try:
                print(f"[DB Size: {counter}] - Misses: {misses} - File Path: {str(file_path)}")
            
                with open(file_path, 'r', encoding='iso8859-1') as f:
                    file_text = f.read()  

                result = query_using_embeddings(file_text)

                result_paths = result["ids"][0]
                result_distances =  result["distances"][0]

                if not check_false_positive(str(file_path), result_paths, result_distances):
                    misses += 1
                
                    with open("metrics/false_positives/false_positives_jina.csv", 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow([" "])
                        csv_writer.writerow(["file: ", str(file_path)])
                        csv_writer.writerow(["ids: ", result["ids"][0]])
                        csv_writer.writerow(["distances: ", result["distances"][0]])

                counter += 1

            except Exception as e:
                print(f"Error: {e}")

    with open("metrics/false_positives/false_positives_jina.csv", 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["misses - default"])
        csv_writer.writerow([misses])

if __name__=="__main__":
    run()