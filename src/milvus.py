# pylint: skip-file

'''
This program loops through all .txt files in an input directory and it's
sub-directories and inputs their contents into the vector database. It then
runs various experiments designed to test the scalability of the database
'''

import os
import sys
import csv
import time
import hashlib
from pathlib import Path
import diskcache as dc
from ollama import Client 
import tracemalloc
from pymilvus import MilvusClient
from rich import print
from tqdm import tqdm
import traceback

COLLECTION_NAME = "POJ_DATASET_milvus_test"
EMBED_MODEL = "llama3.2"
DIMENSION = 3072
K = 300

cache = dc.Cache('embedding_cache')
client = MilvusClient(uri="http://localhost:19530")
ollama_client = Client(host='http://localhost:11434')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def compute_embedding(file_text, embed_model):
    key = f"{file_text}-{embed_model}"
    hashed_key = hashlib.sha256(key.encode()).hexdigest()
    if hashed_key in cache:
        return cache[hashed_key]
    embeddings = ollama_client.embed(model=embed_model, input=file_text)['embeddings']
    cache[hashed_key] = embeddings
    return embeddings

def measure_memory_usage(func, *args, **kwargs):
    tracemalloc.start()
    result = func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory() # in bytes
    tracemalloc.stop()

    return result, peak

def query_using_embeddings(embeddings, k):
    try:
        return client.search(
            collection_name=COLLECTION_NAME,
            data=[embeddings],
            limit=k,
            search_params={"metric_type": "COSINE", "params": {}},
        )

    except Exception as e:
        print(f"Error querying collection: {e}")

def check_false_positive(query_path, results):
    min_distance = results[0]['distance']
    min_distance_indices = [i for i, result in enumerate(results) if result['distance'] == min_distance]

    for index in min_distance_indices:
        if results[index]['id'] == query_path:
            return True  # No false positive, the query path is among the closest matches
    
    return False  # False positive detected, the closest matches do not include the query path

def average_precision(query_path, results):
    query_question_number = query_path.split('/')[1]
    n_relevant = sum(1 for result in results if result['id'].split('/')[1] == query_question_number)

    if n_relevant == 0:
        return 0.0

    precision_scores = []
    retrieved_relevant_count = 0

    for i, result in enumerate(results, start=1):
        is_relevant = result['id'].split('/')[1] == query_question_number

        if is_relevant:
            retrieved_relevant_count += 1
            precision_at_k = retrieved_relevant_count / i
            precision_scores.append(precision_at_k)

    # Average precision = mean of precision scores at ranks where relevant items appear
    return sum(precision_scores) / n_relevant

if __name__=="__main__":
    path=sys.argv[1]
    directory = Path(path)
    
    print(f"[bold red][+] Building database[/bold red]")
    
    # Check if the collection exists
    if any(col == COLLECTION_NAME for col in client.list_collections()):
        client.drop_collection(collection_name=COLLECTION_NAME)
        print(f"[yellow][-] Collection, '{COLLECTION_NAME}' existed and has been dropped.[/yellow]")
    
    # Create a fresh collection to test
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=DIMENSION,
            id_type="string",
            max_length=512
        )
        print(f"[yellow][-] New collection: {COLLECTION_NAME} created successfully.[/yellow]")
    except Exception as e:
        print(f"Error creating collection: {e}")

    progress_bar = tqdm(total=52000, desc="Building database")
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix == '.txt':
            try:
                with open(file_path, 'r', encoding='iso8859-1') as f: 
                    file_text = f.read()

                # Create embedding for text
                start_time = time.time()
                embeddings = compute_embedding(file_text, EMBED_MODEL).pop()
                elapsed_time = time.time() - start_time

                client.insert(
                    collection_name=COLLECTION_NAME,
                    data={"id": str(file_path), "vector": embeddings}
                )
                
                progress_bar.set_postfix({"Embedding time": elapsed_time})
                progress_bar.update(1)
            except Exception as e:
                print(f"Error adding to collection: {e}")
    progress_bar.close()
    
    progress_bar = tqdm(total=52000, desc="Testing false positives")
    misses = 0
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix == '.txt':
            try:
                with open(file_path, 'r', encoding='iso8859-1') as f: 
                    file_text = f.read()

                # Create embedding for text
                start_time = time.time()
                embeddings = compute_embedding(file_text, EMBED_MODEL).pop()
                elapsed_time = time.time() - start_time

                result = query_using_embeddings(embeddings=embeddings, k=K).pop()
                if not check_false_positive(str(file_path), result):
                    misses += 1
                
                    with open(f"metrics/false_positives/false_positives_@{K}_milvus_llama.csv", 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow([" "])
                        csv_writer.writerow(["file: ", str(file_path)])
                        csv_writer.writerow(["results: ", result])
                
                progress_bar.set_postfix({"Misses": misses})
                progress_bar.update(1)

            except Exception as e:
                print(f"Error: {e}")

    progress_bar.close()

    with open(f"metrics/false_positives/false_positives_@{K}_milvus_llama.csv", 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["misses:"])
        csv_writer.writerow([misses])

    print("Indexing complete.")
