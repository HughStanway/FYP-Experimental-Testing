#pylint: skip-file

import os
import sys
import csv
import time
import hashlib
import traceback
import psutil
import weaviate
import tracemalloc
import weaviate.classes as wvc
import diskcache as dc
from ollama import Client
from pathlib import Path
from tqdm import tqdm
from rich import print

COLLECTION_NAME = "POJ_DATASET_weaviate_test"
EMBED_MODEL = "ordis/jina-embeddings-v2-base-code"
COSINE_SEACH_N = 100
ctx_window_sizes = [8192, 6144, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8]

cache = dc.Cache('embedding_cache')
client = weaviate.connect_to_local()
ollama_client = Client(host='http://localhost:11434')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def compute_embedding(file_text, embed_model, ctx_window):
    key = f"{file_text}-{embed_model}-{ctx_window}"
    hashed_key = hashlib.sha256(key.encode()).hexdigest()
    if hashed_key in cache:
        return cache[hashed_key]
    embeddings = ollama_client.embed(model=embed_model, input=file_text, options={"num_ctx": ctx_window})['embeddings']
    cache[hashed_key] = embeddings
    return embeddings

def measure_memory_usage(func, *args, **kwargs):
    tracemalloc.start()
    result = func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory() # in bytes
    tracemalloc.stop()

    return result, peak

def get_current_total_memory_usage():
    process = psutil.Process(os.getpid()) 
    memory_info = process.memory_info()
    return memory_info.rss / (1024 ** 2)

def extract_certainty_and_paths(query_return):
    certainty_list = []
    file_paths_list = []

    for obj in query_return.objects:
        # Extract certainty and file path, and append to respective lists
        certainty_list.append(obj.metadata.certainty)
        file_paths_list.append(obj.properties['file_path'])

    return certainty_list, file_paths_list

def query_using_embeddings(embeddings, k):
    try:
        return collection.query.near_vector(
            near_vector=embeddings,
            limit=k,
            return_metadata=wvc.query.MetadataQuery(certainty=True)
        )

    except Exception as e:
        print(f"Error querying collection: {e}")

def add_to_collection_using_embeddings(embeddings, file_path):
    try:
        collection.data.insert(
            properties={
                "file_path": file_path,
            },
            vector=embeddings
        )
    except Exception as e:
        print(f"Error querying collection: {e}")

def check_false_positive(query_path, result_paths, result_distances):
    min_distance = result_distances[0]
    min_distance_indices = [i for i, distance in enumerate(result_distances) if distance == min_distance]
    
    for index in min_distance_indices:
        if result_paths[index] == query_path:
            return True # No false positive, the query path is among the closest matches
    
    return False # False positive detected, the closest matches do not include the query path

def average_precision(query_path, results):
    query_question_number = query_path.split('/')[1]
    n_relevant = sum(1 for result in results if result.split('/')[1] == query_question_number)

    if n_relevant == 0:
        return 0.0 

    precision_scores = []
    retrieved_relevant_count = 0

    for i, result in enumerate(results, start=1):
        is_relevant = result.split('/')[1] == query_question_number

        if is_relevant:
            retrieved_relevant_count += 1
            precision_at_k = retrieved_relevant_count / i
            precision_scores.append(precision_at_k)

    # Average precision = mean of precision scores at ranks where relevant items appear
    return sum(precision_scores) / n_relevant

if __name__=="__main__":
    path=sys.argv[1]
    for ctx in ctx_window_sizes:
        print(f"[bold red]Testing context window size: {ctx}[/bold red]")
        try:
            # Check if the collection exists
            if any(col == COLLECTION_NAME for col in client.collections.list_all().keys()):
                client.collections.delete(name=COLLECTION_NAME)
                print(f"[yellow][-] Collection, '{COLLECTION_NAME}' existed and has been dropped.[/yellow]")
        
            # Create a fresh collection to test
            collection = client.collections.create(
                COLLECTION_NAME,
                vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            )
            print(f"[yellow][-] New collection: {COLLECTION_NAME} created successfully.[/yellow]")
        except Exception as e:
            print(f"[bold red]Error during collection setup: {e}[/bold red]")
        
        directory = Path(path)
        progress_bar = tqdm(total=52000, desc="Building database")
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix == '.txt':
                try:
                    with open(file_path, 'r', encoding='iso8859-1') as f: 
                        file_text = f.read()

                    start_time = time.time()
                    embeddings = compute_embedding(file_text, EMBED_MODEL, ctx).pop()
                    elapsed_time = time.time() - start_time

                    add_to_collection_using_embeddings(embeddings, str(file_path))

                    progress_bar.set_postfix({"Embedding time": elapsed_time})
                    progress_bar.update(1)

                except Exception as e:
                    print(f"[bold red]Error adding to collection: {e}[/bold red]")
        progress_bar.close()

        directory = Path(path)
        progress_bar = tqdm(total=52000, desc="Running tests")
        total_ap = {
            10:0,
            50:0,
            100:0,
            300:0,
        }
        queries = 0
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix == '.txt':
                try:
                    with open(file_path, 'r', encoding='iso8859-1') as f: 
                        file_text = f.read()

                    start_time = time.time()
                    embeddings = compute_embedding(file_text, EMBED_MODEL, ctx).pop()
                    elapsed_time = time.time() - start_time

                    for k in [10,50,100,300]:
                        result = query_using_embeddings(embeddings=embeddings, k=k)
                        result_distances, result_paths = extract_certainty_and_paths(result)

                        total_ap[k] += average_precision(str(file_path), result_paths)

                    progress_bar.set_postfix({"Embedding time": elapsed_time})
                    progress_bar.update(1)
                    queries += 1

                except Exception as e:
                    print(f"[bold red]Error testing collection: {e}[/bold red]")
                    traceback.print_exc()
        progress_bar.close()
        
        with open("metrics/precision/mean_ap_at_k_ctx_window_weaviate.csv", 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            for k in [10,50,100,300]:
                mean_ap = total_ap[k] / queries
                csv_writer.writerow([EMBED_MODEL, k, ctx, mean_ap])

    client.close()
    print("[yellow]Indexing complete.[/yellow]")