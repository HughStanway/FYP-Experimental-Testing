# pylint: skip-file

import timeit
import os
import sys
import csv
import time
import hashlib
from datetime import datetime
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import diskcache as dc
from ollama import Client 
import tracemalloc
import psutil
from tqdm import tqdm
from rich import print
import voyageai

COLLECTION_NAME = "POJ_DATASET_qdrant_test"
EMBED_MODEL = "voyage-code-3"
COSINE_SEACH_N = 100
DIMENSION = 1024

vo = voyageai.Client()
cache = dc.Cache('embedding_cache_voyage')
client = QdrantClient(url="http://localhost:6333")
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

def compute_voyageai_embedding(file_text, embed_model):
    key = f"{file_text}-{embed_model}"
    hashed_key = hashlib.sha256(key.encode()).hexdigest()
    if hashed_key in cache:
        return cache[hashed_key]
    query_embedding = vo.embed([file_text], model=embed_model, input_type="query").embeddings[0]
    cache[hashed_key] = query_embedding
    return query_embedding

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

def query_using_embeddings(embeddings, k):
    try:
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=embeddings,
            limit=k
        )
    except Exception as e:
        print(f"Error querying collection: {e}")

def add_to_collection_using_embeddings(embeddings, file_path, id):
    try:
        client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
                PointStruct(
                        id=id,
                        vector=embeddings,
                        payload={"file_path": file_path}
                )
                
            ]
        )
    except Exception as e:
        print(f"Error adding to collection: {e}")

def extract_scores_and_paths(results):
    scores = []
    file_paths = []
    for point in results:
        scores.append(point.score)
        file_paths.append(point.payload['file_path'])
    return scores, file_paths

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
    directory = Path(path)

    try:
        # Delete collection if it already exists
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)
            print(f"[yellow]Collection, '{COLLECTION_NAME}' existed and has been dropped.[/yellow]")

        # Create a fresh collection to test
        client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=DIMENSION, distance=Distance.COSINE),
        )
        print("[yellow]Collection created successfully.[/yellow]")
    except Exception as e:
        print(f"[red]Error creating collection: {e}[/red]")
    '''
    # Initialize CSV and write header to record results to
    timestamp = datetime.now().strftime("%Y:%m:%d")
    csv_filename = f'metrics/memory_usage/memory_usage_qdrant_{timestamp}-{EMBED_MODEL.replace("/", "-")}-add.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Query','File Path','Memory Usage (MB)'])
    '''
    progress_bar = tqdm(total=52000, desc="Building database")
    id = 0
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix == '.txt':
            try:
                with open(file_path, 'r', encoding='iso8859-1') as f: 
                    file_text = f.read()

                # Create embedding for text
                start_time = time.time()
                embeddings = compute_voyageai_embedding(file_text, EMBED_MODEL)
                elapsed_time = time.time() - start_time

                add_to_collection_using_embeddings(embeddings=embeddings, file_path=str(file_path), id=id)
                
                id += 1
                progress_bar.set_postfix({"Embedding time": elapsed_time})
                progress_bar.update(1)

            except Exception as e:
                print(f"Error adding to collection: {e}")
    progress_bar.close()
    
    progress_bar = tqdm(total=52000, desc="Testing database")
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

                # Create embedding for text
                start_time = time.time()
                embeddings = compute_voyageai_embedding(file_text, EMBED_MODEL)
                elapsed_time = time.time() - start_time
                
                for k in [10,50,100,300]:
                    results = query_using_embeddings(embeddings=embeddings, k=k)
                    result_scores, result_paths = extract_scores_and_paths(results=results)

                    total_ap[k] += average_precision(str(file_path), result_paths)
                
                progress_bar.set_postfix({"Embedding time": elapsed_time})
                progress_bar.update(1)
                queries += 1

            except Exception as e:
                print(f"Error adding to collection: {e}")
    progress_bar.close()

    with open("metrics/precision/mean_ap_at_k_qdrant.csv", 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for k in [10,50,100,300]:
            mean_ap = total_ap[k] / queries
            csv_writer.writerow([EMBED_MODEL, k, 8192, mean_ap])
    
    print(total_ap)
    print("[yellow]Indexing complete.[/yellow]")