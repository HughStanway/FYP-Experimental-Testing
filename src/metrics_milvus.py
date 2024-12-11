# pylint: skip-file

import timeit
import os
import sys
import csv
import time
import hashlib
from datetime import datetime
from pathlib import Path
import chromadb
import diskcache as dc
from datetime import datetime
from ollama import Client 
from pymilvus import MilvusClient
import tracemalloc
import psutil
from tqdm import tqdm

COLLECTION_NAME = "POJ_DATASET_fixed_db_size_query_test"
EMBED_MODEL = "ordis/jina-embeddings-v2-base-code"
COSINE_SEACH_N = 100
DIMENSION = 768

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

def get_current_total_memory_usage():
    process = psutil.Process(os.getpid()) 
    memory_info = process.memory_info()
    return memory_info.rss / (1024 ** 2)

def query_using_embeddings(embeddings):
    try:
        return client.search(
            collection_name=COLLECTION_NAME,
            data=[embeddings],
            limit=5,
            search_params={"metric_type": "COSINE", "params": {}},
        )

    except Exception as e:
        print(f"Error querying collection: {e}")

if __name__=="__main__":
    path=sys.argv[1]
    # Check if the collection exists
    if any(col == COLLECTION_NAME for col in client.list_collections()):
        client.drop_collection(collection_name=COLLECTION_NAME)
        print(f"Collection, '{COLLECTION_NAME}' existed and has been dropped.")
    
    # Create a fresh collection to test
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=DIMENSION,
            id_type="string",
            max_length=512
            )
        print("Collection created successfully.")
    except Exception as e:
        print(f"Error creating collection: {e}")
    
    # Initialize CSV and write header to record results to
    timestamp = datetime.now().strftime("%Y:%m:%d")
    csv_filename = f'metrics/execution_times/execution_times_const_db_size_{timestamp}-{EMBED_MODEL.replace("/", "-")}-query.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Database Size', 'File Path', 'Execution Time (seconds)'])

    directory = Path(path)
    progress_bar = tqdm(total=52000, desc="Building database")
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix == '.txt':
            try:
                with open(file_path, 'r', encoding='iso8859-1') as f: 
                    file_text = f.read()

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

    db_size = 0
    total_time = 0
    progress_bar = tqdm(total=52000, desc="Execution time test")
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix == '.txt':
            try:
                with open(file_path, 'r', encoding='iso8859-1') as f: 
                    file_text = f.read()

                # Create embedding for text
                start_time = time.time()
                embeddings = compute_embedding(file_text, EMBED_MODEL).pop()
                elapsed_time = time.time() - start_time
                
                # Query the db for the document and time execution
                memory_before = get_current_total_memory_usage()
                execution_time = timeit.timeit(lambda: query_using_embeddings(embeddings=embeddings), number=1)
                memory_after = get_current_total_memory_usage()
                total_time += execution_time

                with open(csv_filename, 'a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([db_size, file_path, execution_time, total_time, memory_before, memory_after])

                db_size += 1
                progress_bar.set_postfix({"Embedding time": elapsed_time})
                progress_bar.update(1)

            except Exception as e:
                print(f"Error: {e}")
    progress_bar.close()

    print("Indexing complete.")
    