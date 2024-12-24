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
from pymilvus import MilvusClient
import time
import hashlib
import diskcache as dc
from pathlib import Path
from ollama import Client 
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
COLLECTION_NAME = "fp_test"
EMBED_MODEL = "ordis/jina-embeddings-v2-base-code"
K = 10
DIMENSION = 768

cache = dc.Cache('embedding_cache')
client = MilvusClient(uri="http://localhost:19530")
ollama_client = Client(host='http://localhost:11434')

def compute_embedding(file_text, embed_model):
    key = f"{file_text}-{embed_model}"
    hashed_key = hashlib.sha256(key.encode()).hexdigest()
    if hashed_key in cache:
        return cache[hashed_key]
    embeddings = ollama_client.embed(model=embed_model, input=file_text)['embeddings']
    cache[hashed_key] = embeddings
    return embeddings

def check_false_positive(query_path, results):
    min_distance = results[0]['distance']
    min_distance_indices = [i for i, result in enumerate(results) if result['distance'] == min_distance]

    for index in min_distance_indices:
        if results[index]['id'] == query_path:
            return True  # No false positive, the query path is among the closest matches
    
    return False  # False positive detected, the closest matches do not include the query path

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

if __name__ == "__main__":
    path=sys.argv[1]
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

    directory = Path(path)
    c= 0
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
                if c == 50:
                    break
                c += 1

            except Exception as e:
                print(f"Error adding to collection: {e}")
    progress_bar.close()
    
    misses = 0
    directory = Path(path)
    progress_bar = tqdm(total=52000, desc="testing database")
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix == '.txt':
            try:
                with open(file_path, 'r', encoding='iso8859-1') as f:
                    file_text = f.read()  

                result = query_using_embeddings(embeddings=embeddings, k=K).pop()
                print(result)
                '''
                if not check_false_positive(str(file_path), result_paths, result_distances):
                    misses += 1
                
                    with open(f"metrics/false_positives/false_positives_@{K}_milvus_jina.csv", 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow([" "])
                        csv_writer.writerow(["file: ", str(file_path)])
                        csv_writer.writerow(["ids: ", result["ids"][0]])
                        csv_writer.writerow(["distances: ", result["distances"][0]])

                progress_bar.set_postfix({"misses": misses})
                progress_bar.update(1)
                '''
                break

            except Exception as e:
                print(f"Error: {e}")
    progress_bar.close()

    with open(f"metrics/false_positives/false_positives_@{K}_milvus_jina.csv", 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["misses:"])
        csv_writer.writerow([misses])