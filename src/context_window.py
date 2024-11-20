# pylint: skip-file

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
COLLECTION_NAME = "POJ_DATASET_ctx_window_test"
EMBED_MODEL = "ordis/jina-embeddings-v2-base-code"

cache = dc.Cache('embedding_cache')
client = chromadb.HttpClient(host='localhost', port=8000)
ollama_client = Client(host='http://localhost:11434')

ctx_window_sizes = [256, 128, 64, 32, 16, 8]

def compute_embedding(file_text, embed_model, ctx_window):
    key = f"{file_text}-{embed_model}-{ctx_window}"
    hashed_key = hashlib.sha256(key.encode()).hexdigest()
    if hashed_key in cache:
        return cache[hashed_key]
    embeddings = ollama_client.embed(model=embed_model, input=file_text, options={"num_ctx": ctx_window})['embeddings']
    cache[hashed_key] = embeddings
    return embeddings

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

def run():
    path=sys.argv[1]
    directory = Path(path)

    for ctx_window in ctx_window_sizes:
        # Check if the collection exists
        if any(col.name == COLLECTION_NAME for col in client.list_collections()):
            client.delete_collection(COLLECTION_NAME)
            print(f"Collection, '{COLLECTION_NAME}' existed and has been dropped.")
        
        # Create a fresh collection to test
        try:
            collection = client.create_collection(
                name=COLLECTION_NAME,
                metadata={
                    "hnsw:search_ef": 100,
                    "hnsw:space": "cosine"
                },
            )
            print("Collection created successfully.")
        except Exception as e:
            print(f"Error creating collection: {e}")

        add_counter = 0
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix == '.txt':
                try:
                    with open(file_path, 'r', encoding='iso8859-1') as f:
                        file_text = f.read() 

                    start_time = time.time()
                    embeddings = compute_embedding(file_text, EMBED_MODEL, ctx_window)
                    elapsed_time = time.time() - start_time
                    print(f"Embedding time: {elapsed_time}")

                    collection.add(
                        embeddings=embeddings,
                        ids=[str(file_path)]
                    )

                    add_counter += 1
                    print(f"Db size: {add_counter} - Ctx Window: {ctx_window} - Adding Document {str(file_path)}")
                    
                except Exception as e:
                    print(f"Error adding to collection: {e}") 

        print(f"Collection size: {collection.count()}")
        queries, total_ap = 0, 0.0
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix == '.txt':
                try:
                    with open(file_path, 'r', encoding='iso8859-1') as f:
                        file_text = f.read()  

                    start_time = time.time()
                    embeddings = compute_embedding(file_text, EMBED_MODEL, ctx_window)
                    elapsed_time = time.time() - start_time
                    print(f"Embedding time: {elapsed_time}")

                    result = collection.query(
                        query_embeddings=embeddings,
                        n_results=10,
                    )
                    total_ap += average_precision(str(file_path), result["ids"][0])
                    
                    queries += 1
                    print(f"[Num queries: {queries}] - Ctx window {ctx_window} - Current mean AP: {total_ap / queries} - Path: {str(file_path)}")

                except Exception as e:
                    print(f"Error: {e}")

        # Calculate mean average precision
        mean_ap = total_ap / queries
        print(f"Mean average precision at 10: {mean_ap} for window size: {ctx_window}")

        with open("metrics/precision/mean_ap_at_10_ctx_window.csv", 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([EMBED_MODEL, ctx_window, mean_ap])

if __name__=="__main__":
    run()