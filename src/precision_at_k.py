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
COLLECTION_NAME = "POJ_DATASET_jina_embedding"
EMBED_MODEL = "ordis/jina-embeddings-v2-base-code"
#EMBED_MODEL = "llama3.2"

cache = dc.Cache('embedding_cache')
client = chromadb.HttpClient(host='localhost', port=8000)
collection = client.get_collection(name=COLLECTION_NAME)
ollama_client = Client(host='http://localhost:11434')

ctx_window_sizes = [8192, 6144, 4096, 2048, 1024, 512, 256]

def compute_embedding(file_text, embed_model):
    key = f"{file_text}-{embed_model}"
    hashed_key = hashlib.sha256(key.encode()).hexdigest()
    if hashed_key in cache:
        return cache[hashed_key]
    embeddings = ollama_client.embed(model=embed_model, input=file_text)['embeddings']
    cache[hashed_key] = embeddings
    return embeddings

def query_using_embeddings(file_text):
    try:
        start_time = time.time()
        embeddings = compute_embedding(file_text, EMBED_MODEL)
        elapsed_time = time.time() - start_time
        print(f"Embedding time: {elapsed_time}")
        results = collection.query(
            query_embeddings=embeddings,
            n_results=10,
        )
        return results

    except Exception as e:
        print(f"Error querying collection: {e}")

def query(file_text):
    try:
        results = collection.query(
            query_texts=[file_text],
            n_results=10,
        )
        return results

    except Exception as e:
        print(f"Error querying collection: {e}")

def precision_at_10(query_path, results):
    query_question_number = query_path.split('/')[1]
    print(query_question_number)
    print(results)
    count = 0
    for result in results:
        result_question_number = result.split('/')[1]
        if result_question_number == query_question_number:
            count += 1
    
    return count / min(10, len(results))

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
    print(f"Collection size: {collection.count()}")
    queries, total_ap = 0, 0.0
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix == '.txt':
            try:
                with open(file_path, 'r', encoding='iso8859-1') as f:
                    file_text = f.read()  

                result = query_using_embeddings(file_text)
                total_ap += average_precision(str(file_path), result["ids"][0])
                
                queries += 1
                print(f"[Num queries: {queries}] - Current mean AP: {total_ap / queries} - Path: {str(file_path)}")

            except Exception as e:
                print(f"Error: {e}")

    # Calculate mean average precision
    mean_ap = total_ap / queries
    print(f"Mean average precision at 10: {mean_ap}")

    with open("metrics/precision/mean_ap_at_10.csv", 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([EMBED_MODEL, path, mean_ap])

if __name__=="__main__":
    run()