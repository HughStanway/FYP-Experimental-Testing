# pylint: skip-file

import voyageai
import hashlib
import csv
import diskcache as dc
from pathlib import Path
import time
from tqdm import tqdm
from pymilvus import MilvusClient
from rich import print
from ollama import Client 
import sys

vo = voyageai.Client()
cache = dc.Cache('embedding_cache_voyage')
client = MilvusClient(uri="http://localhost:19530")
ollama_client = Client(host='http://localhost:11434')

EMBED_MODEL ="voyage-code-3"
COLLECTION_NAME = "Voyageai_test"

def compute_voyageai_embedding(file_text, embed_model, output_dimension):
    key = f"{file_text}-{embed_model}-{output_dimension}"
    hashed_key = hashlib.sha256(key.encode()).hexdigest()
    if hashed_key in cache:
        return cache[hashed_key]
    query_embedding = vo.embed([file_text], model=embed_model, input_type="query", output_dimension=output_dimension).embeddings[0]
    cache[hashed_key] = query_embedding
    return query_embedding

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

def add_to_collection_using_embeddings(embeddings, file_path):
    try:
        client.insert(
            collection_name=COLLECTION_NAME,
            data={"id": file_path, "vector": embeddings}
        )
    except Exception as e:
        print(f"Error adding to collection: {e}")

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

def extract_ids(results):
    ids = []
    for res in results:
        ids.append(res['id'])
    return ids

if __name__=="__main__":
    path = sys.argv[1]
    directory = Path(path)
    file_extension = ".txt"
    
    for output_dimension in [256, 512, 1024, 2048]:
        print(f"[bold green]Running mAP tests for output dimension: {output_dimension}")
        
        # Check if the collection exists
        if any(col == COLLECTION_NAME for col in client.list_collections()):
            client.drop_collection(collection_name=COLLECTION_NAME)
            print(f"[yellow][-] Collection, '{COLLECTION_NAME}' existed and has been dropped.[/yellow]")
        
        # Create a fresh collection to test
        try:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                dimension=output_dimension,
                id_type="string",
                max_length=512
            )
            print(f"[yellow][-] New collection: {COLLECTION_NAME} created successfully.[/yellow]")
        except Exception as e:
            print(f"Error creating collection: {e}")

        progress_bar = tqdm(total=52000, desc="Building database")
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix == file_extension:
                try:
                    with open(file_path, 'r', encoding='iso8859-1') as f: 
                        file_text = f.read()
                    
                    start_time = time.time()
                    embeddings = compute_voyageai_embedding(file_text, EMBED_MODEL, output_dimension)
                    elapsed_time = time.time() - start_time

                    add_to_collection_using_embeddings(embeddings=embeddings, file_path=str(file_path))
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
            if file_path.is_file() and file_path.suffix == file_extension:
                try:
                    with open(file_path, 'r', encoding='iso8859-1') as f: 
                        file_text = f.read()

                    # Create embedding for text
                    start_time = time.time()
                    embeddings = compute_voyageai_embedding(file_text, EMBED_MODEL, output_dimension)
                    elapsed_time = time.time() - start_time
                    
                    for k in [10, 50, 100, 300]:
                        results = query_using_embeddings(embeddings=embeddings, k=k).pop()
                        result_paths = extract_ids(results)

                        total_ap[k] += average_precision(str(file_path), result_paths)
                        
                    progress_bar.set_postfix({"Embedding time": elapsed_time})
                    progress_bar.update(1)
                    queries += 1
                    
                except Exception as e:
                    print(f"Error testing collection: {e}")
        progress_bar.close()
        
        with open("metrics/precision/mean_ap_at_k_milvus_voyage_output_dimension.csv", 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            for k in [10,50,100,300]:
                mean_ap = total_ap[k] / queries
                print(f"mAP@{k}: {mean_ap}")
                csv_writer.writerow([EMBED_MODEL, output_dimension, k, mean_ap])
        
    print("[bold green]Indexing complete.[/bold green]")