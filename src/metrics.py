# pylint: skip-file

'''
This program loops through all .txt files in an input directory and it's
sub-directories and inputs their contents into the vector database. It measures
the execution time needed to make a query at each db size and writes the results 
to a generated csv file
'''

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
from pathlib import Path
from ollama import Client 
import tracemalloc

COLLECTION_NAME = "POJ_DATASET_memory_test"
EMBED_MODEL = "llama3.2"
COSINE_SEACH_N = 100

cache = dc.Cache('embedding_cache')
client = chromadb.HttpClient(host='localhost', port=8000)
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

def query(file_text):
    try:
        return collection.query(
            query_texts=[file_text],
            n_results=5,
        )

    except Exception as e:
        print(f"Error querying collection: {e}")

def query_using_embeddings(embeddings):
    try:
        return collection.query(
            query_embeddings=embeddings,
            n_results=5,
        )

    except Exception as e:
        print(f"Error querying collection: {e}")

def add_to_collection(file_text):
    try:
        collection.add(
            documents=[file_text],
            ids=[str(file_path)]
        )

    except Exception as e:
        print(f"Error querying collection: {e}")

def add_to_collection_using_embeddings(embeddings):
    try:
        collection.add(
            embeddings=embeddings,
            ids=[str(file_path)]
        )

    except Exception as e:
        print(f"Error querying collection: {e}")

if __name__=="__main__":
    path=sys.argv[1]
    for iteration in ["add","query"]:
        # Check if the collection exists
        if any(col.name == COLLECTION_NAME for col in client.list_collections()):
            client.delete_collection(COLLECTION_NAME)
            print(f"Collection, '{COLLECTION_NAME}' existed and has been dropped.")
        
        # Create a fresh collection to test
        try:
            collection = client.create_collection(
                name=COLLECTION_NAME,
                metadata={
                    # The number of neighbors to consider during search.
                    # The default is too low for deterministic results.
                    "hnsw:search_ef": COSINE_SEACH_N,
                    "hnsw:space": "cosine"
                },
            )
            print("Collection created successfully.")
        except Exception as e:
            print(f"Error creating collection: {e}")

        # Initialize CSV and write header to record results to
        timestamp = datetime.now().strftime("%Y:%m:%d")
        csv_filename = f'metrics/memory_usage/memory_usage_{timestamp}-{EMBED_MODEL.replace("/", "-")}-{iteration}.csv'
        with open(csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Database Size', 'File Path', 'Memory Usage (MB)'])

        # The first argument is a folder in which all snippets are stored.
        # Traverse the folder recursively and add all files to the collection, with the file path as the ID.
        db_size = 0
        directory = Path(path)
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix == '.txt':
                try:
                    print(f"[DB Size: {db_size + 1}] - Adding document: {file_path}")
                    with open(file_path, 'r', encoding='iso8859-1') as f: 
                        file_text = f.read()

                    # Create embedding for text
                    start_time = time.time()
                    embeddings = compute_embedding(file_text, EMBED_MODEL)
                    elapsed_time = time.time() - start_time
                    print(f"Embedding time: {elapsed_time}")

                    if iteration == "add":
                        result, memory_used = measure_memory_usage(add_to_collection_using_embeddings, embeddings)
                    else:
                        collection.add(
                            embeddings=embeddings,
                            ids=[str(file_path)]
                        )
                        
                        # Query the db for the document and time execution
                        #execution_time = timeit.timeit(lambda: query_using_embeddings(embeddings=embeddings), number=1)
                        result, memory_used = measure_memory_usage(query_using_embeddings, embeddings)

                    with open(csv_filename, 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow([db_size, file_path, memory_used])

                    db_size += 1

                except Exception as e:
                    print(f"Error: {e}")

        print(f"Collection size: {collection.count()}")
    print("Indexing complete.")
    