# pylint: skip-file

import voyageai
import hashlib
import csv
import diskcache as dc
import sys
from pathlib import Path
import time
from tqdm import tqdm
import chromadb
from rich import print
import datetime

vo = voyageai.Client()
cache = dc.Cache('embedding_cache_voyage')
client = chromadb.HttpClient(host='localhost', port=8000)

EMBED_MODEL = "voyage-code-3"
COLLECTION_NAME = "voyageai_test"


def compute_voyageai_embedding(file_text, embed_model):
    key = f"{file_text}-{embed_model}"
    hashed_key = hashlib.sha256(key.encode()).hexdigest()
    if hashed_key in cache:
        return None
        return cache[hashed_key]
    query_embedding = vo.embed([file_text], model=embed_model, input_type="query").embeddings[0]
    cache[hashed_key] = query_embedding
    return query_embedding

if __name__=="__main__":
    path=sys.argv[1]
    directory = Path(path)

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
                "hnsw:search_ef": 100,
                "hnsw:space": "cosine"
            },
        )
        print("Collection created successfully.")
    except Exception as e:
        print(f"Error creating collection: {e}")

    # Initialize CSV and write header to record results to
    timestamp = datetime.now().strftime("%Y:%m:%d")
    csv_filename = f'metrics/execution_times/execution_times_chroma_{timestamp}-{EMBED_MODEL.replace("/", "-")}-add.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Query','File Path','Memory Usage (MB)'])
    
    progress_bar = tqdm(total=52000, desc="Building database")
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix == '.txt':
            try:
                with open(file_path, 'r', encoding='iso8859-1') as f: 
                    file_text = f.read()

                start_time = time.time()
                embeddings = compute_voyageai_embedding(file_text, EMBED_MODEL)
                elapsed_time = time.time() - start_time

                if embeddings is not None:
                    print(embeddings)
                    print(elapsed_time)

                progress_bar.set_postfix({"Embedding time": elapsed_time})
                progress_bar.update(1)

            except Exception as e:
                print(f"Error adding to collection: {e}")
    progress_bar.close()