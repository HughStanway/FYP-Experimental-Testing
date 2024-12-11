#pylint: skip-file

'''
Testing suit used to test the Milvus database.
'''

import time
import argparse
import os
import diskcache as dc
import chromadb
import hashlib
from pathlib import Path
from ollama import Client
from rich import print
from tqdm import tqdm

# Setup all program clients
os.environ["TOKENIZERS_PARALLELISM"] = "false"
cache = dc.Cache('embedding_cache')
client = chromadb.HttpClient(host='localhost', port=8000)
ollama_client = Client(host='http://localhost:11434')

# Define testing parameters usiing the constants below
COLLECTION_NAME = "test"
EMBED_MODEL = "ordis/jina-embeddings-v2-base-code"
CTX_WINDOW = [8192, 6144, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8]
K = 10

def compute_embedding(file_text, embed_model, ctx_window=None):
    key = f"{file_text}-{embed_model}-{ctx_window}" if ctx_window is not None else f"{file_text}-{embed_model}"
    hashed_key = hashlib.sha256(key.encode()).hexdigest()
    if hashed_key in cache:
        return cache[hashed_key]
    if ctx_window is None:
        embeddings = ollama_client.embed(model=embed_model, input=file_text)['embeddings']
    else:
        embeddings = ollama_client.embed(model=embed_model, input=file_text, options={"num_ctx": ctx_window})['embeddings']
    cache[hashed_key] = embeddings
    return embeddings

def prime_collection() -> None | chromadb.Collection:
    # Check if the collection exists
    if any(col.name == COLLECTION_NAME for col in client.list_collections()):
        client.delete_collection(COLLECTION_NAME)
        print(f"[yellow][-] Collection, '{COLLECTION_NAME}' existed and has been dropped.[/yellow]")
    
    # Create a fresh collection to test
    try:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:search_ef": 100,
                "hnsw:space": "cosine"
            },
        )
        print(f"[yellow][-] New collection: {COLLECTION_NAME} created successfully.[/yellow]")
        return collection
    except Exception as e:
        print(f"[red]Error creating collection: {e}[/red]")

def build_database(args, collection, ctx_window=None) -> None:
    print(f"[bold red][+] Building new chromadb database with dataset: {args.filepath}"
      f"{' for context window: ' + str(ctx_window) if ctx_window is not None else ''}")
    progress_bar = tqdm(total=52000, desc="Building database")
    for file_path in Path(args.filepath).rglob('*'):
        if file_path.is_file() and file_path.suffix == '.txt':
            try:
                with open(file_path, 'r', encoding='iso8859-1') as f:
                    file_text = f.read() 

                start_time = time.time()
                embeddings = compute_embedding(file_text, EMBED_MODEL, ctx_window)
                elapsed_time = time.time() - start_time

                collection.add(
                    embeddings=embeddings,
                    ids=[str(file_path)]
                )

                progress_bar.set_postfix({"Embedding time": elapsed_time})
                progress_bar.update(1)
            except Exception as e:
                print(f"[red]Error adding to collection: {e}[/red]") 
    progress_bar.close()

def run_tests(args, collection, ctx_window=None) -> None:
    progress_bar = tqdm(total=52000, desc="Running Tests")
    for file_path in Path(args.filepath).rglob('*'):
        if file_path.is_file() and file_path.suffix == '.txt':
            try:
                with open(file_path, 'r', encoding='iso8859-1') as f:
                    file_text = f.read() 

                start_time = time.time()
                embeddings = compute_embedding(file_text, EMBED_MODEL, ctx_window)
                elapsed_time = time.time() - start_time

                result = collection.query(
                    query_embeddings=embeddings,
                    n_results=K,
                )

                # Tests here
                if args.execution_time:
                    pass
                elif args.memory_usage:
                    pass
                elif args.map:
                    pass
                elif args.false_positives:
                    pass

                progress_bar.set_postfix({"Embedding time": elapsed_time})
                progress_bar.update(1)
            except Exception as e:
                print(f"[red]Error accessing to collection: {e}[/red]") 
    progress_bar.close()

def run(args):
    global CTX_WINDOW

    # Parse whether the program should run tests for ALL context windows
    if not (args.ctx and EMBED_MODEL == "ordis/jina-embeddings-v2-base-code"):
        CTX_WINDOW = None
    
    for ctx_window in CTX_WINDOW if CTX_WINDOW is not None else [None]:
        collection = prime_collection()
        build_database(args, collection, ctx_window)    
        run_tests(args, collection, ctx_window)

    

if __name__=="__main__":
    # Parse testing flags
    parser = argparse.ArgumentParser(description="Define testing parameters")
    parser.add_argument("filepath", type=str, help="Relative filepath to the dataset directory")
    parser.add_argument("--execution-time", action="store_true", help="Enables measuring execution time for query/insert", dest="execution_time")
    parser.add_argument("--memory-usage", action="store_true", help="Enables measuring memory usage for query/insert", dest="memory_usage")
    parser.add_argument("--map", action="store_true", help="Enables calculating mean average precision at k")
    parser.add_argument("--false-positives", action="store_true", help="Enables calculating mean average precision at k", dest="false_positives")
    parser.add_argument("--ctx", action="store_true", help="Enables tests for all context windows [Only works for ordis/jina-embeddings-v2-base-code model]")
    args = parser.parse_args()

    run(args=args)
    