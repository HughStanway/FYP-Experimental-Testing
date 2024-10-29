'''
This program loops through all .txt files in an input directory and it's
sub-directories and inputs their contents into the vector database. It measures
the execution time needed to make a query at each db size and writes the results 
to a generated csv file
'''

import timeit
import chromadb
import os
import sys
import csv
from datetime import datetime
from pathlib import Path
import chromadb.utils.embedding_functions as embedding_functions

os.environ["TOKENIZERS_PARALLELISM"] = "false"
COLLECTION_NAME = "POJ_DATASET_ollama"
EMBED_MODEL = "llama3.2"

def query(path):
    try:
        with open(path, 'r') as file:
            query_str = file.read()  # Read the entire file content

        results = collection.query(
            query_texts=[query_str],
            n_results=5,
        )

    except Exception as e:
        print(f"Error querying collection: {e}")

if __name__=="__main__":
    # Connect to the ChromaDB server
    client = chromadb.HttpClient(host='localhost', port=8000)

    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name=EMBED_MODEL,
    )
    
    # Check if the collection exists
    if any(col.name == COLLECTION_NAME for col in client.list_collections()):
        client.delete_collection(COLLECTION_NAME)
        print(f"Collection, '{COLLECTION_NAME}' existed and has been dropped.")
    
    # Create a fresh collection to test
    try:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=ollama_ef,
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

    # Initialize CSV and write header
    timestamp = datetime.now().strftime("%Y:%m:%d_%H:%M")
    csv_filename = f'metrics/execution_times_{timestamp}.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Database Size', 'File Path', 'Execution Time (seconds)'])

    # The first argument is a folder in which all snippets are stored.
    # Traverse the folder recursively and add all files to the collection, with the file path as the ID.
    path=sys.argv[1]
    db_size = 0
    directory = Path(path)
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix == '.txt':
            try:
                print(f"[DB Size: {db_size + 1}] - Adding document: {file_path}")
                with open(file_path, 'r') as f:
                    file_text = f.read()

                    # Add the document to the collection.
                    collection.add(
                        documents=[file_text],
                        ids=[str(file_path)]
                    )

                    # Query the db for the document and time execution
                    execution_time = timeit.timeit(lambda: query(file_path), number=1)
                    db_size = db_size + 1

                    with open(csv_filename, 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow([db_size, file_path, execution_time])

            except Exception as e:
                print(f"Error: {e}")

    print("Indexing complete.")