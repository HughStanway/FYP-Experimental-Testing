import timeit
import logging
import chromadb
import os
import sys
import csv
from datetime import datetime
from memory_profiler import memory_usage

os.environ["TOKENIZERS_PARALLELISM"] = "false"
COLLECTION_NAME = "metrics_test"

def query(path):
    try:
        with open(path, 'r') as file:
            query_str = file.read()  # Read the entire file content

        results = collection.query(
            query_texts=[query_str],
            n_results=5,
        )

        # Get the results
        print(results['ids'])

    except Exception as e:
        _logger.error(f"Error querying collection: {e}")

def measure_performance(func, *args, **kwargs):
    mem_usage, result = memory_usage((func, args, kwargs), interval=0.1, retval=True)
    max_mem_used = max(mem_usage)
    exec_time = timeit.timeit(lambda: func(*args, **kwargs), number=1)
    
    return exec_time, max_mem_used


if __name__=="__main__":
    _logger = logging.getLogger()

    # Connect to the ChromaDB server
    client = chromadb.HttpClient(host='localhost', port=8000)

    # Check if the collection exists
    if any(col.name == COLLECTION_NAME for col in client.list_collections()):
        client.delete_collection(COLLECTION_NAME)
        _logger.info(f"Collection, '{COLLECTION_NAME}' existed and has been dropped.")

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
        _logger.info("Collection created successfully.")
    except Exception as e:
        _logger.error(f"Error creating collection: {e}")

    # Initialize CSV and write header
    timestamp = datetime.now().strftime("%Y:%m:%d_%H:%M")
    csv_filename = f'metrics/execution_times_{timestamp}.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Database Size', 'File Path', 'Execution Time (seconds)', 'Memory Usage (MiB)'])

    # The first argument is a folder in which all snippets are stored.
    # Traverse the folder recursively and add all files to the collection, with the file path as the ID.
    path=sys.argv[1]
    db_size = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            # Skip the README file as it is not a code snippet.
            if file == "README.md":
                continue
            with open(os.path.join(root, file), 'r') as f:
                _logger.info(f"Adding document: {os.path.join(root, file)}")
                try:
                    file_text = f.read()
                    file_path = os.path.join(root, file)

                    # Add the document to the collection.
                    collection.add(
                        documents=[file_text],
                        ids=[file_path]
                    )

                    # Query the db for the document and time execution
                    execution_time, memory_usage_mib = measure_performance(query, file_path)

                    db_size = db_size + 1

                    with open(csv_filename, 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow([db_size, file_path, execution_time, memory_usage_mib])

                except RuntimeError as e:
                    _logger.error(f"Error: {e}")

    _logger.info("Indexing complete.")