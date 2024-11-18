import timeit
import os
import sys
import csv
import time
import argparse
from datetime import datetime
from pathlib import Path
import chromadb
from joblib import Memory
from datetime import datetime
from pathlib import Path
from ollama import Client 

os.environ["TOKENIZERS_PARALLELISM"] = "false"
memory = Memory("embedding_cache", verbose=1)

"""
Function used to compute new embeddings or retrieve from cache 
"""

@memory.cache
def compute_embedding(file_text, embed_model, ollama_client):
    embeddings = ollama_client.embed(model=embed_model, input=file_text)['embeddings']
    return embeddings

def get_cached_embedding(file_text, embed_model, ollama_client):
        return compute_embedding(file_text, embed_model, ollama_client)

class CsvWriter:
    def __init__(self, filepath_output):
        # Initialize CSV and write header to record results to
        timestamp = datetime.now().strftime("%Y:%m:%d_%H:%M")
        self.csv_filename = f'{filepath_output}_{timestamp}.csv'
        with open(self.csv_filename, 'w', newline='') as csvfile: # pylint: disable=all
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Database Size', 'File Path', 'Execution Time (seconds)'])

    def write_line(self, rows):
        with open(self.csv_filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            for row in rows:
                csv_writer.writerow(row)

class TestSuit:
    def __init__(self, csvwriter, collection_name, embed_model, format_collection = False):
        self.csvwriter = csvwriter
        self.collection_name = collection_name
        self.embed_model = embed_model

        self.client = chromadb.HttpClient(host='localhost', port=8000)
        self.ollama_client = Client(host='http://localhost:11434')

        if format_collection:
             # Check if the collection exists
            if any(col.name == self.collection_name for col in self.client.list_collections()):
                self.client.delete_collection(self.collection_name)
                print(f"Collection, '{self.collection_name}' existed and has been dropped.")
            
            # Create a fresh collection to test
            try:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={
                        # The number of neighbors to consider during search.
                        # The default is too low for deterministic results.
                        "hnsw:search_ef": 100,
                        "hnsw:space": "cosine"
                    },
                )
                print("Collection created successfully.")
            except Exception as e: # pylint: disable=all
                print(f"Error creating collection: {e}")
        else:
            self.collection = self.client.get_collection(name=self.collection_name)

        """
        Initialise some variables used during tests
        """
        self.misses = 0
    
    """
    Generaly functions used to add and query collections either via file text or embedings
    """
    
    def query(self, file_text):
        try:
            return self.collection.query(
                query_texts=[file_text],
                n_results=5,
            )

        except Exception as e:
            print(f"Error querying collection: {e}")

    def query_using_embeddings(self, embeddings):
        try:
            return self.collection.query(
                query_embeddings=embeddings,
                n_results=5,
            )

        except Exception as e:
            print(f"Error querying collection: {e}")

    def add_to_collection(self, file_text, file_path):
        try:
            self.collection.add(
                documents=[file_text],
                ids=[str(file_path)]
            )

        except Exception as e:
            print(f"Error querying collection: {e}")

    def add_to_collection_using_embeddings(self, embeddings, file_path):
        try:
            self.collection.add(
                embeddings=embeddings,
                ids=[str(file_path)]
            )

        except Exception as e:
            print(f"Error querying collection: {e}")

    def check_false_positive(self, query_path, result_paths, result_distances):
        min_distance = result_distances[0]
        min_distance_indices = [i for i, distance in enumerate(result_distances) if distance == min_distance]
        
        for index in min_distance_indices:
            if result_paths[index] == query_path:
                return True # No false positive, the query path is among the closest matches
        
        return False # False positive detected, the closest matches do not include the query path

    """
    These functions are called to run each type of test, they all take the file_path parameter
    """

    def false_positive_check(self, file_path):
        with open(file_path, 'r', encoding='iso8859-1') as f: 
            file_text = f.read()

        embeddings = get_cached_embedding(file_text, self.embed_model, self.ollama_client)
        result = self.query_using_embeddings(embeddings=embeddings)

        result_paths = result["ids"][0]
        result_distances =  result["distances"][0]

        if not self.check_false_positive(str(file_path), result_paths, result_distances):
            self.misses += 1

            self.csvwriter.write_line(
                [
                    [" "],
                    ["file: ", str(file_path)],
                    ["ids: ", result["ids"][0]],
                    ["distances: ", result["distances"][0]],

                ]
            )

def index(testsuit, data_directory):
    db_size = 0
    directory = Path(data_directory)
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix == '.txt':
            print(f"[DB Size: {db_size + 1}] - Document: {file_path}")

            # Modify below to call intended test function

            testsuit.false_positive_check(file_path)

            db_size += 1

    print(f"Misses: {testsuit.misses}")
    print("Indexing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection_name', type=str, help='Name of Chromadb collection', required=True)
    parser.add_argument('--embed_model', type=str, help='Embedding model to be used', required=True)
    parser.add_argument('--data', type=str, help='Path to data directory', required=True)
    parser.add_argument('--output', type=str, help='Path to the output directory', required=True)
    parser.add_argument('--f', action='store_true', help='If set, tells the TestSuit to format collection during initialisation')
    args = parser.parse_args()

    csvwriter = CsvWriter(args.output)
    testsuit = TestSuit(csvwriter, args.collection_name, args.embed_model, args.f)
    index(testsuit, args.data)