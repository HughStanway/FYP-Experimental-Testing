#pylint: skip-file

# Import general libraries 
import argparse
import time
import json
from rich import print
from typing import List, Tuple, Dict
from tqdm import tqdm
from pathlib import Path

# Import needed classes
from dataclass.results import Results
from shared.testsuit import TestSuit

class AccuracyTests(TestSuit):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.args = args
        self.filename = f'metrics/false_positives/false_positives_@{self.args.k}_{self.args.database}-{self.args.embedding_model.replace("/", "-")}.txt'
        self.false_negative_count = 0

    def save_false_negative_to_file(self, query_path, ids, distances) -> None:
        with open(self.filename, "a") as f:
            f.write(f"{query_path}\n")
            f.write(f"{ids}\n")
            f.write(f"{distances}\n\n")

    def save_final_result_to_file(self, num_false_negatives) -> None:
        with open(self.filename, "a") as f:
            f.write(f"{num_false_negatives}\n\n")

    def check_false_negative_chroma(self, query_path, distances, ids):
        min_distance = distances[0]
        min_distance_indices = [i for i, result in enumerate(distances) if result == min_distance]

        for index in min_distance_indices:

            if ids[index] == query_path:
                return True  # No false positive, the query path is among the closest matches

        return False  # False positive detected, the closest matches do not include the query path
    
    def check_false_negative_milvus(self, query_path, results):
        min_distance = results[0]['distance']
        min_distance_indices = [i for i, result in enumerate(results) if result['distance'] == min_distance]

        for index in min_distance_indices:
            if results[index]['id'] == query_path:
                return True  # No false positive, the query path is among the closest matches
        
        return False  # False positive detected, the closest matches do not include the query path
    
    def extract_results_milvus(self, results):
        distances, ids = [], []
        for result in results:
            distances.append(result['distance'])
            ids.append(result['id'])
        return distances, ids
    
    def extract_results_weaviate(self, results):
        distances, ids = [], []
        for result in results.objects:
            distances.append(result.metadata.certainty)
            ids.append(result.properties["file_path"])
        return distances, ids
    
    def extract_results_qdrant(self, results):
        distances, ids = [], []
        for result in results:
            ids.append(result.payload["file_path"])
            distances.append(result.score)
        return distances, ids
    
    def false_negative_wrapper(self, query_path, results):
        database_type = self.args.database
        if database_type == "chroma":
            distances = results['distances'][0]
            ids = results['ids'][0]

            if not self.check_false_negative_chroma(query_path, distances, ids):
                self.false_negative_count += 1
                self.save_false_negative_to_file(query_path, ids, distances)

        elif database_type == "milvus":
            if not self.check_false_negative_milvus(query_path, results[0]):
                self.false_negative_count += 1
                ids, distances = self.extract_results_milvus(results[0])
                self.save_false_negative_to_file(query_path, ids, distances)

        elif database_type == "weaviate":
            distances, ids = self.extract_results_weaviate(results)

            if not self.check_false_negative_chroma(query_path, distances, ids):
                self.false_negative_count += 1
                self.save_false_negative_to_file(query_path, ids, distances)

        elif database_type == "qdrant":
            distances, ids = self.extract_results_qdrant(results)

            if not self.check_false_negative_chroma(query_path, distances, ids):
                self.false_negative_count += 1
                self.save_false_negative_to_file(query_path, ids, distances)

        else:
            raise ValueError(f"Database type: {database_type} is not supported")
        
    def run(self) -> None:
        directory: Path = Path(self.args.filepath)

        progress_bar = tqdm(total=52000, desc="Building database")
        insert_count = 0
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix == '.txt':
                try:
                    with open(file_path, 'r', encoding='iso8859-1') as f: 
                        file_text = f.read()

                    # Create embedding for text
                    start_time = time.time()
                    embeddings = self.compute_embedding(file_text)
                    elapsed_time = time.time() - start_time
                    
                    # Insert into collection
                    self.add_to_collection_using_embeddings(embeddings, str(file_path), insert_count)
                    
                    insert_count += 1
                    progress_bar.set_postfix({"Embedding time": elapsed_time})
                    progress_bar.update(1)
                    
                except Exception as e:
                    print(f"Error adding to collection: {e}")
        progress_bar.close()
        
        progress_bar = tqdm(total=52000, desc="Testing database")
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix == '.txt':
                try:
                    with open(file_path, 'r', encoding='iso8859-1') as f: 
                        file_text = f.read()

                    # Create embedding for text
                    start_time = time.time()
                    embeddings = self.compute_embedding(file_text)
                    elapsed_time = time.time() - start_time

                    results = self.query_collection_using_embeddings(embeddings, self.args.k)
                    self.false_negative_wrapper(str(file_path), results)

                    progress_bar.set_postfix({"Embedding time": elapsed_time})
                    progress_bar.update(1)

                except Exception as e:
                    print(f"Error querying collection: {e}")
        progress_bar.close()

        self.save_final_result_to_file(self.false_negative_count)

        if self.args.database == "weaviate":
            self.database.client.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Define testing parameters")
    parser.add_argument("filepath", type=str, help="Relative filepath to the dataset directory")
    parser.add_argument(
        "--embedding-model",
        choices=["llama3.2", "ordis/jina-embeddings-v2-base-code", "voyage-code-3", "deepseek-r1:1.5B"],
        required=True,
        help="Specify the name of the embedding model",
        dest="embedding_model"
    )
    parser.add_argument(
        "--database",
        choices=["chroma", "milvus", "weaviate", "qdrant"],
        required=True,
        help="Specify the name of the vector database",
        dest="database"
    )

    parser.add_argument(
        "--k",
        type=int,
        choices=[10, 300],
        required=True,
        help="Specify the query rank, k",
        dest="k"
    )

    args = parser.parse_args()
    AccuracyTests(args=args).run()