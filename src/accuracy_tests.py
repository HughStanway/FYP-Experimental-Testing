#pylint: skip-file

# Import general libraries 
import argparse
import time
import os
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

    def calculate_average_precision(self, query_path: str, results: List[str]) -> float:
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

    def true_and_false_positives(self, query_path: str, result_paths: List[str]) -> Tuple[int, int]:
        query_question_number = query_path.split('/')[1]
        true_positives, false_positives = 0, 0
        for path in result_paths:
            result_question_number = path.split('/')[1]

            if result_question_number == query_question_number:
                true_positives += 1
            else:
                false_positives += 1

        return true_positives, false_positives
    
    def save_to_file(self, metrics: Dict[int, Results]) -> None:
        for metric in metrics.values():
            path = f"metrics/accuracy/{self.args.filepath}/{self.args.database}/"
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"Created directory: {path}")
            file_name = f"accuracy_results_FAISS_{self.args.embedding_model.replace("/", "-")}_@{metric.k}_and_{self.args.database}.txt"
            with open(path + file_name, "w") as f:
                f.write(f"Mean Average Precision (MAP) = {metric.get_map()}\n")
                f.write(f"True Positives = {metric.true_positives}\n")
                f.write(f"False Positives = {metric.false_positives}\n")
                f.write(f"Number of queries = {metric.number_of_queries}\n")
                f.write(f"Number of answers = {metric.number_of_answers}\n")
                f.write(f"Expected number of answers = {metric.expected_number_of_answers}\n")
                f.write(f"Precision = {metric.precision}")
        
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
        query_count = 0
        metrics = {
            10: Results(k=10),
            50: Results(k=50),
            100: Results(k=100),
            300: Results(k=300),

        }
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix == '.txt':
                try:
                    with open(file_path, 'r', encoding='iso8859-1') as f: 
                        file_text = f.read()

                    # Create embedding for text
                    start_time = time.time()
                    embeddings = self.compute_embedding(file_text)
                    elapsed_time = time.time() - start_time
                    
                    for k in [10,50,100,300]:
                        results = self.query_collection_using_embeddings(embeddings, k)
                        result_paths, _ = self.extract_distances_and_paths(results)
                        
                        average_precision = self.calculate_average_precision(str(file_path), result_paths)
                        true_positives, false_positives = self.true_and_false_positives(str(file_path), result_paths)

                        metrics[k].update(true_positives, false_positives, len(result_paths), average_precision)

                    query_count += 1
                    progress_bar.set_postfix({"Embedding time": elapsed_time})
                    progress_bar.update(1)

                except Exception as e:
                    print(f"Error querying collection: {e}")
        progress_bar.close()

        self.save_to_file(metrics)

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

    args = parser.parse_args()
    AccuracyTests(args=args).run()