#pylint: skip-file

# Import general libraries 
import argparse
import time
import csv
import tracemalloc
import datetime
from rich import print
from tqdm import tqdm
from pathlib import Path

# Import needed classes
from shared.testsuit import TestSuit

class ExecutionTests(TestSuit):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        
        # Initialize CSV and write header to record results to
        timestamp = datetime.now().strftime("%Y:%m:%d")
        self.csv_filename = f'metrics/memory_usage/memory_usage_{self.args.database}_{timestamp}-{self.args.embedding_model.replace("/", "-")}-{self.args.operation}.csv'
        with open(self.csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Database Size', 'File Path', 'Memory Usage (bytes)'])

    def save_to_csv(self, db_size: int, file_path: str, memory_usage: float) -> None:
        with open(self.csv_filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([db_size, file_path, memory_usage])

    def measure_memory_usage(func, *args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory() # in bytes
        tracemalloc.stop()

        return result, peak
    
    def run(self) -> None:
        directory: Path = Path(self.args.filepath)

        db_size = 0
        progress_bar = tqdm(total=52000, desc="memory usage test")
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix == '.txt':
                try:
                    with open(file_path, 'r', encoding='iso8859-1') as f: 
                        file_text = f.read()

                    # Create embedding for text
                    start_time = time.time()
                    embeddings = self.compute_embedding(file_text)
                    elapsed_time = time.time() - start_time
                    
                    # Test execution time
                    if self.args.operation == "add":
                        _, memory_usage = self.measure_memory_usage(self.add_to_collection_using_embeddings, embeddings, str(file_path))
                    else:
                        self.add_to_collection_using_embeddings(embeddings, str(file_path), db_size)
                        _, memory_usage = self.measure_memory_usage(self.query_collection_using_embeddings, embeddings, 10)

                    db_size += 1
                    self.save_to_csv(db_size, str(file_path), memory_usage)

                    progress_bar.set_postfix({"Embedding time": elapsed_time})
                    progress_bar.update(1)

                except Exception as e:
                    print(f"Error: {e}")
        progress_bar.close()

        if self.args.database == "weaviate":
            self.database.client.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Define testing parameters")
    parser.add_argument("filepath", type=str, help="Relative filepath to the dataset directory")
    parser.add_argument(
        "--embedding-model",
        choices=["llama3.2", "ordis/jina-embeddings-v2-base-code", "voyage-code-3"],
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
        "--operation",
        choices=["add", "query"],
        required=True,
        help="Specify whether to test insert or query operation",
        dest="operation"
    )
    args = parser.parse_args()
    ExecutionTests(args=args).run()