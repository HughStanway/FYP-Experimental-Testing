#pylint: skip-file

# Import general libraries 
import argparse
import time
import csv
import os
from datetime import datetime
from rich import print
from tqdm import tqdm
from pathlib import Path

# Import needed classes
from shared.testsuit import TestSuit

class StorageTests(TestSuit):
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        super().__init__(args)
        
        # Initialize CSV and write header to record results to
        self.csv_filename = f'metrics/storage_tests/storage_tests_{self.args.database}-{self.args.embedding_model.replace("/", "-")}.csv'
        with open(self.csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Database Size', 'File Path', 'Storage Size'])

    def save_to_csv(self, db_size: int, file_path: str, storage_size: float) -> None:
        with open(self.csv_filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([db_size, file_path, storage_size])

    def get_directory_size(self, path):
        total_size = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size

    def measure_storage_size(self):
        database_type = self.args.database
        if database_type == "chroma":
            return self.get_directory_size("chroma_db")
        elif database_type == "milvus":
            return self.get_directory_size("../milvus/volumes/milvus")
        elif database_type == "weaviate":
            return self.get_directory_size("../weaviate/weaviate_storage/precision_test")
        elif database_type == "qdrant":
            return self.get_directory_size("../test5/qdrant_storage/collections")
        else:
            raise ValueError(f"Database type: {database_type} is not supported")

    def run(self) -> None:
        directory: Path = Path(self.args.filepath)

        self.save_to_csv(0, "", self.measure_storage_size())
        db_size = 0
        progress_bar = tqdm(total=52000, desc="Storage Space tests")
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix == '.txt':
                try:
                    with open(file_path, 'r', encoding='iso8859-1') as f: 
                        file_text = f.read()

                    # Create embedding for text
                    start_time = time.time()
                    embeddings = self.compute_embedding(file_text)
                    elapsed_time = time.time() - start_time

                    # Add to collection
                    self.add_to_collection_using_embeddings(embeddings, str(file_path), db_size)
                    storage_size = self.measure_storage_size()

                    db_size += 1
                    self.save_to_csv(db_size, str(file_path), storage_size)

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
    StorageTests(args=args).run()