'''
This program loops through every file in the directory and queries against
the pre-populated vector database. It queries for the 10 closest results
and checks:

    - The first result should be the query text itself.
    - The next 9 results should be from the same POJ question.
'''

import os
import sys
import csv
import chromadb
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
COLLECTION_NAME = "POJ_DATASET"

client = chromadb.HttpClient(host='localhost', port=8000)
collection = client.get_collection(name=COLLECTION_NAME)

def check_match(query: list[str], res: list[str]) -> bool:
    query_question = query[1]
    res_question = res[1]
    return query_question == res_question

def run():
    path=sys.argv[1]
    hits, misses, times_not_returing_itself, counter = 0, 0, 0, 0
    directory = Path(path)
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix == '.txt':
            try:
                with open(file_path, 'r') as f:
                    file_text = f.read()  

                file_path_list = str(file_path).split("/")

                result = collection.query(
                    query_texts=[file_text],
                    include=["distances"],
                    n_results = 10
                )
                ids = result["ids"][0]
                ids_formatted = [id.split("/") for id in ids]

                first_id = ids_formatted[0]
                if not file_path_list == first_id:
                    times_not_returing_itself += 1

                for i in range(1, 10):
                    if check_match(file_path_list, ids_formatted[i]):
                        hits += 1
                    else:
                        misses += 1    

                counter += 1

                print(f"[DB Size: {counter}] - Hits: {hits}, Misses: {misses}, Times not returning itself: {times_not_returing_itself}")

            except Exception as e:
                print(f"Error: {e}")

    with open("metrics/false_positives.csv", 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["hits", "misses", "times not returing itself"])
        csv_writer.writerow([hits, misses, times_not_returing_itself])

if __name__=="__main__":
    run()