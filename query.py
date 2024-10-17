import chromadb
import numpy as np

CHROMA_DATA_PATH = "chroma_data/"
COLLECTION_NAME = "snippets_default_cosine"
QUERY_PATH = "data/SeSaMeF/60138.java"
TEST_QUERY_PATH = "test/empty.java"

# Open the ChromaDB client and the collection.
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

with open(TEST_QUERY_PATH, 'r') as file:
    query_str = file.read()  # Read the entire file content

results = collection.query(
    query_texts=[query_str],
    n_results=5,
)

print(results)

# Get the first result
result = results['ids'][0][0]

with open(result, 'r') as file:
    result_str = file.read()

print(f"######################## {TEST_QUERY_PATH} ############################")
print(query_str)
print(f"######################## {result} ############################")
print(result_str)
print("####################################################")