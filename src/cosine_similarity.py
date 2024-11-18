# pylint: skip-file

import chromadb
from ollama import Client
import numpy as np

# Parameters
file_path = 'ProgramData/5/63.txt'
model_name = "llama3.2"

COLLECTION_NAME = "POJ_DATASET_ollama_embedding"
client = chromadb.HttpClient(host='localhost', port=8000)
collection = client.get_collection(name=COLLECTION_NAME)

# Read the file content
with open(file_path, "r") as file:
    text_content = file.read()

# Generate embedding for file text
ollama_client = Client(host='http://localhost:11434')
embedding = ollama_client.embed(model=model_name, input=text_content)['embeddings']

# Add embedding to collection
try:
    collection.add(
        embeddings=embedding,
        ids=[file_path]
    )
    print("Added to collection")

    doc_count = collection.count()
    print(f"Total documents in collection after add: {doc_count}")

except Exception as e:
    print(f"Error adding to collection: {e}")


try:
    results = collection.query(
        query_embeddings=embedding,
        n_results=5,
    )
except Exception as e:
        print(f"Error querying collection: {e}")

result_paths = results["ids"][0]
result_distances =  results["distances"][0]

print(f"Query path: {file_path}")
print(f"Result paths: {result_paths}")
print(f"Result distances: {result_distances}")
'''
# Retrieve the stored embedding
try:
    result = collection.get(ids=[file_path])
    print(result)
    if 'embeddings' in result and result['embeddings']:
        stored_embedding = np.array(result['embeddings'][0])  # Convert to numpy array
        print("Successfully retrieved stored embedding.")
    else:
        print("No embedding found in the collection for this file path.")
        stored_embedding = None
except Exception as e:
    print(f"Error retrieving stored embedding: {e}")

# Calculate cosine similarity if stored embedding is found
if stored_embedding is not None:
    norm_new = np.array(embedding) / np.linalg.norm(embedding)
    norm_stored = stored_embedding / np.linalg.norm(stored_embedding)

    cosine_similarity = np.dot(norm_new, norm_stored)
    print(f"Cosine similarity between stored and new embedding: {cosine_similarity}")
else:
    print("Could not retrieve the stored embedding for comparison.")
'''