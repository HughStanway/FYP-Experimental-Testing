import os
import pprint
import sys
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DATA_PATH = "chroma_data/"
COLLECTION_NAME = "snippets_default_cosine"

# Open the ChromaDB client and the collection.
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
collection = client.get_collection(name=COLLECTION_NAME)
#default_ef = embedding_functions.DefaultEmbeddingFunction()

print("Retrieval")
print("Test,fileA,fileB")

# Traverse the folder recursively and use the files to retrieve all matching ids from the collection.
path=sys.argv[1]
threshold = float(sys.argv[2])/100

for root, dirs, files in os.walk(path):
    for file in files:
        # Skip the README file as it is not a code snippet.
        if file == "README.md":
            continue
        with open(os.path.join(root, file), 'r') as f:
            text = f.read()
            n = 1
            while n != 0:
                #q_embedding = default_ef(text)[0]
                print(f"Retrieving document: {os.path.join(root, file)} [{n}]")
                ##print(q_embedding)
                result = collection.query(
                    query_texts=[text],
                    #query_embeddings=[q_embedding],
                    include=["documents", "embeddings", "distances"],
                    n_results = n*10
                )
                ids = result["ids"][0]

                n += 1

                distances = result["distances"][0]
                embeddings = result["embeddings"][0]

                # Print the IDs and distances of the retrieved snippets.
                # Only print the ones where is distance is less than the threshold.
                for i, id in enumerate(ids):
                    distance = distances[i]
                    embedding = embeddings[i]
                    # Assert that the first ID matches the file.
                    if i == 0 and distance > 0.001 and id != os.path.join(root, file):
                        print("Error: first ID does not match file!")
                        print(embedding)

                    # Have we met the threshold?
                    if distance < threshold:
                        print(f"#{i}: ID: {id}, Distance: {distance}")
                        print(f"T,{os.path.join(root, file)},{id},{distance}")
                    else:
                        # No need to continue search, threshold met
                        print(f"-{i}: ID: {id}, Distance: {distance}")
                        n = 0
                        break

