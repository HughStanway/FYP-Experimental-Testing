import os
import sys
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DATA_PATH = "chroma_data/"
COLLECTION_NAME = "snippets_default_cosine"

# Open the ChromaDB client and the collection.
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

#client.delete_collection(name=COLLECTION_NAME)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={
        # The number of neighbors to consider during search.
        # The default is too low for deterministic results.
        "hnsw:search_ef": 100,
        "hnsw:space": "cosine"
    },
)

print("Indexing...")

# The first argument is a folder in which all snippets are stored.
# Traverse the folder recursively and add all files to the collection, with the file path as the ID.
path=sys.argv[1]
print(path)
for root, dirs, files in os.walk(path):
    for file in files:
        print(file)
        # Skip the README file as it is not a code snippet.
        if file == "README.md":
            continue
        with open(os.path.join(root, file), 'r') as f:
            print(f"Adding document: {os.path.join(root, file)}")
            try:
                # Add the document to the collection.
                collection.add(
                    documents=[f.read()],
                    ids=[os.path.join(root, file)]
                )
            except RuntimeError as e:
                print(f"Error: {e}")

print("Indexing complete.")

# print("Retrieve all ids.")
# data = collection.get(include=["embeddings"])
# ids = data["ids"]
# embeddings = data["embeddings"]
# for x in range(len(ids)):
#     print(ids[x])
#     print(embeddings[x])
# print("Retrieval complete.")
