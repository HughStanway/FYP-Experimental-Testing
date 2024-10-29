import chromadb
import os
import sys

COLLECTION_NAME = "snippets_default_cosine"
TEST_QUERY_PATH = "test/empty.java"
INDEXING = True
QUERY = False

# Connect to the ChromaDB server
client = chromadb.HttpClient(host='localhost', port=8000)

# Create a collection to test
try:
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            # The number of neighbors to consider during search.
            # The default is too low for deterministic results.
            "hnsw:search_ef": 100,
            "hnsw:space": "cosine"
        },
    )
    print("Collection created successfully.")
except Exception as e:
    print(f"Error creating collection: {e}")


# Add an embedding to the collection
if INDEXING:
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

# Query the collection
if QUERY:
    try:
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

        print(f"\n######################## {TEST_QUERY_PATH} ############################")
        print(query_str)
        print(f"######################## {result} ############################")
        print(result_str)
        print("####################################################")
    except Exception as e:
        print(f"Error querying collection: {e}")
