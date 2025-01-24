#pylint: skip-file

# Import general libraries 
import argparse
import sys
import hashlib
import diskcache as dc
from rich import print
from abc import ABC, abstractmethod
from typing import List, Tuple

# Import vecotr database libraries/ apis
import voyageai
import chromadb
import weaviate
import voyageai
import weaviate.classes as wvc
from ollama import Client 
from pymilvus import MilvusClient
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Import needed classes
from dataclass.database import Database

"""
Initialise ollama client, voyage client and caches
"""
cache = dc.Cache('embedding_cache')
cache_voyage = dc.Cache('embedding_cache_voyage')
ollama_client = Client(host='http://localhost:11434')
vo = voyageai.Client()

class TestSuit(ABC):
    COLLECTION_NAME = "Precision_test"
    DIMENSIONS = {
        "llama3.2": 3072,
        "ordis/jina-embeddings-v2-base-code": 768,
        "voyage-code-3": 1024,
        "deepseek-r1:1.5B": 1536,
    }

    def __init__(self, args: argparse.Namespace) -> None:
        self.args: argparse.Namespace = args
        self.database: Database = self.init_database()

    def init_chromadb(self) -> Database:
        try:
            client = chromadb.HttpClient(host='localhost', port=8000)

            # Check if the collection exists
            if any(col == TestSuit.COLLECTION_NAME for col in client.list_collections()):
                client.delete_collection(TestSuit.COLLECTION_NAME)
                print(f"[yellow][-] Collection, '{TestSuit.COLLECTION_NAME}' existed and has been dropped.[/yellow]")
        
            # Create a fresh collection to test
            collection = client.create_collection(
                name=TestSuit.COLLECTION_NAME,
                metadata={
                    "hnsw:search_ef": 100,
                    "hnsw:space": "cosine"
                },
            )
            print(f"[yellow][-] New collection: {TestSuit.COLLECTION_NAME} created successfully.[/yellow]")
            return Database(name=self.args.database, client=client, collection=collection)

        except Exception as e:
            print(f"Error creating collection: {e}")
            sys.exit("Exiting the program due to error.")

    def init_milvus(self) -> Database:
        try:
            client = MilvusClient(uri="http://localhost:19530")

            # Check if the collection exists
            if any(col == TestSuit.COLLECTION_NAME for col in client.list_collections()):
                client.drop_collection(collection_name=TestSuit.COLLECTION_NAME)
                print(f"[yellow][-] Collection, '{TestSuit.COLLECTION_NAME}' existed and has been dropped.[/yellow]")
            
            # Create a fresh collection to test             
            client.create_collection(
                collection_name=self.COLLECTION_NAME,
                dimension=TestSuit.DIMENSIONS[self.args.embedding_model],
                id_type="string",
                max_length=512
            )
            print(f"[yellow][-] New collection: {TestSuit.COLLECTION_NAME} created successfully.[/yellow]")
            return Database(name=self.args.database, client=client)
        
        except Exception as e:
            print(f"Error creating collection: {e}")
            sys.exit("Exiting the program due to error.")

    def init_weaviate(self) -> Database:
        try:
            client = weaviate.connect_to_local()

            # Check if the collection exists
            if any(col == TestSuit.COLLECTION_NAME for col in client.collections.list_all().keys()):
                client.collections.delete(name=TestSuit.COLLECTION_NAME)
                print(f"[yellow][-] Collection, '{TestSuit.COLLECTION_NAME}' existed and has been dropped.[/yellow]")
        
            # Create a fresh collection to test
            collection = client.collections.create(
                TestSuit.COLLECTION_NAME,
                vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            )
            print(f"[yellow][-] New collection: {TestSuit.COLLECTION_NAME} created successfully.[/yellow]")
            return Database(name=self.args.database, client=client, collection=collection)
        
        except Exception as e:
            print(f"[bold red]Error during collection setup: {e}[/bold red]")
            sys.exit("Exiting the program due to error.")

    def init_qdrant(self) -> Database:
        try:
            client = QdrantClient(url="http://localhost:6333")

            # Delete collection if it already exists
            if client.collection_exists(TestSuit.COLLECTION_NAME):
                client.delete_collection(TestSuit.COLLECTION_NAME)
                print(f"[yellow]Collection, '{TestSuit.COLLECTION_NAME}' existed and has been dropped.[/yellow]")

            # Create a fresh collection to test
            client.create_collection(
            collection_name=TestSuit.COLLECTION_NAME,
            vectors_config=VectorParams(size=TestSuit.DIMENSIONS[self.args.embedding_model], distance=Distance.COSINE),
            )
            print("[yellow]Collection created successfully.[/yellow]")
            return Database(name=self.args.database, client=client)

        except Exception as e:
            print(f"[red]Error creating collection: {e}[/red]")
            sys.exit("Exiting the program due to error.")

    def init_database(self) -> Database:
        database_type = self.args.database
        if database_type == "chroma":
            return self.init_chromadb()
        elif database_type == "milvus":
            return self.init_milvus()
        elif database_type == "weaviate":
            return self.init_weaviate()
        elif database_type == "qdrant":
            return self.init_qdrant()
        else:
            raise ValueError(f"Database type: {database_type} is not supported")
        
    def compute_embedding(self, file_text: str) -> list[float]:
        key = f"{file_text}-{self.args.embedding_model}"
        hashed_key = hashlib.sha256(key.encode()).hexdigest()

        if self.args.embedding_model != "voyage-code-3" and hashed_key in cache:
            return cache[hashed_key].pop()
        
        if self.args.embedding_model == "voyage-code-3" and hashed_key in cache_voyage:
            return cache_voyage[hashed_key]
        
        if self.args.embedding_model == "voyage-code-3":
            embeddings = vo.embed([file_text], model=self.args.embedding_model, input_type="query").embeddings[0]
            cache_voyage[hashed_key] = embeddings
        else:
            embeddings = ollama_client.embed(model=self.args.embedding_model, input=file_text)['embeddings'].pop()
            cache[hashed_key] = embeddings

        return embeddings

    def add_to_collection_using_embeddings(self,
                                           embeddings: list[int],
                                           file_path: str,
                                           counter: int) -> None:
        database_type = self.database.name
        if database_type == "chroma":
            self.database.collection.add(
                embeddings=[embeddings],
                ids=[file_path]
            )
        elif database_type == "milvus":
            self.database.client.insert(
                collection_name=TestSuit.COLLECTION_NAME,
                data={"id": str(file_path), "vector": embeddings}
            )
        elif database_type == "weaviate":
            self.database.collection.data.insert(
                properties={
                    "file_path": file_path,
                },
                vector=embeddings
            )
        elif database_type == "qdrant":
            self.database.client.upsert(
            collection_name=TestSuit.COLLECTION_NAME,
            points=[
                    PointStruct(
                            id=counter,
                            vector=embeddings,
                            payload={"file_path": file_path}
                    )
                    
                ]
            )
        else:
            raise ValueError(f"Database type: {database_type} is not supported")
        
    def query_collection_using_embeddings(self, 
                                          embeddings: list[float],
                                          k: int):
        database_type = self.database.name
        if database_type == "chroma":
            return self.database.collection.query(
                query_embeddings=[embeddings],
                n_results=k,
            )
        elif database_type == "milvus":
            return self.database.client.search(
                collection_name=TestSuit.COLLECTION_NAME,
                data=[embeddings],
                limit=k,
                search_params={"metric_type": "COSINE", "params": {}},
            )
        elif database_type == "weaviate":
            return self.database.collection.query.near_vector(
                near_vector=embeddings,
                limit=k,
                return_metadata=wvc.query.MetadataQuery(certainty=True)
            )
        elif database_type == "qdrant":
            return self.database.client.search(
                collection_name=TestSuit.COLLECTION_NAME,
                query_vector=embeddings,
                limit=k
            )
        else:
            raise ValueError(f"Database type: {database_type} is not supported")
        
    def extract_distances_and_paths(self, results) -> Tuple[List[float], List[str]]:
        database_type = self.database.name
        if database_type == "chroma":
            return results["ids"][0], results["distances"][0]
        elif database_type == "milvus":
            extracted_results = results.pop()
            result_paths, result_distances = zip(*((result['id'], result['distance']) for result in extracted_results))
            return list(result_paths), list(result_distances)
        elif database_type == "weaviate":
            result_paths, result_distances = zip(*((obj.properties['file_path'], obj.metadata.certainty) for obj in results.objects))
            return list(result_paths), list(result_distances)
        elif database_type == "qdrant":
            result_paths, result_distances = zip(*((point.payload['file_path'], point.score) for point in results))
            return list(result_paths), list(result_distances)
        else:
            raise ValueError(f"Database type: {database_type} is not supported")

    @abstractmethod
    def run(self) -> None:
        pass
