import os
import hashlib

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
import chromadb
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "all-MiniLM-L6-v2")
embedding_model = SentenceTransformer(MODEL_PATH)

file_path = Path(__file__).resolve()
db_parent_dir = file_path.parents[2]
db_path = os.path.join(db_parent_dir, "chroma_db")

class LocalSentenceTransformerEmbeddingFunction(embedding_functions.EmbeddingFunction):

    def __init__(self, model: SentenceTransformer):
        self.model = model

    def __call__(self, input):
        embeddings = self.model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()


class VectorDBManager:
    def __init__(
        self,
        collection_name: str = "my_collection",
        persist_directory: str = db_path,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        self.embedding_function = LocalSentenceTransformerEmbeddingFunction(embedding_model)

        self.client = None
        self.collection = None
        self._prepare_chroma_db()

    def _prepare_chroma_db(self):
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
        )

    @staticmethod
    def _hash_document(document: str, user_id: str = None) -> str:
        key = f"{user_id or ''}:{document.strip()}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _document_exists(self, doc_id: str) -> bool:
        existing = self.collection.get(ids=[doc_id])
        return len(existing.get("ids", [])) > 0

    def save_data(self, document: str, user_id: str = None):
        content_hash = self._hash_document(document, user_id)

        id = content_hash

        if self._document_exists(id):
            return None

        metadata = {"user_id": user_id} if user_id else None

        self.collection.add(
            documents=[document],
            ids=[id],
            metadatas=[metadata] if metadata else None,
        )
        return id

    def retrieve_data(self, query: str, n_results: int = 5, user_id: str = None):
        where = {"user_id": user_id} if user_id else None
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
        )
        return results