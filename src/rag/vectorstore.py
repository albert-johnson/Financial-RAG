from __future__ import annotations

import os
from typing import List, Dict, Optional, Any

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

from src.config import (
    VECTOR_DIR,
    VECTOR_COLLECTION_DOCS,
    VECTOR_COLLECTION_TABLES,
    EMBEDDING_MODEL_NAME,
)


class VectorStoreManager:
    def __init__(self, persist_directory: str | os.PathLike = VECTOR_DIR) -> None:
        self.persist_directory = str(persist_directory)
        # Prefer OpenAI embeddings when available; fallback to local
        if os.getenv("OPENAI_API_KEY"):
            self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        else:
            self._embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self._docs_store = Chroma(
            collection_name=VECTOR_COLLECTION_DOCS,
            embedding_function=self._embeddings,
            persist_directory=self.persist_directory,
        )
        self._tables_store = Chroma(
            collection_name=VECTOR_COLLECTION_TABLES,
            embedding_function=self._embeddings,
            persist_directory=self.persist_directory,
        )

    # Documents
    def add_documents(self, documents: List[Document]) -> List[str]:
        ids = [doc.metadata.get("doc_chunk_id") for doc in documents]
        return self._docs_store.add_documents(documents=documents, ids=ids)

    def delete_documents_by_source(self, source_path: str) -> None:
        self._docs_store.delete(where={"source": source_path})

    def similarity_search(self, query: str, k: int = 6) -> List[Document]:
        return self._docs_store.similarity_search(query, k=k)

    # Tables
    def add_table_descriptions(self, documents: List[Document]) -> List[str]:
        ids = [doc.metadata.get("table_id") for doc in documents]
        return self._tables_store.add_documents(documents=documents, ids=ids)

    def delete_tables_by_source(self, source_path: str) -> None:
        self._tables_store.delete(where={"source": source_path})

    def search_tables(self, query: str, k: int = 4) -> List[Document]:
        return self._tables_store.similarity_search(query, k=k)

    def persist(self) -> None:
        # Chroma persists automatically on add/delete; explicit persist for safety
        self._docs_store.persist()
        self._tables_store.persist()