from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from config.settings import settings
from llm.openai_llm import OPENAI_API_KEY
import logging

logger = logging.getLogger(__name__)

class RetrieverBuilder:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            api_key=OPENAI_API_KEY,
            model="text-embedding-3-large",
        )

    def build_hybrid_retriever(self, docs, *, collection_name: str | None = None):
        """
        Build a hybrid retriever using BM25 and vector-based retrieval.
        IMPORTANT:
        - Use a unique collection_name per doc to prevent cross-document retrieval.
        """
        try:
            collection = collection_name or "default"

            vector_store = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=settings.CHROMA_DB_PATH,
                collection_name=collection,
            )
            logger.info(f"Vector store ready (collection='{collection}').")

            bm25 = BM25Retriever.from_documents(docs)
            logger.info("BM25 retriever created successfully.")

            vector_retriever = vector_store.as_retriever(
                search_kwargs={"k": settings.VECTOR_SEARCH_K}
            )
            logger.info("Vector retriever created successfully.")

            hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25, vector_retriever],
                weights=settings.HYBRID_RETRIEVER_WEIGHTS,
            )
            logger.info("Hybrid retriever created successfully.")
            return hybrid_retriever

        except Exception as e:
            logger.error(f"Failed to build hybrid retriever: {e}")
            raise
