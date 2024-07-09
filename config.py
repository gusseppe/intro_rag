import os
from dotenv import load_dotenv

load_dotenv('rag.env')

CONFIG = {
    "PERSIST_DIRECTORY": os.getenv("PERSIST_DIRECTORY", "chroma_db_index"),
    "DOCS_DIRECTORY": os.getenv("DOCS_DIRECTORY", "./sagemaker_documentation"),
    "API": os.getenv("API", "env"),
    "LC_CACHE_PATH": os.getenv("LC_CACHE_PATH", ".langchain.db"),
    "LLM_NAME": os.getenv("LLM_NAME", "llama3-8b-8192"),
    "LLM_TEMPERATURE": float(os.getenv("LLM_TEMPERATURE", 0.0)),
    "EMBEDDING_MODEL_NAME": os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en"),
    "EMBEDDING_MODEL_KWARGS": {"device": os.getenv("EMBEDDING_MODEL_DEVICE", "cpu")},
    "EMBEDDING_ENCODE_KWARGS": {"normalize_embeddings": True},
    "CHUNK_SIZE": int(os.getenv("CHUNK_SIZE", 1000)),
    "CHUNK_OVERLAP": int(os.getenv("CHUNK_OVERLAP", 200)),
    "DOCS_GLOB_PATTERN": os.getenv("DOCS_GLOB_PATTERN", "**/*.md"),
    "RETRIEVER_SEARCH_TYPE": os.getenv("RETRIEVER_SEARCH_TYPE", "mmr"),
    "RETRIEVER_SEARCH_K": int(os.getenv("RETRIEVER_SEARCH_K", 6)),
    "RETRIEVER_LAMBDA_MULT": float(os.getenv("RETRIEVER_LAMBDA_MULT", 0.25)),
    "RERANKER_MODEL_NAME": os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base"),
    "RERANKER_TOP_N": int(os.getenv("RERANKER_TOP_N", 3)),
}
