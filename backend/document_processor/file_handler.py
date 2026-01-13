import os
import hashlib
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from config import constants
from config.settings import settings
from utils.logging import logger


class DocumentProcessor:
    """
     - Validating file sizes before processing
     - Using caching to avoid redundant processing of previously uploaded files
     - Extracting structured content from documents using Docling
     - Splitting text into chunks using MarkdownHeaderTextSplitter for better retrieval in vector databasesr
    """
    #Initializes cache directory and header settings.
    def __init__(self):
        self.headers = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        self.cache_dir = Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # Ensures that uploaded files do not exceed the size limit.
    def validate_files(self, files: List) -> None:
        """Validate the total size of the uploaded files."""
        total_size = sum(os.path.getsize(f.name) for f in files)

        if total_size > constants.MAX_TOTAL_SIZE:
            raise ValueError(
                f"Total size exceeds {constants.MAX_TOTAL_SIZE//1024//1024}MB limit"
            )

    # Handles document processing, caching, and deduplication.
    def process(self, files: List) -> List:
        """Process files with caching for subsequent queries"""
        self.validate_files(files)
        all_chunks = []
        seen_hashes = set()

        for file in files:
            try:
                # Generate content-based hash for caching
                with open(file.name, "rb") as f:
                    file_hash = self._generate_hash(f.read())

                cache_path = self.cache_dir / f"{file_hash}.pkl"

                if self._is_cache_valid(cache_path):
                    logger.info(f"Loading from cache: {file.name}")
                    chunks = self._load_from_cache(cache_path)
                else:
                    logger.info(f"Processing and caching: {file.name}")
                    chunks = self._process_file(file)
                    self._save_to_cache(chunks, cache_path)

                # Deduplicate chunks across files
                for chunk in chunks:
                    chunk_hash = self._generate_hash(chunk.page_content.encode())
                    if chunk_hash not in seen_hashes:
                        all_chunks.append(chunk)
                        seen_hashes.add(chunk_hash)

            except Exception as e:
                logger.error(f"Failed to process {file.name}: {str(e)}")
                continue

        logger.info(f"Total unique chunks: {len(all_chunks)}")
        return all_chunks

    # Converts a document into Markdown and splits it into chunks.
    def _process_file(self, file) -> List:
        """Original processing logic with Docling"""

        if not file.name.endswith((".pdf", ".docx", ".txt", ".md")):
            logger.warning(f"Skipping unsupported file type: {file.name}")
            return []

        converter = DocumentConverter()
        markdown = converter.convert(file.name).document.export_to_markdown()
        splitter = MarkdownHeaderTextSplitter(self.headers)
        return splitter.split_text(markdown)

    # Creates a unique hash of file content.
    def _generate_hash(self, content: bytes) -> str:

        return hashlib.sha256(content).hexdigest()

    # Saves processed document chunks to cache.
    def _save_to_cache(self, chunks: List, cache_path: Path):
        with open(cache_path, "wb") as f:
            pickle.dump({"timestamp": datetime.now().timestamp(), "chunks": chunks}, f)

    # Loads cached document chunks if available.
    def _load_from_cache(self, cache_path: Path) -> List:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data["chunks"]

    # Checks if a cached file is still valid.
    def _is_cache_valid(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False

        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return cache_age < timedelta(days=settings.CACHE_EXPIRE_DAYS)
