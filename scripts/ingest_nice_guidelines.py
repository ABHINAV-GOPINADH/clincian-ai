"""
Script to ingest NICE NG97 guidelines into Pinecone.

Usage:
    python scripts/ingest_nice_guidelines.py --pdf path/to/nice_ng97.pdf
    python scripts/ingest_nice_guidelines.py --text path/to/nice_guidelines.txt
"""

import argparse
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.aether.tools.rag_tool import nice_rag
from src.aether.utils.logger import logger


def load_documents(file_path: str) -> List[Document]:
    """Load documents from PDF or text file."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if path.suffix.lower() == '.pdf':
        logger.info(f"Loading PDF: {file_path}")
        loader = PyPDFLoader(str(path))
    elif path.suffix.lower() in ['.txt', '.md']:
        logger.info(f"Loading text file: {file_path}")
        loader = TextLoader(str(path))
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents")
    return documents


def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks


def ingest_documents(chunks: List[Document]):
    """Ingest document chunks into Pinecone."""
    logger.info("Starting ingestion into Pinecone...")
    
    # Add documents to vector store
    nice_rag.vector_store.add_documents(chunks)
    
    logger.info(f"Successfully ingested {len(chunks)} chunks into Pinecone")


def main():
    parser = argparse.ArgumentParser(description="Ingest NICE NG97 guidelines into Pinecone")
    parser.add_argument('--pdf', type=str, help='Path to PDF file')
    parser.add_argument('--text', type=str, help='Path to text file')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Chunk size for splitting')
    parser.add_argument('--chunk-overlap', type=int, default=200, help='Chunk overlap')
    
    args = parser.parse_args()
    
    if not args.pdf and not args.text:
        parser.error("Either --pdf or --text must be provided")
    
    file_path = args.pdf or args.text
    
    try:
        # Load documents
        documents = load_documents(file_path)
        
        # Split into chunks
        chunks = split_documents(documents, args.chunk_size, args.chunk_overlap)
        
        # Ingest into Pinecone
        ingest_documents(chunks)
        
        logger.info("✅ Ingestion complete!")
        
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    main()