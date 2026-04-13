"""Ingest full NICE NG97 guideline PDF into Pinecone - LOCAL EMBEDDINGS"""

import sys
from pathlib import Path
import os
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import pymupdf4llm
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from aether.config.settings import settings
from aether.utils.logger import logger

# --- CONFIGURATION ---
# Place your 419-page PDF in the project root or a 'data' folder and update this path!
PDF_FILE_PATH = project_root / "data" / "NICE_NG97_Guideline.pdf" 


def ingest_pdf_data():
    """Ingest full NICE guidelines PDF into Pinecone."""
    
    os.environ['PINECONE_API_KEY'] = settings.pinecone_api_key
    
    logger.info("=" * 80)
    logger.info("Ingesting Full NICE NG97 PDF Guidelines (Semantic Chunking & Local Embeddings)")
    logger.info("=" * 80)
    
    # 1. Verify PDF exists
    if not PDF_FILE_PATH.exists():
        logger.error(f"❌ PDF not found at: {PDF_FILE_PATH}")
        logger.info("Please place your NICE guideline PDF at the specified path.")
        return False

    try:
        # 2. PDF to Markdown Conversion
        logger.info(f"\n1. Reading and converting PDF to Markdown: {PDF_FILE_PATH.name}...")
        logger.info("   (This might take a minute for a 419-page PDF...)")
        
        md_text = pymupdf4llm.to_markdown(str(PDF_FILE_PATH))
        logger.info("✅ PDF converted to Markdown successfully.")

        # 3. Semantic Chunking
        logger.info("\n2. Performing Semantic Chunking...")
        
        # Split by Markdown headers to keep clinical context intact
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(md_text)
        
        # Secondary split for any exceptionally long chapters to fit LLM context windows
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=150 # Slight overlap so sentences aren't hard-cut in the middle
        )
        documents = text_splitter.split_documents(md_header_splits)
        
        # Enrich metadata
        for doc in documents:
            doc.metadata["source"] = PDF_FILE_PATH.name
            doc.metadata["type"] = "Clinical Guideline"

        logger.info(f"✅ Created {len(documents)} highly semantic chunks from {PDF_FILE_PATH.name}.")

        # 4. Initialize Pinecone
        logger.info("\n3. Connecting to Pinecone...")
        pc = Pinecone(api_key=settings.pinecone_api_key)
        
        existing_indexes = pc.list_indexes()
        index_names = [idx.name for idx in existing_indexes]
        
        if settings.pinecone_index_name not in index_names:
            logger.error(f"❌ Index '{settings.pinecone_index_name}' does not exist!")
            logger.info(f"   Available indexes: {index_names}")
            return False
        
        index = pc.Index(settings.pinecone_index_name)
        logger.info("✅ Connected to Pinecone")
        
        # 5. Initialize local embeddings
        logger.info("\n4. Initializing local embeddings...")
        logger.info("   Model: all-MiniLM-L6-v2 (384 dimensions)")
        
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",  
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("✅ Local embeddings initialized")
        
        # 6. Ingest into Vector Store
        logger.info("\n5. Generating embeddings and ingesting into Pinecone...")
        logger.info("   (This will take a few minutes for a large document...)")
        
        vector_store = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=settings.pinecone_index_name
        )
        logger.info("✅ All documents embedded and ingested successfully")
        
        # 7. Verify ingestion
        logger.info("\n6. Verifying ingestion...")
        time.sleep(5)  # Wait for Pinecone to index the new massive batch
        stats = index.describe_index_stats()
        logger.info(f"✅ Index now contains {stats.total_vector_count} total vectors")
        
        # 8. Test retrieval
        logger.info("\n7. Testing retrieval...")
        test_query = "What cognitive assessment instruments are recommended for mild impairment?"
        results = vector_store.similarity_search(test_query, k=2)
        
        logger.info(f"✅ Retrieved {len(results)} documents for test query")
        if results:
            logger.info(f"\n📄 Top result preview:")
            logger.info(f"   Metadata: {results[0].metadata}")
            logger.info(f"   Text: {results[0].page_content[:200]}...")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ Massive PDF Data Ingestion Complete!")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = ingest_pdf_data()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n\nIngestion cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\n❌ Ingestion failed: {e}")
        sys.exit(1)