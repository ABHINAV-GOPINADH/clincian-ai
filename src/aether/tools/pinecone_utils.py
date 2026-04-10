"""Pinecone initialization and management utilities."""

from pinecone import Pinecone, ServerlessSpec
from aether.config.settings import settings
from aether.utils.logger import logger
import time


def initialize_pinecone():
    """Initialize Pinecone client."""
    pc = Pinecone(api_key=settings.pinecone_api_key)
    logger.info("Pinecone client initialized")
    return pc


def create_index_if_not_exists(pc: Pinecone, index_name: str = None):
    """Create Pinecone index if it doesn't exist."""
    if index_name is None:
        index_name = settings.pinecone_index_name
    
    # Check if index exists
    existing_indexes = pc.list_indexes()
    index_names = [idx['name'] for idx in existing_indexes]
    
    if index_name in index_names:
        logger.info(f"Index '{index_name}' already exists")
        return pc.Index(index_name)
    
    # Determine dimension based on embedding model
    dimension = 1536 if not settings.use_google_embeddings else 768
    
    logger.info(f"Creating index '{index_name}' with dimension {dimension}")
    
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region=settings.pinecone_environment.split('-')[0] + '-' + 
                   settings.pinecone_environment.split('-')[1] + '-' +
                   settings.pinecone_environment.split('-')[2]
        )
    )
    
    # Wait for index to be ready
    while not pc.describe_index(index_name).status['ready']:
        logger.info("Waiting for index to be ready...")
        time.sleep(1)
    
    logger.info(f"Index '{index_name}' created successfully")
    return pc.Index(index_name)


def get_index_stats(index_name: str = None):
    """Get statistics about the Pinecone index."""
    if index_name is None:
        index_name = settings.pinecone_index_name
    
    pc = initialize_pinecone()
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    
    logger.info(f"Index stats for '{index_name}':")
    logger.info(f"  - Total vectors: {stats['total_vector_count']}")
    logger.info(f"  - Dimension: {stats['dimension']}")
    
    return stats


def delete_index(index_name: str = None):
    """Delete a Pinecone index (use with caution!)."""
    if index_name is None:
        index_name = settings.pinecone_index_name
    
    pc = initialize_pinecone()
    pc.delete_index(index_name)
    logger.warning(f"Index '{index_name}' deleted")