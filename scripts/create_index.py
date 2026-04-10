"""Create Pinecone index for NICE NG97 guidelines - LOCAL EMBEDDINGS"""

import sys
from pathlib import Path
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from pinecone import Pinecone, ServerlessSpec
from aether.config.settings import settings
from aether.utils.logger import logger


def create_index():
    """Create Pinecone index if it doesn't exist."""
    
    logger.info("=" * 80)
    logger.info("Creating Pinecone Index for NICE NG97 Guidelines")
    logger.info("=" * 80)
    
    try:
        # Initialize Pinecone
        logger.info("\n1. Initializing Pinecone client...")
        pc = Pinecone(api_key=settings.pinecone_api_key)
        logger.info("✅ Pinecone client initialized")
        
        # Check existing indexes
        logger.info(f"\n2. Checking existing indexes...")
        existing_indexes = pc.list_indexes()
        index_names = [index.name for index in existing_indexes]
        
        logger.info(f"   Existing indexes: {index_names}")
        
        if settings.pinecone_index_name in index_names:
            logger.warning(f"⚠️  Index '{settings.pinecone_index_name}' already exists!")
            
            response = input("\nDo you want to delete and recreate it? (yes/no): ").lower()
            
            if response in ['yes', 'y']:
                logger.info(f"Deleting existing index '{settings.pinecone_index_name}'...")
                pc.delete_index(settings.pinecone_index_name)
                logger.info("✅ Index deleted")
                
                logger.info("Waiting for deletion to complete...")
                time.sleep(10)
            else:
                logger.info("Keeping existing index. Exiting...")
                return
        
        # Create new index
        logger.info(f"\n3. Creating index '{settings.pinecone_index_name}'...")
        logger.info("   Configuration:")
        logger.info(f"   - Name: {settings.pinecone_index_name}")
        logger.info(f"   - Dimension: 384 (all-MiniLM-L6-v2)")  # CHANGED
        logger.info(f"   - Metric: cosine")
        logger.info(f"   - Cloud: AWS")
        logger.info(f"   - Region: {settings.pinecone_environment}")
        
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=384,  # CHANGED: all-MiniLM-L6-v2 dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=settings.pinecone_environment
            )
        )
        
        logger.info("✅ Index created successfully!")
        
        # Wait for index to be ready
        logger.info("\n4. Waiting for index to be ready...")
        for i in range(15):
            try:
                index = pc.Index(settings.pinecone_index_name)
                stats = index.describe_index_stats()
                logger.info("✅ Index is ready!")
                break
            except Exception:
                logger.info(f"   Waiting... ({i+1}/15)")
                time.sleep(2)
        
        # Verify
        logger.info("\n5. Verifying index...")
        index = pc.Index(settings.pinecone_index_name)
        stats = index.describe_index_stats()
        
        logger.info("✅ Index verified!")
        logger.info(f"\n📊 Index Statistics:")
        logger.info(f"   Name: {settings.pinecone_index_name}")
        logger.info(f"   Dimension: {stats.dimension}")
        logger.info(f"   Total vectors: {stats.total_vector_count}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ Index Creation Complete!")
        logger.info("=" * 80)
        logger.info("\nNext steps:")
        logger.info("1. Ingest sample NICE guidelines:")
        logger.info("   python scripts/ingest_sample_data.py")
        logger.info("\n2. Test RAG retrieval:")
        logger.info("   python scripts/test_rag.py")
        
    except Exception as e:
        logger.error(f"\n❌ Failed to create index: {e}")
        logger.error("\nTroubleshooting:")
        logger.error("1. Check your Pinecone API key in .env")
        logger.error("2. Verify your Pinecone environment/region")
        logger.error("3. Check Pinecone console: https://app.pinecone.io/")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    try:
        create_index()
    except KeyboardInterrupt:
        logger.info("\n\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"\n\n❌ Operation failed: {e}")
        sys.exit(1)