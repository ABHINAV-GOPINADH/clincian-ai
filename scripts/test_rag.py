"""Test RAG retrieval from Pinecone."""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))  # ADDED

from aether.tools.rag_tool import nice_rag  # CHANGED: removed 'src.'
from aether.tools.pinecone_utils import get_index_stats  # CHANGED
from aether.utils.logger import logger  # CHANGED


def test_rag():
    """Test RAG retrieval."""
    
    logger.info("=" * 80)
    logger.info("Testing Pinecone RAG Integration")
    logger.info("=" * 80)
    
    # Check index stats
    logger.info("\n1. Checking index statistics...")
    try:
        stats = get_index_stats()
        logger.info(f"📊 Index Statistics:")
        logger.info(f"   Total vectors: {stats['total_vector_count']}")
        logger.info(f"   Dimensions: {stats.get('dimension', 'N/A')}")
        logger.info(f"   Namespaces: {stats.get('namespaces', {})}")
        
        if stats['total_vector_count'] == 0:
            logger.warning("⚠️  Index is empty! Run ingestion script first.")
            logger.info("\nTo populate the index, run:")
            logger.info("  python scripts/ingest_nice_guidelines.py")
            return
    except Exception as e:
        logger.error(f"❌ Failed to get index stats: {e}")
        logger.info("\nMake sure Pinecone is configured correctly in .env:")
        logger.info("  PINECONE_API_KEY=your_key")
        logger.info("  PINECONE_INDEX_NAME=nice-ng97-guidelines")
        logger.info("  PINECONE_ENVIRONMENT=your_environment")
        return
    
    # Test queries
    test_queries = [
        "What cognitive assessment instruments are recommended?",
        "MMSE scoring and interpretation",
        "Activities of Daily Living assessment",
        "Depression screening in dementia",
        "Risk assessment for dementia patients"
    ]
    
    logger.info("\n2. Testing retrieval with sample queries...")
    logger.info("=" * 80)
    
    successful_queries = 0
    failed_queries = 0
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n--- Query {i}/{len(test_queries)}: {query} ---")
        
        try:
            docs = nice_rag.retrieve_guidance(query, top_k=2)
            
            if docs:
                logger.info(f"✅ Retrieved {len(docs)} documents")
                successful_queries += 1
                
                for j, doc in enumerate(docs, 1):
                    logger.info(f"\n📄 Document {j}:")
                    logger.info(f"   Content preview: {doc.page_content[:200]}...")
                    
                    # Show metadata if available
                    if hasattr(doc, 'metadata') and doc.metadata:
                        logger.info(f"   Metadata: {doc.metadata}")
            else:
                logger.warning("⚠️  No documents retrieved")
                failed_queries += 1
                
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            failed_queries += 1
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Test Summary")
    logger.info("=" * 80)
    logger.info(f"✅ Successful queries: {successful_queries}/{len(test_queries)}")
    logger.info(f"❌ Failed queries: {failed_queries}/{len(test_queries)}")
    
    if successful_queries == len(test_queries):
        logger.info("\n🎉 All tests passed! RAG is working correctly.")
    elif successful_queries > 0:
        logger.warning(f"\n⚠️  Partial success. {failed_queries} queries failed.")
    else:
        logger.error("\n❌ All queries failed. Check your Pinecone setup.")
    
    # Test instrument-specific retrieval
    logger.info("\n" + "=" * 80)
    logger.info("3. Testing instrument-specific retrieval methods...")
    logger.info("=" * 80)
    
    instruments = ["MMSE", "MoCA", "ADL"]
    
    for instrument in instruments:
        logger.info(f"\n--- Testing {instrument} retrieval ---")
        try:
            guidance = nice_rag.retrieve_for_instrument(instrument)
            if guidance:
                logger.info(f"✅ Retrieved guidance for {instrument}")
                logger.info(f"   Preview: {guidance[:150]}...")
            else:
                logger.warning(f"⚠️  No guidance found for {instrument}")
        except Exception as e:
            logger.error(f"❌ Failed to retrieve {instrument}: {e}")
    
    # Test condition-specific retrieval
    logger.info("\n" + "=" * 80)
    logger.info("4. Testing condition-specific retrieval methods...")
    logger.info("=" * 80)
    
    conditions = ["Alzheimer's disease", "vascular dementia", "depression"]
    
    for condition in conditions:
        logger.info(f"\n--- Testing {condition} retrieval ---")
        try:
            guidance = nice_rag.retrieve_for_condition(condition)
            if guidance:
                logger.info(f"✅ Retrieved guidance for {condition}")
                logger.info(f"   Preview: {guidance[:150]}...")
            else:
                logger.warning(f"⚠️  No guidance found for {condition}")
        except Exception as e:
            logger.error(f"❌ Failed to retrieve {condition}: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("RAG Testing Complete")
    logger.info("=" * 80)


def test_rag_initialization():
    """Test RAG initialization separately."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing RAG Initialization")
    logger.info("=" * 80)
    
    try:
        logger.info("Initializing RAG tool...")
        nice_rag.initialize()
        logger.info("✅ RAG tool initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ RAG initialization failed: {e}")
        logger.info("\nTroubleshooting steps:")
        logger.info("1. Check .env file has correct Pinecone credentials")
        logger.info("2. Verify Pinecone index exists and is accessible")
        logger.info("3. Ensure OpenAI API key is set (for embeddings)")
        return False


def run_interactive_test():
    """Run interactive RAG testing."""
    logger.info("\n" + "=" * 80)
    logger.info("Interactive RAG Testing")
    logger.info("=" * 80)
    logger.info("Enter queries to test RAG retrieval (type 'exit' to quit)")
    
    while True:
        try:
            query = input("\n🔍 Enter query: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                logger.info("Exiting interactive mode...")
                break
            
            if not query:
                continue
            
            logger.info(f"Searching for: {query}")
            docs = nice_rag.retrieve_guidance(query, top_k=3)
            
            if docs:
                logger.info(f"✅ Found {len(docs)} relevant documents\n")
                for i, doc in enumerate(docs, 1):
                    print(f"\n{'='*60}")
                    print(f"Result {i}:")
                    print(f"{'='*60}")
                    print(doc.page_content)
                    if hasattr(doc, 'metadata') and doc.metadata:
                        print(f"\nMetadata: {doc.metadata}")
            else:
                logger.warning("⚠️  No documents found")
                
        except KeyboardInterrupt:
            logger.info("\nExiting...")
            break
        except Exception as e:
            logger.error(f"❌ Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Pinecone RAG integration")
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help="Run in interactive mode"
    )
    parser.add_argument(
        '--init-only',
        action='store_true',
        help="Only test initialization"
    )
    
    args = parser.parse_args()
    
    try:
        if args.init_only:
            test_rag_initialization()
        elif args.interactive:
            if test_rag_initialization():
                run_interactive_test()
        else:
            if test_rag_initialization():
                test_rag()
    except KeyboardInterrupt:
        logger.info("\n\nTest interrupted by user")
    except Exception as e:
        logger.error(f"\n\n❌ Test failed with error: {e}")
        sys.exit(1)