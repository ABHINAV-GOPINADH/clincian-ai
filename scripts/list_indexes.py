"""List all Pinecone indexes."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from pinecone import Pinecone
from aether.config.settings import settings
from aether.utils.logger import logger

try:
    pc = Pinecone(api_key=settings.pinecone_api_key)
    indexes = pc.list_indexes()
    
    print("\n" + "="*60)
    print("PINECONE INDEXES")
    print("="*60)
    
    if not indexes:
        print("\n❌ No indexes found in your account!")
        print("\nYou need to create the index first:")
        print("  python scripts/create_index.py")
    else:
        print(f"\n✅ Found {len(indexes)} index(es):\n")
        for idx in indexes:
            print(f"📌 {idx.name}")
            print(f"   Dimension: {idx.dimension}")
            print(f"   Metric: {idx.metric}")
            print(f"   Status: {idx.status.state if hasattr(idx, 'status') else 'Unknown'}")
            print()
    
    print("="*60)
    print(f"\nLooking for: '{settings.pinecone_index_name}'")
    print("="*60)
    
except Exception as e:
    logger.error(f"Failed to list indexes: {e}")
    print("\n❌ Error connecting to Pinecone!")
    print("\nCheck your .env file:")
    print(f"  PINECONE_API_KEY={settings.pinecone_api_key[:10]}...")
    print(f"  PINECONE_ENVIRONMENT={settings.pinecone_environment}")