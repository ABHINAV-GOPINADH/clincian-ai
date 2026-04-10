"""Get detailed index information including region."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from pinecone import Pinecone
from aether.config.settings import settings

print("\n" + "="*70)
print("  📊 PINECONE INDEX DETAILS")
print("="*70)

try:
    pc = Pinecone(api_key=settings.pinecone_api_key)
    
    # Get list of indexes with full details
    indexes = pc.list_indexes()
    
    for idx in indexes:
        print(f"\n📌 Index: {idx.name}")
        print(f"   Dimension: {idx.dimension}")
        print(f"   Metric: {idx.metric}")
        print(f"   Host: {idx.host}")
        
        # Try to get spec details
        if hasattr(idx, 'spec'):
            print(f"\n   📋 Spec Details:")
            spec = idx.spec
            
            # Check if serverless
            if hasattr(spec, 'serverless'):
                print(f"      Type: Serverless")
                serverless = spec.serverless
                if hasattr(serverless, 'cloud'):
                    print(f"      Cloud: {serverless.cloud}")
                if hasattr(serverless, 'region'):
                    print(f"      Region: {serverless.region}")
                    print(f"\n   ✅ USE THIS IN .env:")
                    print(f"      PINECONE_ENVIRONMENT={serverless.region}")
            
            # Check if pod-based
            elif hasattr(spec, 'pod'):
                print(f"      Type: Pod-based")
                pod = spec.pod
                if hasattr(pod, 'environment'):
                    print(f"      Environment: {pod.environment}")
                    print(f"\n   ✅ USE THIS IN .env:")
                    print(f"      PINECONE_ENVIRONMENT={pod.environment}")
                if hasattr(pod, 'pod_type'):
                    print(f"      Pod Type: {pod.pod_type}")
        
        print()
    
    print("="*70)
    print("\n💡 NEXT STEPS:")
    print("="*70)
    print("\n1. Copy the PINECONE_ENVIRONMENT value from above")
    print("2. Update your .env file")
    print("3. Run: python scripts/create_index.py")
    print()

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()