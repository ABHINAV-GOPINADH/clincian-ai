"""Find the correct Pinecone region from your existing index."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from pinecone import Pinecone
from aether.config.settings import settings

print("\n" + "="*70)
print("  🔍 FINDING YOUR PINECONE REGION")
print("="*70)

try:
    pc = Pinecone(api_key=settings.pinecone_api_key)
    indexes = pc.list_indexes()
    
    if not indexes:
        print("\n❌ No indexes found!")
        print("\n📋 Available regions for serverless:")
        print("   • us-east-1")
        print("   • us-west-2")
        print("   • eu-west-1")
        print("\nCheck your Pinecone dashboard: https://app.pinecone.io/")
    else:
        print(f"\n✅ Found {len(indexes)} existing index(es):\n")
        
        for idx in indexes:
            print(f"📌 {idx.name}")
            print(f"   Dimension: {idx.dimension}")
            print(f"   Metric: {idx.metric}")
            
            # Get the host to determine region
            if hasattr(idx, 'host'):
                host = idx.host
                print(f"   Host: {host}")
                
                # Extract region from host
                if 'us-east-1' in host:
                    region = 'us-east-1'
                elif 'us-west-2' in host:
                    region = 'us-west-2'
                elif 'eu-west-1' in host:
                    region = 'eu-west-1'
                elif 'gcp-starter' in host:
                    region = 'gcp-starter'
                else:
                    region = 'unknown (check Pinecone console)'
                
                print(f"   🌍 Region: {region}")
            
            print()
        
        print("="*70)
        print("\n💡 UPDATE YOUR .env FILE:")
        print("="*70)
        
        # Suggest the region
        first_index = indexes[0]
        if hasattr(first_index, 'host'):
            host = first_index.host
            
            if 'us-east-1' in host:
                suggested_region = 'us-east-1'
            elif 'us-west-2' in host:
                suggested_region = 'us-west-2'
            elif 'eu-west-1' in host:
                suggested_region = 'eu-west-1'
            elif 'gcp-starter' in host:
                print("\n⚠️  You're using GCP Starter (free tier)")
                print("Change PINECONE_ENVIRONMENT to: gcp-starter")
                suggested_region = None
            else:
                suggested_region = None
            
            if suggested_region:
                print(f"\nChange this line in .env:")
                print(f"  PINECONE_ENVIRONMENT={suggested_region}")
        
        print()

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("="*70)