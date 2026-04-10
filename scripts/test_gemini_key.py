"""Test if Gemini API key is valid."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from aether.config.settings import settings
import google.generativeai as genai

print("\n" + "="*70)
print("  🔑 TESTING GEMINI API KEY")
print("="*70)

# Show masked key
key = settings.google_api_key
if key:
    masked_key = f"{key[:10]}...{key[-4:]}" if len(key) > 14 else "***"
    print(f"\n📋 API Key (masked): {masked_key}")
    print(f"   Length: {len(key)} characters")
else:
    print("\n❌ No API key found!")
    print("\nCheck your .env file:")
    print("  GOOGLE_API_KEY=your_actual_key_here")
    sys.exit(1)

# Test the key
print("\n🔌 Testing API key...")

try:
    genai.configure(api_key=settings.google_api_key)
    
    # Try to list models
    models = genai.list_models()
    
    print("✅ API key is VALID!")
    print("\n📊 Available models:")
    
    for model in list(models)[:5]:
        print(f"   • {model.name}")
    
    print("\n✅ Gemini API is working correctly!")
    
except Exception as e:
    print(f"\n❌ API key is INVALID!")
    print(f"\nError: {e}")
    print("\n🔧 Fix:")
    print("1. Go to: https://makersuite.google.com/app/apikey")
    print("2. Click 'Create API Key'")
    print("3. Copy the NEW key")
    print("4. Update .env file:")
    print("   GOOGLE_API_KEY=AIzaSy_your_actual_key_here")
    print()
    sys.exit(1)

print("\n" + "="*70)