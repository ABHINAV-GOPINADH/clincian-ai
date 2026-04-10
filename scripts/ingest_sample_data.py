"""Ingest sample NICE NG97 guideline data into Pinecone - LOCAL EMBEDDINGS"""

import sys
from pathlib import Path
import os
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from aether.config.settings import settings
from aether.utils.logger import logger


# Sample NICE NG97 guideline excerpts
SAMPLE_GUIDELINES = [
    {
        "text": "NICE NG97 recommends using validated cognitive assessment instruments including MMSE (Mini-Mental State Examination), MoCA (Montreal Cognitive Assessment), and ACE-III (Addenbrooke's Cognitive Examination) for initial evaluation of cognitive impairment in dementia assessment.",
        "metadata": {"source": "NICE NG97", "section": "Cognitive Assessment", "instrument": "MMSE, MoCA, ACE-III"}
    },
    {
        "text": "The Mini-Mental State Examination (MMSE) is a 30-point questionnaire used extensively in clinical and research settings to measure cognitive impairment. Scores of 24-30 indicate normal cognition, 18-23 indicate mild impairment, and scores below 18 indicate severe impairment.",
        "metadata": {"source": "NICE NG97", "section": "MMSE Scoring", "instrument": "MMSE"}
    },
    {
        "text": "Activities of Daily Living (ADL) assessment should include basic activities such as bathing, dressing, eating, toileting, and transferring. Instrumental Activities of Daily Living (IADL) include more complex tasks like managing finances, medications, shopping, and using transportation.",
        "metadata": {"source": "NICE NG97", "section": "Functional Assessment", "instrument": "ADL, IADL"}
    },
    {
        "text": "Depression screening in dementia patients should use validated tools such as the Geriatric Depression Scale (GDS) or Cornell Scale for Depression in Dementia. Depression is common in dementia and can exacerbate cognitive symptoms.",
        "metadata": {"source": "NICE NG97", "section": "Depression Assessment", "instrument": "GDS"}
    },
    {
        "text": "Risk assessment for dementia patients should include evaluation of falls risk, wandering risk, self-neglect, medication non-adherence, and capacity for decision-making. High-risk patients require enhanced monitoring and safety planning.",
        "metadata": {"source": "NICE NG97", "section": "Risk Assessment", "category": "Safety"}
    },
    {
        "text": "The Montreal Cognitive Assessment (MoCA) is more sensitive than MMSE for detecting mild cognitive impairment. It assesses multiple cognitive domains including visuospatial abilities, executive functions, attention, language, and memory. Scores below 26 suggest cognitive impairment.",
        "metadata": {"source": "NICE NG97", "section": "MoCA", "instrument": "MoCA"}
    },
    {
        "text": "Neuropsychiatric symptoms in dementia should be assessed using the Neuropsychiatric Inventory (NPI), which evaluates delusions, hallucinations, agitation, depression, anxiety, apathy, irritability, disinhibition, and aberrant motor behavior.",
        "metadata": {"source": "NICE NG97", "section": "Behavioral Assessment", "instrument": "NPI"}
    },
    {
        "text": "Clinical Dementia Rating (CDR) scale stages dementia from 0 (no impairment) to 3 (severe dementia) based on cognitive and functional performance in six domains: memory, orientation, judgment, community affairs, home and hobbies, and personal care.",
        "metadata": {"source": "NICE NG97", "section": "Staging", "instrument": "CDR"}
    },
    {
        "text": "The Addenbrooke's Cognitive Examination (ACE-III) is a comprehensive cognitive test assessing five cognitive domains: attention, memory, verbal fluency, language, and visuospatial abilities. Maximum score is 100, with scores below 88 suggesting cognitive impairment.",
        "metadata": {"source": "NICE NG97", "section": "ACE-III", "instrument": "ACE-III"}
    },
    {
        "text": "For patients with suspected Alzheimer's disease, baseline assessment should include detailed cognitive testing, functional assessment, neuropsychiatric evaluation, and MRI brain imaging to rule out other causes of dementia.",
        "metadata": {"source": "NICE NG97", "section": "Alzheimer's Assessment", "condition": "Alzheimer's disease"}
    }
]


def ingest_sample_data():
    """Ingest sample NICE guidelines into Pinecone."""
    
    # Set environment variable for langchain-pinecone
    os.environ['PINECONE_API_KEY'] = settings.pinecone_api_key
    
    logger.info("=" * 80)
    logger.info("Ingesting Sample NICE NG97 Guidelines (Local Embeddings)")
    logger.info("=" * 80)
    
    try:
        # Initialize Pinecone
        logger.info("\n1. Connecting to Pinecone...")
        pc = Pinecone(api_key=settings.pinecone_api_key)
        
        # Check if index exists
        existing_indexes = pc.list_indexes()
        index_names = [idx.name for idx in existing_indexes]
        
        if settings.pinecone_index_name not in index_names:
            logger.error(f"❌ Index '{settings.pinecone_index_name}' does not exist!")
            logger.info(f"   Available indexes: {index_names}")
            logger.info("\nCreate it first:")
            logger.info("  python scripts/create_index.py")
            return False
        
        index = pc.Index(settings.pinecone_index_name)
        logger.info("✅ Connected to Pinecone")
        
        # Initialize local embeddings (FREE!)
        logger.info("\n2. Initializing local embeddings...")
        logger.info("   Model: all-MiniLM-L6-v2 (384 dimensions)")
        logger.info("   (First run may download model ~90MB)")
        
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",  # Fast, good quality, 384 dimensions
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("✅ Local embeddings initialized")
        
        # Create documents
        logger.info("\n3. Creating documents...")
        documents = [
            Document(page_content=item["text"], metadata=item["metadata"])
            for item in SAMPLE_GUIDELINES
        ]
        logger.info(f"✅ Created {len(documents)} documents")
        
        # Initialize vector store and ingest
        logger.info("\n4. Generating embeddings and ingesting into Pinecone...")
        logger.info("   (This may take 30-60 seconds...)")
        
        vector_store = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=settings.pinecone_index_name
        )
        logger.info("✅ Documents ingested successfully")
        
        # Verify ingestion
        logger.info("\n5. Verifying ingestion...")
        time.sleep(3)  # Wait for indexing
        stats = index.describe_index_stats()
        logger.info(f"✅ Index now contains {stats.total_vector_count} vectors")
        
        # Test retrieval
        logger.info("\n6. Testing retrieval...")
        test_query = "What cognitive assessment instruments are recommended?"
        results = vector_store.similarity_search(test_query, k=2)
        
        logger.info(f"✅ Retrieved {len(results)} documents for test query")
        if results:
            logger.info(f"\n📄 Sample result:")
            logger.info(f"   {results[0].page_content[:150]}...")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ Sample Data Ingestion Complete!")
        logger.info("=" * 80)
        logger.info(f"\n📊 Summary:")
        logger.info(f"   Ingested: {len(SAMPLE_GUIDELINES)} guideline excerpts")
        logger.info(f"   Embeddings: all-MiniLM-L6-v2 (384 dimensions)")
        logger.info(f"   Total vectors in index: {stats.total_vector_count}")
        logger.info("\nYou can now test RAG retrieval:")
        logger.info("  python scripts/test_rag.py")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = ingest_sample_data()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n\nIngestion cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\n❌ Ingestion failed: {e}")
        sys.exit(1)