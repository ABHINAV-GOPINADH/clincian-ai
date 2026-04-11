"""RAG tool for retrieving NICE NG97 dementia guidelines - LOCAL EMBEDDINGS"""

from typing import List, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings  # CHANGED
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone
from aether.config.settings import settings
from aether.utils.logger import logger
from crewai.tools import tool

class NICEGuidelineRAG:
    """RAG tool for retrieving NICE NG97 dementia guidelines."""
    
    def __init__(self):
        self.vector_store: Optional[PineconeVectorStore] = None
        self.initialized = False
    
    def _initialize(self):
        """Initialize Pinecone vector store with local embeddings."""
        if self.initialized:
            return
            
        try:
            logger.info("Initializing Pinecone connection...")
            pc = Pinecone(api_key=settings.pinecone_api_key)
            
            # Check if index exists
            existing_indexes = pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes]
            
            if settings.pinecone_index_name not in index_names:
                logger.warning(f"⚠️  Pinecone index '{settings.pinecone_index_name}' does not exist!")
                logger.info(f"Available indexes: {index_names}")
                logger.info("\nTo create and populate the index, run:")
                logger.info("  python scripts/create_index.py")
                logger.info("  python scripts/ingest_sample_data.py")
                raise ValueError(
                    f"Pinecone index '{settings.pinecone_index_name}' not found. "
                    "Please create it first using: python scripts/create_index.py"
                )
            
            index = pc.Index(settings.pinecone_index_name)
            
            # Use local HuggingFace embeddings (FREE!)
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            self.vector_store = PineconeVectorStore(
                index=index,
                embedding=embeddings,
                text_key="text"
            )
            
            self.initialized = True
            logger.info("✅ NICE NG97 RAG tool initialized successfully (local embeddings)")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG tool: {e}")
            raise
    
    def retrieve_guidance(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve relevant NICE guidance documents."""
        if not self.initialized:
            self._initialize()
        
        if not self.vector_store:
            logger.warning("Vector store not initialized, returning empty results")
            return []
        
        try:
            results = self.vector_store.similarity_search(query, k=top_k)
            logger.debug(f"Retrieved {len(results)} documents for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Failed to retrieve guidance: {e}")
            return []
    
    def retrieve_for_instrument(self, instrument_type: str) -> str:
        """Retrieve guidance for specific assessment instrument."""
        query = f"NICE NG97 dementia assessment {instrument_type} recommendations"
        docs = self.retrieve_guidance(query, top_k=3)
        
        if not docs:
            return f"No guidance found for {instrument_type}"
        
        return "\n\n".join([
            f"[{idx + 1}] {doc.page_content}"
            for idx, doc in enumerate(docs)
        ])
    
    def retrieve_for_condition(self, condition: str) -> str:
        """Retrieve guidance for specific clinical condition."""
        query = f"NICE NG97 dementia {condition} clinical guidance"
        docs = self.retrieve_guidance(query, top_k=3)
        
        if not docs:
            return f"No guidance found for {condition}"
        
        return "\n\n".join([
            f"[{idx + 1}] {doc.page_content}"
            for idx, doc in enumerate(docs)
        ])
    
    def initialize(self):
        """Explicitly initialize the RAG tool (for testing)."""
        self._initialize()


# Singleton instance
nice_rag = NICEGuidelineRAG()


# Simple function wrapper for CrewAI tools
@tool("NICE NG97 Guidance Retrieval")
def nice_guidance_tool(query: str) -> str:
    """
    Retrieve NICE NG97 dementia assessment guidelines.
    
    Args:
        query: Search query for NICE guidelines
        
    Returns:
        Relevant guideline excerpts
    """
    try:
        docs = nice_rag.retrieve_guidance(query, top_k=5)
        if not docs:
            return "No relevant guidelines found for this query."
        return "\n\n---\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        logger.error(f"Error retrieving guidance: {e}")
        return f"Error: Unable to retrieve guidelines - {str(e)}"