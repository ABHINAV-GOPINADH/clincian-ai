from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from pinecone import Pinecone
from crewai_tools import BaseTool
from aether.config.settings import settings
from aether.utils.logger import logger


class NICEGuidelineRAG:
    """RAG tool for retrieving NICE NG97 dementia guidelines."""
    
    def __init__(self):
        self.vector_store = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Pinecone vector store."""
        try:
            pc = Pinecone(api_key=settings.pinecone_api_key)
            index = pc.Index(settings.pinecone_index_name)
            
            embeddings = OpenAIEmbeddings(
                openai_api_key=settings.openai_api_key
            )
            
            self.vector_store = PineconeVectorStore(
                index=index,
                embedding=embeddings,
                text_key="text"
            )
            
            logger.info("NICE NG97 RAG tool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RAG tool: {e}")
            raise
    
    def retrieve_guidance(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve relevant NICE guidance documents."""
        if not self.vector_store:
            self._initialize()
        
        results = self.vector_store.similarity_search(query, k=top_k)
        logger.debug(f"Retrieved {len(results)} documents for query: {query[:50]}...")
        return results
    
    def retrieve_for_instrument(self, instrument_type: str) -> str:
        """Retrieve guidance for specific assessment instrument."""
        query = f"NICE NG97 dementia assessment {instrument_type} recommendations"
        docs = self.retrieve_guidance(query, top_k=3)
        
        return "\n\n".join([
            f"[{idx + 1}] {doc.page_content}"
            for idx, doc in enumerate(docs)
        ])
    
    def retrieve_for_condition(self, condition: str) -> str:
        """Retrieve guidance for specific clinical condition."""
        query = f"NICE NG97 dementia {condition} clinical guidance"
        docs = self.retrieve_guidance(query, top_k=3)
        
        return "\n\n".join([
            f"[{idx + 1}] {doc.page_content}"
            for idx, doc in enumerate(docs)
        ])


class NICEGuidanceTool(BaseTool):
    """CrewAI tool wrapper for NICE guideline retrieval."""
    
    name: str = "NICE NG97 Guideline Retrieval"
    description: str = (
        "Retrieves relevant sections from NICE NG97 dementia assessment guidelines. "
        "Use this when you need evidence-based recommendations for assessment instruments, "
        "clinical pathways, or quality standards."
    )
    
    def __init__(self):
        super().__init__()
        self.rag = NICEGuidelineRAG()
    
    def _run(self, query: str) -> str:
        """Execute the tool."""
        docs = self.rag.retrieve_guidance(query, top_k=5)
        return "\n\n---\n\n".join([doc.page_content for doc in docs])


# Singleton instance
nice_rag = NICEGuidelineRAG()
nice_guidance_tool = NICEGuidanceTool()