from typing import Dict, Optional
from datetime import datetime
from aether.schemas.clinical import AgentContext, AuditEntry
from aether.utils.logger import logger


class ContextStore:
    """Shared context store for agent orchestration."""
    
    def __init__(self):
        self._store: Dict[str, AgentContext] = {}
    
    def initialize_context(self, encounter_id: str) -> AgentContext:
        """Initialize a new context for an encounter."""
        context = AgentContext(
            encounter_id=encounter_id,
            audit_trail=[]
        )
        self._store[encounter_id] = context
        logger.info(f"Context initialized for encounter: {encounter_id}")
        return context
    
    def get_context(self, encounter_id: str) -> Optional[AgentContext]:
        """Retrieve context for an encounter."""
        return self._store.get(encounter_id)
    
    def update_context(self, encounter_id: str, **updates) -> None:
        """Update context with new data."""
        context = self._store.get(encounter_id)
        if not context:
            raise ValueError(f"Context not found for encounter: {encounter_id}")
        
        for key, value in updates.items():
            setattr(context, key, value)
        
        self._store[encounter_id] = context
    
    def add_audit_entry(self, encounter_id: str, agent: str, action: str, data: any) -> None:
        """Add an audit trail entry."""
        context = self.get_context(encounter_id)
        if not context:
            raise ValueError(f"Context not found for encounter: {encounter_id}")
        
        entry = AuditEntry(
            agent=agent,
            timestamp=datetime.now().isoformat(),
            action=action,
            data=data
        )
        context.audit_trail.append(entry)
        self._store[encounter_id] = context
    
    def clear_context(self, encounter_id: str) -> None:
        """Clear context for an encounter."""
        if encounter_id in self._store:
            del self._store[encounter_id]
            logger.info(f"Context cleared for encounter: {encounter_id}")


# Singleton instance
context_store = ContextStore()