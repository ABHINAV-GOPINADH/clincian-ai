# aether/schemas/state.py
"""
LangGraph State Schema for AETHER Clinical Workflow
Replaces context_store with native state management
"""
from typing import TypedDict, Optional, Annotated
from operator import add
from pydantic import BaseModel

from aether.schemas.clinical import (
    ReferralInput,
    PatientData,
    ClinicalHistory,
    PatientProfile,
    AssessmentPlan,
    ClinicalBrief,
    QAResult
)


class AuditEntry(BaseModel):
    """Single audit log entry."""
    timestamp: float
    encounter_id: str
    agent: str
    action: str
    data: dict
    step_number: int


class RiskFlag(BaseModel):
    """Risk flag with metadata."""
    flag_type: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    identified_by: str
    timestamp: float


class AetherState(TypedDict):
    """
    Complete state schema for AETHER clinical workflow.
    Uses Annotated with 'add' operator for list fields to enable proper merging.
    """
    # Input
    referral_input: ReferralInput
    encounter_id: str
    start_time: float
    
    # Agent outputs (immutable replacements)
    patient_data: Optional[PatientData]
    clinical_history: Optional[ClinicalHistory]
    patient_profile: Optional[PatientProfile]
    assessment_plan: Optional[AssessmentPlan]
    clinical_brief: Optional[ClinicalBrief]
    qa_result: Optional[QAResult]
    
    # Audit and tracking (use Annotated for proper list concatenation)
    audit_log: Annotated[list[AuditEntry], add]
    risk_flags: Annotated[list[RiskFlag], add]
    
    # Workflow metadata
    current_step: str
    steps_completed: Annotated[list[str], add]
    error: Optional[str]
    agent_execution_order: Annotated[list[str], add]
    
    # Performance tracking
    step_timings: dict[str, float]  # step_name -> duration_ms