from datetime import date, datetime
from enum import Enum
from typing import List, Optional, Any
from pydantic import BaseModel, Field, EmailStr


# ========== ENUMS ==========
class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class ConditionStatus(str, Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    CHRONIC = "chronic"


class Severity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class RiskCategory(str, Enum):
    CLINICAL = "clinical"
    SAFETY = "safety"
    MEDICATION = "medication"
    SOCIAL = "social"
    COGNITIVE = "cognitive"


class RiskSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CognitiveDomain(str, Enum):
    MEMORY = "memory"
    ATTENTION = "attention"
    LANGUAGE = "language"
    EXECUTIVE = "executive"
    VISUOSPATIAL = "visuospatial"


class InstrumentType(str, Enum):
    ADAS_COG = "ADAS-Cog"
    MMSE = "MMSE"
    MOCA = "MoCA"
    ACE_III = "ACE-III"
    ADL = "ADL"
    IADL = "IADL"
    CDR = "CDR"
    GDS = "GDS"
    NPI = "NPI"


class Priority(str, Enum):
    ESSENTIAL = "essential"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"


class QAStatus(str, Enum):
    GREEN = "green"
    AMBER = "amber"
    RED = "red"


class IssueSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# ========== INPUT SCHEMAS ==========
class ReferralInput(BaseModel):
    referral_text: str = Field(..., description="Raw GP referral letter text")
    nhs_number: Optional[str] = None
    referral_reason: Optional[str] = None
    encounter_id: Optional[str] = None


# ========== PATIENT DATA SCHEMAS ==========
class ContactInfo(BaseModel):
    phone: Optional[str] = None
    email: Optional[EmailStr] = None
    address: Optional[str] = None


class GPDetails(BaseModel):
    practice_name: str
    gp_name: Optional[str] = None
    contact_number: Optional[str] = None


class PatientName(BaseModel):
    first: str
    last: str


class PatientData(BaseModel):
    nhs_number: str
    name: PatientName
    date_of_birth: str
    age: int
    gender: Gender
    contact_info: ContactInfo
    gp_details: GPDetails
    referral_reason: str
    referral_date: str


# ========== CLINICAL HISTORY SCHEMAS ==========
class Condition(BaseModel):
    code: str = Field(..., description="SNOMED-CT code")
    display: str
    onset_date: Optional[str] = None
    status: ConditionStatus
    severity: Optional[Severity] = None


class Medication(BaseModel):
    name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    start_date: Optional[str] = None
    indication: Optional[str] = None


class PastAssessment(BaseModel):
    date: str
    type: str
    outcome: str


class TimelineEvent(BaseModel):
    date: str
    event: str
    significance: str = Field(..., pattern="^(low|medium|high)$")


class ClinicalHistory(BaseModel):
    conditions: List[Condition]
    medications: List[Medication]
    allergies: List[str]
    past_assessments: List[PastAssessment]
    timeline_events: List[TimelineEvent]


# ========== PROFILER SCHEMAS ==========
class RiskFlag(BaseModel):
    category: RiskCategory
    severity: RiskSeverity
    description: str
    reasoning: str
    mitigation_strategy: Optional[str] = None


class CognitiveIndicator(BaseModel):
    domain: CognitiveDomain
    concern: str
    evidence_source: str


class ComplexitySummary(BaseModel):
    score: int = Field(..., ge=1, le=10)
    factors: List[str]


class PatientProfile(BaseModel):
    risk_flags: List[RiskFlag]
    cognitive_indicators: List[CognitiveIndicator]
    complexity_summary: ComplexitySummary
    information_gaps: List[str]


# ========== PLANNER SCHEMAS ==========
class Instrument(BaseModel):
    name: str
    type: InstrumentType
    priority: Priority
    rationale: str
    nice_guidance_reference: Optional[str] = None
    estimated_duration: int = Field(..., description="Duration in minutes")


class AssessmentPlan(BaseModel):
    instruments: List[Instrument]
    total_estimated_duration: int
    priority_order: List[str]
    contraindications: List[str]
    special_considerations: List[str]
    nice_compliance_notes: str


# ========== BRIEF SCHEMAS ==========
class BriefHeader(BaseModel):
    patient_name: str
    nhs_number: str
    date_of_birth: str
    assessment_date: str
    clinician: Optional[str] = None


class RelevantHistory(BaseModel):
    medical: List[str]
    psychiatric: List[str]
    social: List[str]


class ClinicalBrief(BaseModel):
    header: BriefHeader
    executive_summary: str = Field(..., max_length=500)
    presenting_concerns: List[str]
    relevant_history: RelevantHistory
    risk_summary: List[RiskFlag]
    recommended_assessments: List[Instrument]
    key_considerations: List[str]
    nice_guidance_alignment: str
    additional_notes: Optional[str] = None


# ========== QA SCHEMAS ==========
class ValidationIssue(BaseModel):
    field: str
    severity: IssueSeverity
    message: str
    suggestion: Optional[str] = None


class ClinicalAccuracy(BaseModel):
    score: float = Field(..., ge=0, le=100)
    issues: List[ValidationIssue]


class NICECompliance(BaseModel):
    compliant: bool
    gaps: List[str]


class DataCompleteness(BaseModel):
    percentage: float = Field(..., ge=0, le=100)
    missing_fields: List[str]


class SafetyChecks(BaseModel):
    passed: bool
    flags: List[str]


class QAResult(BaseModel):
    overall_status: QAStatus
    clinical_accuracy: ClinicalAccuracy
    nice_compliance: NICECompliance
    data_completeness: DataCompleteness
    safety_checks: SafetyChecks
    recommendations: List[str]


# ========== ORCHESTRATOR SCHEMAS ==========
class AuditEntry(BaseModel):
    agent: str
    timestamp: str
    action: str
    data: Any


class AgentContext(BaseModel):
    encounter_id: str
    patient_history: Optional[ClinicalHistory] = None
    patient_profile: Optional[PatientProfile] = None
    risk_flags: Optional[List[RiskFlag]] = None
    assessment_plan: Optional[AssessmentPlan] = None
    audit_trail: List[AuditEntry] = Field(default_factory=list)


class OrchestratorMetadata(BaseModel):
    processing_time_ms: int
    agent_execution_order: List[str]
    version: str = "1.0.0"


class OrchestratorOutput(BaseModel):
    patient_data: PatientData
    clinical_history: ClinicalHistory
    patient_profile: PatientProfile
    assessment_plan: AssessmentPlan
    clinical_brief: ClinicalBrief
    qa_result: QAResult
    metadata: OrchestratorMetadata