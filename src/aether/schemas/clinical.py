from datetime import date
from enum import Enum
from typing import List, Optional, Any, Dict, Union,Literal
from pydantic import BaseModel, Field, EmailStr, field_validator, AliasChoices
import re

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


# ========== NEW ENUMS ==========

class UrgencyLevel(str, Enum):
    """
    Maps to referral.urgency in the payload.
    Reflects NHS referral urgency categories.
    """
    ROUTINE = "routine"
    SOON = "soon"
    URGENT = "urgent"
    TWO_WEEK_WAIT = "two_week_wait"


class MedicationStatus(str, Enum):
    """
    Maps to medications[].status in the payload.
    """
    ACTIVE = "active"
    DISCONTINUED = "discontinued"
    ON_HOLD = "on_hold"


class MedicationRoute(str, Enum):
    """
    Maps to medications[].route in the payload.
    Covers all routes present across the 5 cases.
    """
    ORAL = "oral"
    TRANSDERMAL = "transdermal"
    INHALED = "inhaled"
    SUBCUTANEOUS = "subcutaneous"
    INTRAVENOUS = "intravenous"


class AllergyVerificationStatus(str, Enum):
    """
    Maps to allergies[].verification_status in the payload.
    """
    CONFIRMED = "confirmed"
    UNCONFIRMED = "unconfirmed"
    REFUTED = "refuted"
    ENTERED_IN_ERROR = "entered_in_error"


class AllergyCriticality(str, Enum):
    """
    Maps to allergies[].criticality in the payload.
    """
    LOW = "low"
    HIGH = "high"
    UNABLE_TO_ASSESS = "unable_to_assess"


class SmokingStatus(str, Enum):
    """
    Maps to social_history.smoking_status free-text in the payload.
    Standardised enum to allow downstream logic.
    """
    NEVER = "never"
    EX_SMOKER = "ex_smoker"
    CURRENT = "current"
    UNKNOWN = "unknown"


class ConditionVerificationStatus(str, Enum):
    """
    Captures whether a condition is query/provisional or confirmed.
    Derived from conditions[].verified_by field patterns in the payload.
    """
    CONFIRMED = "confirmed"
    PROVISIONAL = "provisional"
    QUERY = "query"
    REFUTED = "refuted"


class ReferralSpecialty(str, Enum):
    """
    Maps to receiving_service.specialty in the payload.
    """
    OLD_AGE_PSYCHIATRY = "Old Age Psychiatry"
    NEUROLOGY = "Neurology"
    NEUROLOGY_OLD_AGE_PSYCHIATRY = "Neurology / Old Age Psychiatry"
    LIAISON_PSYCHIATRY = "Liaison Psychiatry / Old Age Psychiatry"
    GENERAL_MEDICINE = "General Medicine"


class InformantRelationship(str, Enum):
    """
    Maps to next_of_kin.relationship in the payload.
    """
    SPOUSE = "spouse"
    DAUGHTER = "daughter"
    SON = "son"
    SIBLING = "sibling"
    FRIEND = "friend"
    FORMAL_CARER = "formal_carer"
    OTHER = "other"


# ========== INPUT SCHEMAS ==========
class ReferralInput(BaseModel):
    referral_text: str = Field(..., description="Raw GP referral letter text")
    nhs_number: Optional[str] = None
    referral_reason: Optional[str] = None
    encounter_id: Optional[str] = None


# ========== NEW: REFERRING AND RECEIVING CLINICIAN/SERVICE SCHEMAS ==========

class ReferringClinician(BaseModel):
    """
    Maps to referral.referring_clinician in the payload.
    Captures the identity and organisation of the referring clinician.
    """
    name: str
    role: str
    gmc_number: Optional[str] = None
    organisation_ods: Optional[str] = None
    organisation_name: Optional[str] = None


class ReceivingService(BaseModel):
    """
    Maps to referral.receiving_service in the payload.
    Captures the destination service for the referral.
    """
    name: str
    ods_code: Optional[str] = None
    specialty: Optional[ReferralSpecialty] = None


class ReferralRecord(BaseModel):
    """
    Maps to the referral block in the payload.
    Extends ReferralInput with structured NHS referral metadata.
    ReferralInput is preserved for raw-text ingestion by the Clinical History Agent.
    """
    referral_id: str
    referral_date: str
    urgency: UrgencyLevel
    referral_reason: str
    referring_clinician: ReferringClinician
    receiving_service: ReceivingService
    encounter_id: str


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


# ========== NEW: EXTENDED DEMOGRAPHICS ==========

class EthnicityCode(BaseModel):
    """
    Maps to patient.demographics.ethnicity in the payload.
    Uses NHS 2001 Census Ethnic Category codes.
    """
    code: str = Field(..., description="NHS 2001 Census Ethnic Category code, e.g. 'A'")
    display: str
    system: str = "NHS 2001 Census Ethnic Category"


class LanguageCode(BaseModel):
    """
    Maps to patient.demographics.language in the payload.
    Uses BCP-47 language tags.
    """
    code: str = Field(..., description="BCP-47 language tag, e.g. 'en'")
    display: str
    system: str = "BCP-47"


class StructuredAddress(BaseModel):
    """
    Maps to patient.demographics.address in the payload.
    More granular than the ContactInfo.address string field.
    ContactInfo is preserved for simple string use-cases.
    """
    line_1: str
    line_2: Optional[str] = None
    city: Optional[str] = None
    county: Optional[str] = None
    postcode: Optional[str] = None
    country: Optional[str] = "GB"


class NextOfKin(BaseModel):
    """
    Maps to patient.demographics.contact.next_of_kin in the payload.
    Captures carer/informant detail needed by the Profiler and Planner agents.
    """
    name: str
    relationship: InformantRelationship
    relationship_fhir: Optional[str] = Field(
        None,
        description="FHIR R4 UK Core relationship code, e.g. 'SPS', 'DAUC', 'SONC'"
    )
    phone: Optional[str] = None
    consent_to_contact: bool = False
    informant_available_for_iqcode: Optional[bool] = None
    cohabiting: Optional[bool] = None

    # Lasting Power of Attorney fields — present in Case 3
    poa_health_welfare: Optional[bool] = Field(
        None,
        description="True if next of kin holds registered LPA for Health & Welfare"
    )
    poa_registered_date: Optional[str] = Field(
        None,
        description="Date LPA was registered with the Office of the Public Guardian"
    )


class ExtendedContactInfo(BaseModel):
    """
    Maps to patient.demographics.contact in the payload.
    Extends ContactInfo with structured next_of_kin.
    ContactInfo is kept for backward compatibility in PatientData.
    """
    phone: Optional[str] = None
    email: Optional[EmailStr] = None
    address: Optional[StructuredAddress] = None
    next_of_kin: Optional[NextOfKin] = None


class ExtendedGPDetails(BaseModel):
    """
    Maps to patient.gp_practice in the payload.
    Extends GPDetails with ODS code and address.
    GPDetails is kept for backward compatibility in PatientData.
    """
    ods_code: Optional[str] = None
    name: str
    address: Optional[str] = None
    gp_name: Optional[str] = None


class ExtendedPatientDemographics(BaseModel):
    """
    Full structured demographics block from the payload.
    PatientData is preserved for the existing pipeline interface.
    This model is used internally by the Clinical History Agent
    to capture the richer payload structure.
    """
    nhs_number: str
    title: Optional[str] = None
    given_name: str
    family_name: str
    date_of_birth: str
    age_at_referral: int
    sex: str
    birth_sex: Optional[str] = None
    gender: Gender
    ethnicity: Optional[EthnicityCode] = None
    language: Optional[LanguageCode] = None
    address: Optional[StructuredAddress] = None
    contact: Optional[ExtendedContactInfo] = None


# ========== CLINICAL HISTORY SCHEMAS ==========
class Condition(BaseModel):
    code: Optional[str] = None
    display: str
    status: str
    severity: Optional[str] = None

    @field_validator('code')
    @classmethod
    def validate_clinical_code(cls, v):
        if v is None:
            return v
        v = v.strip()
        if re.match(r'^[A-Z]\d{5}$', v):
            raise ValueError(
                f"Rejected: '{v}' matches an ODS Clinic Code format, "
                f"not a clinical diagnostic code."
            )
        invalid_keywords = ['ODS', 'GMC', 'NHS', 'Clinic']
        if any(keyword in v.upper() for keyword in invalid_keywords):
            raise ValueError(
                f"Rejected: '{v}' appears to be an administrative number."
            )
        return v


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

class Event(BaseModel):
    date: date | str | None = None  # accept ISO string or date
    description: str
    severity: Literal['mild', 'moderate', 'severe'] | str | None = None


class ClinicalHistory(BaseModel):
    conditions: List[Condition]
    medications: List[Medication]
    allergies: List[str]
    past_assessments: List[PastAssessment]
    timeline_events: List[TimelineEvent]
    significant_events: list[Event] | None = Field(
        default=None,
        validation_alias=AliasChoices("significant_events","timeline_events","events","recent_events")
    )


# ========== NEW: EXTENDED CLINICAL HISTORY MODELS ==========

class ExtendedCondition(BaseModel):
    """
    Maps to clinical_history.conditions[] in the payload.
    Extends Condition with SNOMED/ICD-10 coding and provenance.
    Condition is preserved for the existing pipeline interface.
    """
    snomed_code: Optional[str] = Field(
        None,
        description="SNOMED CT concept identifier"
    )
    icd10_code: Optional[str] = Field(
        None,
        description="ICD-10 diagnostic code"
    )
    display: str
    clinical_status: ConditionStatus = ConditionStatus.ACTIVE
    verification_status: ConditionVerificationStatus = ConditionVerificationStatus.CONFIRMED
    onset_date: Optional[str] = None
    verified_by: Optional[str] = Field(
        None,
        description="Clinician or service that verified the condition"
    )
    clinical_note: Optional[str] = None

    @field_validator('snomed_code')
    @classmethod
    def validate_snomed(cls, v):
        """
        SNOMED CT concept IDs are purely numeric.
        Rejects any value that contains non-digit characters.
        """
        if v is None:
            return v
        v = v.strip()
        if not re.match(r'^\d+$', v):
            raise ValueError(
                f"Rejected: '{v}' is not a valid SNOMED CT concept ID "
                f"(must be numeric only)."
            )
        return v


class ExtendedMedication(BaseModel):
    """
    Maps to clinical_history.medications[] in the payload.
    Extends Medication with dm+d VMPID, route, status, and prescriber.
    Medication is preserved for the existing pipeline interface.
    """
    display: str
    dmd_vmpid: Optional[str] = Field(
        None,
        description="NHS dm+d Virtual Medicinal Product ID"
    )
    dose: Optional[str] = None
    route: Optional[MedicationRoute] = None
    frequency: Optional[str] = None
    status: MedicationStatus = MedicationStatus.ACTIVE
    prescribed_date: Optional[str] = None
    prescriber: Optional[str] = None
    clinical_note: Optional[str] = Field(
        None,
        description="Clinician annotation — cautions, monitoring requirements, titration notes"
    )


class ExtendedAllergy(BaseModel):
    """
    Maps to clinical_history.allergies[] in the payload.
    Replaces the List[str] allergies field for structured allergy recording.
    ClinicalHistory.allergies (List[str]) is preserved for backward compatibility.
    """
    substance_display: str
    substance_snomed: Optional[str] = Field(
        None,
        description="SNOMED CT concept ID for the allergen substance"
    )
    reaction: Optional[str] = None
    severity: Optional[Severity] = None
    criticality: Optional[AllergyCriticality] = None
    verification_status: AllergyVerificationStatus = AllergyVerificationStatus.CONFIRMED
    onset_date: Optional[str] = None

    @field_validator('substance_snomed')
    @classmethod
    def validate_substance_snomed(cls, v):
        if v is None:
            return v
        v = v.strip()
        if not re.match(r'^\d+$', v):
            raise ValueError(
                f"Rejected: '{v}' is not a valid SNOMED CT concept ID."
            )
        return v


class SocialHistory(BaseModel):
    """
    Maps to clinical_history.social_history in the payload.
    New model — not present in the original schemas.
    Captures psychosocial context required by Profiler and Planner agents.
    """
    lives_with: Optional[str] = None
    informal_carer: Optional[str] = None
    smoking_status: Optional[SmokingStatus] = None
    smoking_pack_years: Optional[float] = Field(
        None,
        description="Pack-year history where documented (e.g. Case 5: 20 pack-years)"
    )
    alcohol_units_per_week: Optional[float] = None
    mobility: Optional[str] = None
    driving: Optional[bool] = None
    driving_clinical_note: Optional[str] = Field(
        None,
        description="DVLA notification status or driving review notes (e.g. Case 4)"
    )
    housing: Optional[str] = Field(
        None,
        description="Housing type, e.g. supported living (Case 5)"
    )
    occupation_retired: Optional[str] = None
    education_years: Optional[int] = Field(
        None,
        description="Years of formal education — relevant for cognitive test norms (Case 4: 18)"
    )
    pre_morbid_adl: Optional[str] = Field(
        None,
        description="Pre-morbid functional status narrative (Case 3)"
    )
    poa_active: Optional[bool] = Field(
        None,
        description="Flag indicating an active Lasting Power of Attorney (Case 3)"
    )
    carer_fatigue_noted: Optional[bool] = Field(
        None,
        description="Clinician-noted carer fatigue — relevant for safeguarding (Case 5)"
    )


# ========== NEW: PRIOR ASSESSMENT SCHEMAS ==========

class ACEIIISubScores(BaseModel):
    """
    Maps to prior_assessments[].sub_scores in the payload (Case 4).
    Captures domain-level ACE-III breakdown for the Profiler agent.
    """
    attention_orientation: Optional[int] = Field(None, ge=0, le=18)
    memory: Optional[int] = Field(None, ge=0, le=26)
    fluency: Optional[int] = Field(None, ge=0, le=14)
    language: Optional[int] = Field(None, ge=0, le=26)
    visuospatial: Optional[int] = Field(None, ge=0, le=16)


class PriorCognitiveAssessment(BaseModel):
    """
    Maps to patient.prior_assessments[] in the payload.
    Extends PastAssessment with instrument coding, scoring, and informant detail.
    PastAssessment is preserved for the existing pipeline interface.
    """
    instrument: str = Field(..., description="Full instrument name")
    instrument_snomed: Optional[str] = Field(
        None,
        description="SNOMED CT concept ID for the assessment instrument"
    )
    instrument_loinc: Optional[str] = Field(
        None,
        description="LOINC code for the assessment instrument"
    )
    date: str
    score: Optional[int] = None
    max_score: Optional[int] = None
    interpretation: Optional[str] = None
    sub_scores: Optional[ACEIIISubScores] = Field(
        None,
        description="Domain-level sub-scores where available (currently ACE-III, Case 4)"
    )
    performer: Optional[str] = None
    setting: Optional[str] = None
    informant: Optional[str] = Field(
        None,
        description="Informant name where score derived from informant section (Case 5 GPCOG)"
    )
    notes: Optional[str] = None


# ========== NEW: EXTENDED TIMELINE EVENT ==========

class ExtendedTimelineEvent(BaseModel):
    """
    Maps to patient.timeline_events[] in the payload.
    Extends TimelineEvent with ISO timestamp, event type coding, and source author.
    TimelineEvent is preserved for the existing pipeline interface.
    """
    effective_at: str = Field(
        ...,
        description="ISO 8601 date or datetime of the event"
    )
    event_type_display: str
    detail: str
    source_author: Optional[str] = None


# ========== NEW: FULL PATIENT RECORD ==========

class FullPatientRecord(BaseModel):
    """
    Top-level patient record assembled by the Clinical History Agent
    from the enriched payload structure.
    PatientData is preserved for the existing orchestrator interface —
    this model carries the additional clinical depth.
    """
    demographics: ExtendedPatientDemographics
    gp_practice: ExtendedGPDetails
    referral: ReferralRecord
    conditions: List[ExtendedCondition] = Field(default_factory=list)
    medications: List[ExtendedMedication] = Field(default_factory=list)
    allergies: List[ExtendedAllergy] = Field(default_factory=list)
    social_history: Optional[SocialHistory] = None
    prior_assessments: List[PriorCognitiveAssessment] = Field(default_factory=list)
    timeline_events: List[ExtendedTimelineEvent] = Field(default_factory=list)


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


# ========== NEW: PROFILER SUPPORTING MODELS ==========

class PolypharmacyFlag(BaseModel):
    """
    Synthesised by the Profiler agent from ExtendedMedication list.
    Captures medication interaction and compliance risks (e.g. Cases 2, 3).
    """
    medication_count: int
    high_risk_medications: List[str] = Field(
        default_factory=list,
        description="Medications warranting specific monitoring, e.g. anticoagulants, NSAIDs"
    )
    interaction_concerns: List[str] = Field(default_factory=list)
    compliance_risk_noted: bool = False
    clinical_note: Optional[str] = None


class CapacityFlag(BaseModel):
    """
    Raised by the Profiler agent where Mental Capacity Act assessment is indicated.
    Directly triggered by Case 3 (capacity concern post-stroke).
    """
    mca_assessment_required: bool
    specific_decision: Optional[str] = Field(
        None,
        description="The specific decision under question, e.g. 'transfer to rehabilitation unit'"
    )
    poa_in_place: Optional[bool] = None
    poa_holder: Optional[str] = None
    clinical_note: Optional[str] = None


class CarerRiskFlag(BaseModel):
    """
    Raised by the Profiler agent where carer wellbeing is a concern.
    Triggered by Case 5 (husband carer fatigue) and Case 2 (wife as primary carer).
    """
    carer_name: Optional[str] = None
    carer_relationship: Optional[InformantRelationship] = None
    fatigue_noted: bool = False
    own_health_concerns: Optional[str] = Field(
        None,
        description="Carer's own health issues (e.g. Case 5: husband has mild heart failure)"
    )
    support_services_in_place: Optional[bool] = None
    referral_recommended: Optional[bool] = None


class DrivingRiskFlag(BaseModel):
    """
    Raised by the Profiler agent where driving status intersects with cognitive concern.
    Relevant to Cases 1 and 4.
    """
    currently_driving: bool
    dvla_notified: Optional[bool] = None
    notification_date: Optional[str] = None
    clinical_note: Optional[str] = None


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


# ========== NEW: PLANNER SUPPORTING MODELS ==========

class InformantAssessmentPlan(BaseModel):
    """
    Captures the informant-based assessment component of the plan.
    Relevant when next_of_kin.informant_available_for_iqcode is True
    (Cases 2, 3, 4, 5).
    """
    informant_name: str
    informant_relationship: InformantRelationship
    instruments_planned: List[str] = Field(
        default_factory=list,
        description="e.g. ['IQCode', 'GPCOG-Informant']"
    )
    iqcode_indicated: bool = False
    rationale: Optional[str] = None


class NeuropsychologicalReferral(BaseModel):
    """
    Captures a recommendation for formal neuropsychological assessment.
    Directly indicated in Case 4 referral text.
    """
    recommended: bool
    rationale: Optional[str] = None
    priority: Optional[Priority] = None
    domains_of_concern: List[CognitiveDomain] = Field(default_factory=list)
    nice_guidance_reference: Optional[str] = None


class CommunicationAdaptation(BaseModel):
    """
    Captures assessment adaptations required for communication barriers.
    Directly indicated in Case 3 (expressive aphasia).
    """
    adaptation_required: bool
    reason: Optional[str] = Field(
        None,
        description="e.g. 'Expressive aphasia post left MCA stroke'"
    )
    strategies: List[str] = Field(
        default_factory=list,
        description="e.g. ['Non-verbal response options', 'Extended time allowance', "
                    "'SaLT involvement']"
    )
    instruments_to_modify: List[str] = Field(default_factory=list)
    instruments_contraindicated: List[str] = Field(
        default_factory=list,
        description="Instruments requiring verbal fluency that cannot be adapted"
    )


class ExtendedAssessmentPlan(BaseModel):
    """
    Extends AssessmentPlan with informant, neuropsychology, and
    communication adaptation components.
    AssessmentPlan is preserved for the existing orchestrator interface.
    """
    base_plan: AssessmentPlan
    informant_plan: Optional[InformantAssessmentPlan] = None
    neuropsychological_referral: Optional[NeuropsychologicalReferral] = None
    communication_adaptations: Optional[CommunicationAdaptation] = None
    capacity_assessment_required: bool = False
    differential_diagnosis_focus: List[str] = Field(
        default_factory=list,
        description="Primary differentials driving instrument selection, "
                    "e.g. ['PDD', 'DLB', 'Vascular dementia', "
                    "'Alzheimer's disease']"
    )


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


# ========== NEW: QA SUPPORTING MODELS ==========

class MedicationSafetyCheck(BaseModel):
    """
    Granular medication safety output from the QA agent.
    Covers anticoagulant dosing (Cases 2, 3), NSAID risk (Case 5),
    and cholinesterase inhibitor monitoring (Cases 2, 4).
    """
    medication_name: str
    check_type: str = Field(
        ...,
        description="e.g. 'renal_dose_adjustment', 'nsaid_elderly_risk', "
                    "'cholinesterase_monitoring', 'anticoagulant_review'"
    )
    passed: bool
    detail: Optional[str] = None
    action_required: Optional[str] = None


class CodingValidationResult(BaseModel):
    """
    QA-level validation of clinical codes in the output.
    Extends the existing Condition.validate_clinical_code validator
    with a reportable result model for the QA agent.
    """
    field: str
    code_value: str
    code_system: str = Field(
        ...,
        description="e.g. 'SNOMED CT', 'ICD-10', 'dm+d', 'LOINC'"
    )
    valid: bool
    rejection_reason: Optional[str] = None


class ExtendedQAResult(BaseModel):
    """
    Extends QAResult with medication safety and coding validation layers.
    QAResult is preserved for the existing orchestrator interface.
    """
    base_result: QAResult
    medication_safety_checks: List[MedicationSafetyCheck] = Field(default_factory=list)
    coding_validation: List[CodingValidationResult] = Field(default_factory=list)
    capacity_assessment_flagged: bool = Field(
        False,
        description="True if QA confirms MCA assessment is indicated and planned"
    )
    informant_corroboration_available: bool = Field(
        False,
        description="True if an informant is available and planned into the assessment"
    )


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


# ========== NEW: EXTENDED ORCHESTRATOR OUTPUT ==========

class ExtendedAgentContext(BaseModel):
    """
    Extends AgentContext with the richer data models produced
    by the extended pipeline.
    AgentContext is preserved for backward compatibility.
    """
    encounter_id: str
    full_patient_record: Optional[FullPatientRecord] = None
    patient_history: Optional[ClinicalHistory] = None
    patient_profile: Optional[PatientProfile] = None
    polypharmacy_flag: Optional[PolypharmacyFlag] = None
    capacity_flag: Optional[CapacityFlag] = None
    carer_risk_flag: Optional[CarerRiskFlag] = None
    driving_risk_flag: Optional[DrivingRiskFlag] = None
    risk_flags: Optional[List[RiskFlag]] = None
    assessment_plan: Optional[AssessmentPlan] = None
    extended_assessment_plan: Optional[ExtendedAssessmentPlan] = None
    audit_trail: List[AuditEntry] = Field(default_factory=list)


class ExtendedOrchestratorOutput(BaseModel):
    """
    Extends OrchestratorOutput with the full enriched record and
    extended plan/QA models.
    OrchestratorOutput is preserved for the existing interface contract.
    """
    # Preserved original fields
    patient_data: PatientData
    clinical_history: ClinicalHistory
    patient_profile: PatientProfile
    assessment_plan: AssessmentPlan
    clinical_brief: ClinicalBrief
    qa_result: QAResult
    metadata: OrchestratorMetadata

    # Extended fields
    full_patient_record: Optional[FullPatientRecord] = None
    extended_assessment_plan: Optional[ExtendedAssessmentPlan] = None
    extended_qa_result: Optional[ExtendedQAResult] = None