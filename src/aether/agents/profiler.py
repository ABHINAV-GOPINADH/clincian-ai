from crewai import Agent, Task
from langchain_anthropic import ChatAnthropic
from aether.config.settings import settings
from aether.schemas.clinical import PatientData, ClinicalHistory, PatientProfile
from aether.utils.logger import logger


class ProfilerAgent:
    """Clinical Risk Profiler Agent - Identifies risks and cognitive indicators."""
    
    def __init__(self):
        self.llm = ChatAnthropic(
            model=settings.claude_model,
            anthropic_api_key=settings.anthropic_api_key,
            temperature=0.4,
            max_tokens=8192,
        )
        
        self.agent = Agent(
            role="Clinical Risk Profiler",
            goal="Identify clinical risks, safety concerns, and cognitive indicators for dementia assessment",
            backstory=(
                "You are a consultant psychiatrist specializing in old-age psychiatry and "
                "dementia risk assessment. You have 20 years of experience identifying subtle cognitive "
                "indicators and safety risks in complex patients."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )
    
    def create_task(self, patient_data: PatientData, clinical_history: ClinicalHistory) -> Task:
        """Create profiling task."""
        conditions_text = "\n".join([
            f"- {c.display} ({c.status}{f', {c.severity}' if c.severity else ''})"
            for c in clinical_history.conditions
        ])
        
        medications_text = "\n".join([
            f"- {m.name}{f' {m.dosage}' if m.dosage else ''}"
            for m in clinical_history.medications
        ])
        
        return Task(
            description=f"""
Perform comprehensive risk profiling and cognitive indicator analysis.

PATIENT:
{patient_data.name.first} {patient_data.name.last}, {patient_data.age}yo {patient_data.gender}

REFERRAL REASON:
{patient_data.referral_reason}

CLINICAL CONDITIONS:
{conditions_text}

MEDICATIONS:
{medications_text}

ALLERGIES:
{', '.join(clinical_history.allergies) if clinical_history.allergies else 'None documented'}

Analyze and identify:

1. RISK FLAGS across categories:
   - Clinical risks (falls, delirium, polypharmacy)
   - Safety concerns (wandering, self-neglect, capacity)
   - Medication interactions or concerns
   - Social risks (isolation, carer strain)
   - Cognitive deterioration pace

2. COGNITIVE INDICATORS by domain:
   - Memory (episodic, semantic, working)
   - Attention and concentration
   - Language and communication
   - Executive function
   - Visuospatial abilities

3. COMPLEXITY SCORE (1-10) based on:
   - Number of comorbidities
   - Polypharmacy
   - Psychosocial factors
   - Diagnostic uncertainty

4. INFORMATION GAPS that need clarification

Provide evidence-based reasoning for each flag.

Return the data as valid JSON matching the PatientProfile schema.
            """.strip(),
            expected_output="Comprehensive patient risk and cognitive profile as JSON",
            agent=self.agent,
        )
    
    def execute(self, patient_data: PatientData, clinical_history: ClinicalHistory) -> PatientProfile:
        """Execute the profiler agent."""
        logger.info("ProfilerAgent: Analyzing patient profile")
        
        task = self.create_task(patient_data, clinical_history)
        result = task.execute()
        
        patient_profile = PatientProfile.model_validate_json(result)
        
        logger.info(f"ProfilerAgent: Identified {len(patient_profile.risk_flags)} risk flags")
        return patient_profile