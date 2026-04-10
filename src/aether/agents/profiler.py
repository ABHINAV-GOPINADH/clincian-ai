from crewai import Agent, Task
from aether.config.llm_config import agent_llm  # CHANGED
from aether.schemas.clinical import PatientData, ClinicalHistory, PatientProfile
from aether.utils.logger import logger
import re  # ADDED


class ProfilerAgent:
    """Clinical Risk Profiler Agent - Identifies risks and cognitive indicators."""
    
    def __init__(self):
        self.llm = agent_llm  # CHANGED
        
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
1. RISK FLAGS across categories
2. COGNITIVE INDICATORS by domain
3. COMPLEXITY SCORE (1-10)
4. INFORMATION GAPS

Return ONLY valid JSON. No markdown.
            """.strip(),
            expected_output="Comprehensive patient risk and cognitive profile as JSON",
            agent=self.agent,
        )
    
    def _clean_json_response(self, result: str) -> str:  # ADDED
        """Clean Gemini response."""
        result = result.strip()
        if result.startswith("```"):
            result = re.sub(r'^```(?:json)?\n', '', result)
            result = re.sub(r'\n```$', '', result)
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            result = json_match.group(0)
        return result.strip()
    
    def execute(self, patient_data: PatientData, clinical_history: ClinicalHistory) -> PatientProfile:
        """Execute the profiler agent."""
        logger.info("ProfilerAgent: Analyzing patient profile")
        
        task = self.create_task(patient_data, clinical_history)
        result = task.execute()
        
        result = self._clean_json_response(result)  # ADDED
        
        patient_profile = PatientProfile.model_validate_json(result)
        
        logger.info(f"ProfilerAgent: Identified {len(patient_profile.risk_flags)} risk flags")
        return patient_profile