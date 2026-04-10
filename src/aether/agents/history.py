from crewai import Agent, Task
from aether.schemas.clinical import PatientData, ClinicalHistory
from aether.utils.logger import logger
from aether.config.llm_config import agent_llm
import re


class ClinicalHistoryAgent:
    """Clinical Historian Agent - Structures comprehensive patient clinical history."""
    
    def __init__(self):
        self.llm = agent_llm  # CHANGED: Using Gemini
        
        self.agent = Agent(
            role="Clinical Historian",
            goal="Structure comprehensive patient clinical history with SNOMED-CT coding",
            backstory=(
                "You are a senior clinical informaticist specialized in dementia care pathways. "
                "You have deep knowledge of SNOMED-CT coding, medication classification, and timeline "
                "reconstruction from narrative clinical notes."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )
    
    def create_task(self, patient_data: PatientData, clinical_notes: str = None) -> Task:
        """Create history structuring task."""
        return Task(
            description=f"""
Analyze and structure the clinical history for this patient.

PATIENT CONTEXT:
Name: {patient_data.name.first} {patient_data.name.last}
NHS#: {patient_data.nhs_number}
Age: {patient_data.age} | Gender: {patient_data.gender}

REFERRAL REASON:
{patient_data.referral_reason}

{f"ADDITIONAL CLINICAL NOTES:\n{clinical_notes}" if clinical_notes else ""}

Extract and structure:
1. Medical conditions with SNOMED-CT codes
2. Current medications with dosage and indication
3. Known allergies
4. Past psychiatric/cognitive assessments
5. Timeline of significant clinical events

Return ONLY valid JSON. No markdown, no explanations.
            """.strip(),
            expected_output="Structured clinical history as JSON",
            agent=self.agent,
        )
    
    def _clean_json_response(self, result: str) -> str:
        """Clean Gemini response to extract pure JSON."""
        result = result.strip()
        if result.startswith("```"):
            result = re.sub(r'^```(?:json)?\n', '', result)
            result = re.sub(r'\n```$', '', result)
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            result = json_match.group(0)
        return result.strip()
    
    def execute(self, patient_data: PatientData, clinical_notes: str = None) -> ClinicalHistory:
        """Execute the clinical history agent."""
        logger.info("ClinicalHistoryAgent: Structuring patient history")
        
        task = self.create_task(patient_data, clinical_notes)
        result = task.execute()
        
        # CHANGED: Clean Gemini response
        result = self._clean_json_response(result)
        
        clinical_history = ClinicalHistory.model_validate_json(result)
        
        logger.info(f"ClinicalHistoryAgent: Structured {len(clinical_history.conditions)} conditions")
        return clinical_history