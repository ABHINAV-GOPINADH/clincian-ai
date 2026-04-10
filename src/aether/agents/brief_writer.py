from datetime import date
from crewai import Agent, Task
from aether.config.llm_config import agent_llm  # CHANGED
from aether.schemas.clinical import (
    PatientData, ClinicalHistory, PatientProfile, AssessmentPlan, ClinicalBrief
)
from aether.utils.logger import logger
import re  # ADDED


class BriefWriterAgent:
    """Clinical Documentation Specialist Agent."""
    
    def __init__(self):
        self.llm = agent_llm  # CHANGED
        
        self.agent = Agent(
            role="Clinical Documentation Specialist",
            goal="Synthesize comprehensive clinical information into concise, actionable one-page briefs",
            backstory=(
                "You are an expert clinical writer with a background in psychiatry and "
                "neuropsychology. You excel at distilling complex clinical data into clear briefs."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )
    
    def create_task(
        self,
        patient_data: PatientData,
        clinical_history: ClinicalHistory,
        patient_profile: PatientProfile,
        assessment_plan: AssessmentPlan
    ) -> Task:
        """Create brief writing task."""
        today = date.today().isoformat()
        
        return Task(
            description=f"""
Create a comprehensive one-page clinical brief.

Use all provided data to synthesize a professional clinical brief with:
1. Header (patient details, date)
2. Executive summary (max 500 chars)
3. Presenting concerns
4. Relevant history
5. Risk summary
6. Recommended assessments
7. Key considerations
8. NICE guidance alignment

Return ONLY valid JSON. No markdown.
            """.strip(),
            expected_output="One-page clinical brief as JSON",
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
    
    def execute(
        self,
        patient_data: PatientData,
        clinical_history: ClinicalHistory,
        patient_profile: PatientProfile,
        assessment_plan: AssessmentPlan
    ) -> ClinicalBrief:
        """Execute the brief writer agent."""
        logger.info("BriefWriterAgent: Composing clinical brief")
        
        task = self.create_task(patient_data, clinical_history, patient_profile, assessment_plan)
        result = task.execute()
        
        result = self._clean_json_response(result)  # ADDED
        
        clinical_brief = ClinicalBrief.model_validate_json(result)
        
        logger.info("BriefWriterAgent: Brief completed")
        return clinical_brief