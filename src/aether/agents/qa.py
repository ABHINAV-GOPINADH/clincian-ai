from crewai import Agent, Task
from aether.config.llm_config import agent_llm  # CHANGED
from aether.schemas.clinical import (
    PatientData, ClinicalHistory, PatientProfile, AssessmentPlan, ClinicalBrief, QAResult
)
from aether.tools.rag_tool import nice_guidance_tool, nice_rag
from aether.utils.logger import logger
import re  # ADDED


class QAAgent:
    """Clinical Quality Assurance Agent."""
    
    def __init__(self):
        self.llm = agent_llm  # CHANGED
        
        self.agent = Agent(
            role="Clinical Quality Assurance Specialist",
            goal="Validate clinical accuracy, NICE compliance, and safety of assessment outputs",
            backstory=(
                "You are a clinical governance lead with expertise in dementia care pathways. "
                "You perform rigorous quality checks ensuring clinical accuracy and patient safety."
            ),
            llm=self.llm,
            tools=[nice_guidance_tool],
            verbose=True,
            allow_delegation=False,
        )
    
    def create_task(
        self,
        patient_data: PatientData,
        clinical_history: ClinicalHistory,
        patient_profile: PatientProfile,
        assessment_plan: AssessmentPlan,
        clinical_brief: ClinicalBrief
    ) -> Task:
        """Create QA validation task."""
        nice_compliance = nice_rag.retrieve_guidance(
            "NICE NG97 dementia assessment mandatory requirements quality standards",
            top_k=3
        )
        nice_compliance_text = "\n\n---\n\n".join([doc.page_content for doc in nice_compliance])
        
        return Task(
            description=f"""
Perform comprehensive quality assurance validation.

Validate across:
1. Clinical accuracy (score 0-100)
2. NICE NG97 compliance
3. Data completeness (%)
4. Safety checks

Determine overall status: GREEN (≥90), AMBER (70-89), or RED (<70)

Return ONLY valid JSON. No markdown.
            """.strip(),
            expected_output="Quality assurance validation report as JSON",
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
        assessment_plan: AssessmentPlan,
        clinical_brief: ClinicalBrief
    ) -> QAResult:
        """Execute the QA agent."""
        logger.info("QAAgent: Starting quality validation")
        
        task = self.create_task(
            patient_data, clinical_history, patient_profile, assessment_plan, clinical_brief
        )
        result = task.execute()
        
        result = self._clean_json_response(result)  # ADDED
        
        qa_result = QAResult.model_validate_json(result)
        
        logger.info(f"QAAgent: Validation complete - Status: {qa_result.overall_status}")
        return qa_result