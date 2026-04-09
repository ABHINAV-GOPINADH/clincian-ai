from crewai import Agent, Task
from langchain_anthropic import ChatAnthropic
from aether.config.settings import settings
from aether.schemas.clinical import (
    PatientData, ClinicalHistory, PatientProfile, AssessmentPlan, ClinicalBrief, QAResult
)
from aether.tools.rag_tool import nice_guidance_tool, nice_rag
from aether.utils.logger import logger


class QAAgent:
    """Clinical Quality Assurance Agent - Validates clinical accuracy and safety."""
    
    def __init__(self):
        self.llm = ChatAnthropic(
            model=settings.claude_model,
            anthropic_api_key=settings.anthropic_api_key,
            temperature=0.2,
            max_tokens=16384,
        )
        
        self.agent = Agent(
            role="Clinical Quality Assurance Specialist",
            goal="Validate clinical accuracy, NICE compliance, and safety of assessment outputs",
            backstory=(
                "You are a clinical governance lead with expertise in dementia care pathways. "
                "You perform rigorous quality checks ensuring clinical accuracy, guideline compliance, "
                "and patient safety before any clinical output is released."
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
        # Retrieve NICE compliance requirements
        nice_compliance = nice_rag.retrieve_guidance(
            "NICE NG97 dementia assessment mandatory requirements quality standards",
            top_k=3
        )
        nice_compliance_text = "\n\n---\n\n".join([doc.page_content for doc in nice_compliance])
        
        return Task(
            description=f"""
Perform comprehensive quality assurance validation.

PATIENT DATA:
{patient_data.model_dump_json(indent=2)}

CLINICAL BRIEF:
{clinical_brief.model_dump_json(indent=2)}

ASSESSMENT PLAN:
{assessment_plan.model_dump_json(indent=2)}

NICE NG97 REQUIREMENTS:
{nice_compliance_text}

Validate across four dimensions:

1. CLINICAL ACCURACY (score 0-100):
   - Data consistency (demographics, history align?)
   - Clinical logic (risks match conditions?)
   - Appropriate clinical language
   - No contradictions
   
   Flag issues as: error (blocks release), warning (review needed), info (minor)

2. NICE NG97 COMPLIANCE:
   - Assessment instruments meet guidelines
   - Mandatory components included
   - Contraindications considered
   
   List any gaps in compliance

3. DATA COMPLETENESS (% complete):
   - Required fields populated
   - No "unknown" in critical fields
   - Timeline consistency
   
   List missing fields

4. SAFETY CHECKS:
   - High-risk flags addressed
   - Contraindications noted
   - Safety mitigation present
   
   List any safety concerns

OVERALL STATUS:
- GREEN: Ready for clinical use (score ≥90, compliant, no safety issues)
- AMBER: Needs review (score 70-89, minor gaps)
- RED: Must fix before use (score <70, non-compliant, or safety issues)

Provide actionable recommendations for any issues found.

Return the data as valid JSON matching the QAResult schema.
            """.strip(),
            expected_output="Quality assurance validation report as JSON",
            agent=self.agent,
        )
    
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
        
        qa_result = QAResult.model_validate_json(result)
        
        logger.info(f"QAAgent: Validation complete - Status: {qa_result.overall_status}")
        return qa_result    