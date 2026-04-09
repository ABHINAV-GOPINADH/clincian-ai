from datetime import date
from crewai import Agent, Task
from langchain_anthropic import ChatAnthropic
from aether.config.settings import settings
from aether.schemas.clinical import (
    PatientData, ClinicalHistory, PatientProfile, AssessmentPlan, ClinicalBrief
)
from aether.utils.logger import logger


class BriefWriterAgent:
    """Clinical Documentation Specialist Agent - Creates concise clinical briefs."""
    
    def __init__(self):
        self.llm = ChatAnthropic(
            model=settings.claude_model,
            anthropic_api_key=settings.anthropic_api_key,
            temperature=0.5,
            max_tokens=16384,
        )
        
        self.agent = Agent(
            role="Clinical Documentation Specialist",
            goal="Synthesize comprehensive clinical information into concise, actionable one-page briefs",
            backstory=(
                "You are an expert clinical writer with a background in psychiatry and "
                "neuropsychology. You excel at distilling complex clinical data into clear, structured "
                "briefs that support clinical decision-making."
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
Create a comprehensive one-page clinical brief for the assessing clinician.

PATIENT DATA:
{patient_data.model_dump_json(indent=2)}

CLINICAL HISTORY:
{clinical_history.model_dump_json(indent=2)}

RISK PROFILE:
{patient_profile.model_dump_json(indent=2)}

ASSESSMENT PLAN:
{assessment_plan.model_dump_json(indent=2)}

Synthesize into a clinical brief with:

1. HEADER:
   - Patient name, NHS#, DOB
   - Assessment date: {today}
   - Clinician: [To be assigned]

2. EXECUTIVE SUMMARY (max 500 chars):
   - Age, referral reason
   - Key clinical concerns
   - Assessment priority level

3. PRESENTING CONCERNS:
   - Bullet list of cognitive/functional concerns
   - Risk flags (prioritize high/critical)

4. RELEVANT HISTORY:
   - Medical: significant conditions
   - Psychiatric: previous diagnoses, treatments
   - Social: living situation, support network

5. RISK SUMMARY:
   - Critical and high-severity flags only
   - Mitigation strategies

6. RECOMMENDED ASSESSMENTS:
   - Essential instruments with rationale
   - Total estimated time
   - Special considerations

7. KEY CONSIDERATIONS:
   - Clinical decision points
   - Safety concerns
   - Information gaps

8. NICE GUIDANCE ALIGNMENT:
   - How plan meets NG97 standards
   - Compliance notes

Write professionally, concisely, using clinical terminology.

Return the data as valid JSON matching the ClinicalBrief schema.
            """.strip(),
            expected_output="One-page clinical brief as JSON",
            agent=self.agent,
        )
    
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
        
        clinical_brief = ClinicalBrief.model_validate_json(result)
        
        logger.info("BriefWriterAgent: Brief completed")
        return clinical_brief