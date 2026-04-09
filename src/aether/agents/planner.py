from crewai import Agent, Task
from langchain_anthropic import ChatAnthropic
from aether.config.settings import settings
from aether.schemas.clinical import PatientData, PatientProfile, AssessmentPlan
from aether.tools.rag_tool import nice_guidance_tool, nice_rag
from aether.utils.logger import logger


class AssessmentPlannerAgent:
    """Neuropsychological Assessment Planner Agent - Designs NICE-compliant assessment batteries."""
    
    def __init__(self):
        self.llm = ChatAnthropic(
            model=settings.claude_model,
            anthropic_api_key=settings.anthropic_api_key,
            temperature=0.3,
            max_tokens=16384,
        )
        
        self.agent = Agent(
            role="Neuropsychological Assessment Planner",
            goal="Design evidence-based, NICE NG97-compliant assessment batteries",
            backstory=(
                "You are a principal clinical psychologist specializing in dementia assessment. "
                "You design tailored neuropsychological batteries aligned with NICE NG97 guidelines, "
                "considering patient complexity, safety, and diagnostic precision."
            ),
            llm=self.llm,
            tools=[nice_guidance_tool],
            verbose=True,
            allow_delegation=False,
        )
    
    def create_task(self, patient_data: PatientData, patient_profile: PatientProfile) -> Task:
        """Create assessment planning task."""
        # Retrieve NICE guidance
        nice_guidance = nice_rag.retrieve_guidance(
            "NICE NG97 dementia assessment instruments cognitive testing recommendations",
            top_k=5
        )
        nice_guidance_text = "\n\n---\n\n".join([doc.page_content for doc in nice_guidance])
        
        risk_flags_text = "\n".join([
            f"- [{r.severity.upper()}] {r.category}: {r.description}"
            for r in patient_profile.risk_flags
        ])
        
        cognitive_concerns_text = "\n".join([
            f"- {c.domain}: {c.concern}"
            for c in patient_profile.cognitive_indicators
        ])
        
        return Task(
            description=f"""
Design a comprehensive, NICE NG97-compliant assessment battery.

PATIENT CONTEXT:
{patient_data.name.first} {patient_data.name.last}, {patient_data.age}yo
Referral: {patient_data.referral_reason}

RISK SUMMARY:
{risk_flags_text}

COGNITIVE CONCERNS:
{cognitive_concerns_text}

COMPLEXITY: {patient_profile.complexity_summary.score}/10
Factors: {', '.join(patient_profile.complexity_summary.factors)}

NICE NG97 GUIDANCE EXCERPTS:
{nice_guidance_text}

Design an assessment plan including:

1. INSTRUMENTS selection from:
   - Cognitive: ADAS-Cog, MMSE, MoCA, ACE-III
   - Functional: ADL, IADL
   - Staging: CDR
   - Behavioral: GDS (depression), NPI (neuropsychiatric)

2. PRIORITIZATION:
   - Essential (must-do)
   - Recommended (should-do)
   - Optional (nice-to-have)

3. RATIONALE for each instrument:
   - Clinical indication
   - Relevant to cognitive concerns
   - NICE guideline alignment

4. CONTRAINDICATIONS/SPECIAL CONSIDERATIONS:
   - Patient-specific adaptations
   - Safety concerns
   - Communication needs

5. ESTIMATED DURATION for each and total

Ensure alignment with NICE NG97 recommendations and cite guidance where applicable.

Return the data as valid JSON matching the AssessmentPlan schema.
            """.strip(),
            expected_output="NICE-compliant assessment battery as JSON",
            agent=self.agent,
        )
    
    def execute(self, patient_data: PatientData, patient_profile: PatientProfile) -> AssessmentPlan:
        """Execute the assessment planner agent."""
        logger.info("AssessmentPlannerAgent: Designing assessment battery")
        
        task = self.create_task(patient_data, patient_profile)
        result = task.execute()
        
        assessment_plan = AssessmentPlan.model_validate_json(result)
        
        logger.info(f"AssessmentPlannerAgent: Planned {len(assessment_plan.instruments)} instruments")
        return assessment_plan