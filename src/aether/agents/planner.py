from crewai import Agent, Task
from aether.config.llm_config import agent_llm  # CHANGED
from aether.schemas.clinical import PatientData, PatientProfile, AssessmentPlan
from aether.tools.rag_tool import nice_guidance_tool, nice_rag
from aether.utils.logger import logger
import re  # ADDED


class AssessmentPlannerAgent:
    """Neuropsychological Assessment Planner Agent."""
    
    def __init__(self):
        self.llm = agent_llm  # CHANGED
        
        self.agent = Agent(
            role="Neuropsychological Assessment Planner",
            goal="Design evidence-based, NICE NG97-compliant assessment batteries",
            backstory=(
                "You are a principal clinical psychologist specializing in dementia assessment. "
                "You design tailored neuropsychological batteries aligned with NICE NG97 guidelines."
            ),
            llm=self.llm,
            tools=[nice_guidance_tool],
            verbose=True,
            allow_delegation=False,
        )
    
    def create_task(self, patient_data: PatientData, patient_profile: PatientProfile) -> Task:
        """Create assessment planning task."""
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

PATIENT: {patient_data.name.first} {patient_data.name.last}, {patient_data.age}yo

RISK SUMMARY:
{risk_flags_text}

COGNITIVE CONCERNS:
{cognitive_concerns_text}

NICE NG97 GUIDANCE:
{nice_guidance_text}

Design an assessment plan with instruments, prioritization, rationale, and duration.

Return ONLY valid JSON. No markdown.
            """.strip(),
            expected_output="NICE-compliant assessment battery as JSON",
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
    
    def execute(self, patient_data: PatientData, patient_profile: PatientProfile) -> AssessmentPlan:
        """Execute the assessment planner agent."""
        logger.info("AssessmentPlannerAgent: Designing assessment battery")
        
        task = self.create_task(patient_data, patient_profile)
        result = task.execute()
        
        result = self._clean_json_response(result)  # ADDED
        
        assessment_plan = AssessmentPlan.model_validate_json(result)
        
        logger.info(f"AssessmentPlannerAgent: Planned {len(assessment_plan.instruments)} instruments")
        return assessment_plan