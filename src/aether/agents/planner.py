from crewai import Agent, Task, Crew
from aether.config.llm_config import agent_llm 
from aether.schemas.clinical import PatientData, PatientProfile, AssessmentPlan
from aether.tools.rag_tool import nice_guidance_tool, nice_rag
from aether.utils.logger import logger

class AssessmentPlannerAgent:
    """Neuropsychological Assessment Planner Agent."""
    
    def __init__(self):
        self.llm = agent_llm 
        
        self.agent = Agent(
            role="Neuropsychological Assessment Planner",
            goal="Design evidence-based, NICE NG97-compliant assessment batteries",
            backstory=(
                "You are a principal clinical psychologist specializing in dementia assessment. "
                "You design tailored neuropsychological batteries aligned with NICE NG97 guidelines. "
                "You select specific, validated instruments (e.g., ACE-III, MoCA, GDS) based on patient risk profiles."
            ),
            llm=self.llm,
            tools=[nice_guidance_tool],
            verbose=True,
            allow_delegation=False,
        )
    
    def create_task(self, patient_data: PatientData, patient_profile: PatientProfile) -> Task:
        """Create assessment planning task."""
        # 1. Fetch RAG Context
        try:
            nice_guidance = nice_rag.retrieve_guidance(
                "NICE NG97 dementia assessment instruments cognitive testing recommendations",
                top_k=5
            )
            nice_guidance_text = "\n\n---\n\n".join([doc.page_content for doc in nice_guidance])
        except Exception as e:
            logger.warning(f"RAG retrieval failed, using fallback knowledge: {e}")
            nice_guidance_text = "Standard NICE NG97 recommendations apply. Prefer validated cognitive instruments."
        
        # 2. Format inputs safely
        risk_flags_text = "\n".join([
            f"- [{getattr(r, 'severity', 'UNKNOWN')}] {getattr(r, 'category', 'General')}: {r.description}"
            for r in patient_profile.risk_flags
        ]) if getattr(patient_profile, 'risk_flags', None) else "None identified"
        
        cognitive_concerns_text = "\n".join([
            f"- {getattr(c, 'domain', 'General')}: {c.concern}"
            for c in patient_profile.cognitive_indicators
        ]) if getattr(patient_profile, 'cognitive_indicators', None) else "None reported"
        
        return Task(
            description=f"""
Design a comprehensive, NICE NG97-compliant assessment battery for the following patient.

PATIENT: {patient_data.name.first} {patient_data.name.last}, {patient_data.age}yo

RISK SUMMARY:
{risk_flags_text}

COGNITIVE CONCERNS:
{cognitive_concerns_text}


Design an assessment plan mapping specific validated instruments to the cognitive concerns.
Include the instrument name, rationale for using it, prioritization (high/medium/low), and estimated duration.

CRITICAL RULE: Return ONLY a valid JSON object matching the AssessmentPlan schema exact keys. Do not deviate.
            """.strip(),
            expected_output="NICE-compliant assessment battery as a structured JSON object",
            agent=self.agent,
            output_pydantic=AssessmentPlan 
        )
    
    def execute(self, patient_data: PatientData, patient_profile: PatientProfile) -> AssessmentPlan:
        """Execute the assessment planner agent."""
        logger.info("Step 1: AssessmentPlannerAgent execution started")
        
        task = self.create_task(patient_data, patient_profile)
        
        logger.info("Step 2: Initializing CrewAI environment")
        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            verbose=False,
        )
        
        logger.info("Step 3: Kicking off Crew... (Waiting for LLM & RAG tools)")
        result = crew.kickoff()
        
        # SAFETY NET: Fallback parser
        if result.pydantic is None:
            logger.warning("CrewAI native parsing returned None. Falling back to manual validation...")
            import re
            raw_text = result.raw
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                raw_text = json_match.group(0)
            assessment_plan = AssessmentPlan.model_validate_json(raw_text)
        else:
            assessment_plan = result.pydantic 
        
        num_instruments = len(assessment_plan.instruments) if getattr(assessment_plan, 'instruments', None) else 0
        logger.info(f"Step 4: Success! Planned {num_instruments} instruments.")
        
        return assessment_plan