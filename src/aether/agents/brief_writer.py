from datetime import date
from crewai import Agent, Task, Crew
from aether.config.llm_config import agent_llm 
from aether.schemas.clinical import (
    PatientData, ClinicalHistory, PatientProfile, AssessmentPlan, ClinicalBrief
)
from aether.utils.logger import logger

class BriefWriterAgent:
    """Clinical Documentation Specialist Agent."""
    
    def __init__(self):
        self.llm = agent_llm 
        
        self.agent = Agent(
            role="Clinical Documentation Specialist",
            goal="Synthesize comprehensive clinical information into concise, actionable one-page briefs",
            backstory=(
                "You are an expert clinical writer with a background in psychiatry and "
                "neuropsychology. You excel at distilling complex clinical data into clear, "
                "professional, and highly structured clinical briefs for referring GPs and specialists."
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
Create a comprehensive one-page clinical brief based on the following synthesized data.
Current Date: {today}

--- 1. PATIENT DEMOGRAPHICS ---
{patient_data.model_dump_json(indent=2)}

--- 2. CLINICAL HISTORY ---
{clinical_history.model_dump_json(indent=2)}

--- 3. RISK PROFILE & COGNITIVE CONCERNS ---
{patient_profile.model_dump_json(indent=2)}

--- 4. RECOMMENDED ASSESSMENT PLAN ---
{assessment_plan.model_dump_json(indent=2)}

Synthesize this data into a professional clinical brief with:
1. Header (patient details, date)
2. Executive summary (concise overview of the patient and primary referral reason)
3. Presenting concerns (extracted from history and profile)
4. Relevant history (conditions, meds, allergies)
5. Risk summary (highlighting high-priority safety/clinical risks)
6. Recommended assessments (summarize the planned instruments and rationale)
7. Key considerations
8. NICE guidance alignment (how the plan aligns with NG97)

CRITICAL RULE: Return ONLY a valid JSON object matching the ClinicalBrief schema exactly. Do not invent any new data.
            """.strip(),
            expected_output="One-page clinical brief as a structured JSON object",
            agent=self.agent,
            output_pydantic=ClinicalBrief # <-- CrewAI Magic
        )
    
    def execute(
        self,
        patient_data: PatientData,
        clinical_history: ClinicalHistory,
        patient_profile: PatientProfile,
        assessment_plan: AssessmentPlan
    ) -> ClinicalBrief:
        """Execute the brief writer agent."""
        logger.info("Step 1: BriefWriterAgent execution started")
        
        task = self.create_task(patient_data, clinical_history, patient_profile, assessment_plan)
        
        logger.info("Step 2: Initializing CrewAI environment")
        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            verbose=False,
        )
        
        logger.info("Step 3: Kicking off Crew... (Waiting for LLM to synthesize data)")
        result = crew.kickoff()
        
        # SAFETY NET: Check if CrewAI's parser failed
        if result.pydantic is None:
            logger.warning("CrewAI native parsing returned None. Falling back to manual validation...")
            import re
            raw_text = result.raw
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                raw_text = json_match.group(0)
            clinical_brief = ClinicalBrief.model_validate_json(raw_text)
        else:
            clinical_brief = result.pydantic 
        
        logger.info("Step 4: Success! Brief completed.")
        return clinical_brief