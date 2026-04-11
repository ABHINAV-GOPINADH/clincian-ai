from crewai import Agent, Task, Crew
from aether.config.llm_config import agent_llm 
from aether.schemas.clinical import (
    PatientData, ClinicalHistory, PatientProfile, AssessmentPlan, ClinicalBrief, QAResult
)
from aether.tools.rag_tool import nice_guidance_tool, nice_rag
from aether.utils.logger import logger

class QAAgent:
    """Clinical Quality Assurance Agent."""
    
    def __init__(self):
        self.llm = agent_llm 
        
        self.agent = Agent(
            role="Clinical Quality Assurance Specialist",
            goal="Validate clinical accuracy, NICE compliance, and safety of assessment outputs",
            backstory=(
                "You are a clinical governance lead with expertise in dementia care pathways. "
                "You perform rigorous quality checks ensuring clinical accuracy, patient safety, "
                "and strict alignment with clinical guidelines."
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
        # 1. Fetch RAG Context safely
        try:
            nice_compliance = nice_rag.retrieve_guidance(
                "NICE NG97 dementia assessment mandatory requirements quality standards",
                top_k=3
            )
            nice_compliance_text = "\n\n---\n\n".join([doc.page_content for doc in nice_compliance])
        except Exception as e:
            logger.warning(f"RAG retrieval failed, using fallback knowledge: {e}")
            nice_compliance_text = "Ensure standard NICE NG97 compliance and patient safety."
        
        return Task(
            description=f"""
Perform comprehensive quality assurance validation on the generated clinical brief and assessment plan.

--- INPUT DATA FOR AUDIT ---
PATIENT DEMOGRAPHICS: {patient_data.model_dump_json()}
CLINICAL HISTORY: {clinical_history.model_dump_json()}
RISK PROFILE: {patient_profile.model_dump_json()}
PROPOSED ASSESSMENT PLAN: {assessment_plan.model_dump_json()}
FINAL CLINICAL BRIEF: {clinical_brief.model_dump_json()}

--- NICE NG97 QUALITY STANDARDS ---
{nice_compliance_text}

Validate across the following dimensions:
1. Clinical accuracy (score 0-100): Does the brief accurately reflect the raw history and risks?
2. NICE NG97 compliance: Does the assessment plan meet the provided standards?
3. Data completeness (%): Are any major fields missing or marked 'Unknown' unnecessarily?
4. Safety checks: Have high-severity risks been properly mitigated in the plan?

Determine overall status: GREEN (≥90), AMBER (70-89), or RED (<70)

CRITICAL RULE: Return ONLY a valid JSON object matching the QAResult schema exactly. Do not invent metrics not supported by the data.
            """.strip(),
            expected_output="Quality assurance validation report as a structured JSON object",
            agent=self.agent,
            output_pydantic=QAResult # <-- CrewAI Magic
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
        logger.info("Step 1: QAAgent execution started")
        
        task = self.create_task(
            patient_data, clinical_history, patient_profile, assessment_plan, clinical_brief
        )
        
        logger.info("Step 2: Initializing CrewAI environment")
        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            verbose=False,
        )
        
        logger.info("Step 3: Kicking off Crew... (Waiting for LLM audit)")
        result = crew.kickoff()
        
        # SAFETY NET: Fallback parser
        if result.pydantic is None:
            logger.warning("CrewAI native parsing returned None. Falling back to manual validation...")
            import re
            raw_text = result.raw
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                raw_text = json_match.group(0)
            qa_result = QAResult.model_validate_json(raw_text)
        else:
            qa_result = result.pydantic 
        
        status = getattr(qa_result, 'overall_status', 'UNKNOWN')
        logger.info(f"Step 4: Validation complete - Status: {status}")
        return qa_result