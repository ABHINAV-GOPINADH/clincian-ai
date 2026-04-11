from crewai import Agent, Task, Crew
from aether.config.llm_config import agent_llm 
from aether.schemas.clinical import PatientData, ClinicalHistory, PatientProfile
from aether.utils.logger import logger

class ProfilerAgent:
    """Clinical Risk Profiler Agent - Identifies risks and cognitive indicators."""
    
    def __init__(self):
        self.llm = agent_llm 
        
        self.agent = Agent(
            role="Clinical Risk Profiler",
            goal="Identify clinical risks, safety concerns, and cognitive indicators for dementia assessment",
            backstory=(
                "You are a consultant psychiatrist specializing in old-age psychiatry and "
                "dementia risk assessment. You have 20 years of experience identifying subtle cognitive "
                "indicators and safety risks in complex patients."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )
    
    def create_task(self, patient_data: PatientData, clinical_history: ClinicalHistory) -> Task:
        """Create profiling task."""
        # Safely handle missing severity or dosage attributes
        conditions_text = "\n".join([
            f"- {c.display} ({c.status}{f', {c.severity}' if getattr(c, 'severity', None) else ''})"
            for c in clinical_history.conditions
        ]) if clinical_history.conditions else "None recorded"
        
        medications_text = "\n".join([
            f"- {m.name}{f' {m.dosage}' if getattr(m, 'dosage', None) else ''}"
            for m in clinical_history.medications
        ]) if clinical_history.medications else "None recorded"
        
        return Task(
            description=f"""
Perform comprehensive risk profiling and cognitive indicator analysis.

PATIENT:
{patient_data.name.first} {patient_data.name.last}, {patient_data.age}yo {patient_data.gender}

REFERRAL REASON:
{patient_data.referral_reason}

CLINICAL CONDITIONS:
{conditions_text}

MEDICATIONS:
{medications_text}

ALLERGIES:
{', '.join(clinical_history.allergies) if clinical_history.allergies else 'None documented'}

Analyze and identify:
1. RISK FLAGS across categories (e.g., fall risk, living alone)
2. COGNITIVE INDICATORS by domain (e.g., memory, executive function)
3. COMPLEXITY SCORE (1-10)
4. INFORMATION GAPS

CRITICAL RULE: Return ONLY a valid JSON object matching the PatientProfile schema.
            """.strip(),
            expected_output="Comprehensive patient risk and cognitive profile as JSON",
            agent=self.agent,
            output_pydantic=PatientProfile # <-- The CrewAI Magic!
        )
    
    def execute(self, patient_data: PatientData, clinical_history: ClinicalHistory) -> PatientProfile:
        """Execute the profiler agent."""
        logger.info("Step 1: ProfilerAgent execution started")
        
        task = self.create_task(patient_data, clinical_history)
        
        logger.info("Step 2: Initializing CrewAI environment")
        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            verbose=False,
        )
        
        logger.info("Step 3: Kicking off Crew... (Waiting for LLM response)")
        result = crew.kickoff()
        
        # SAFETY NET: Check if CrewAI's parser failed
        if result.pydantic is None:
            logger.warning("CrewAI native parsing returned None. Falling back to manual validation...")
            import re
            raw_text = result.raw
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                raw_text = json_match.group(0)
            patient_profile = PatientProfile.model_validate_json(raw_text)
        else:
            patient_profile = result.pydantic 
        
        # Safe logging in case risk_flags is None
        num_flags = len(patient_profile.risk_flags) if getattr(patient_profile, 'risk_flags', None) else 0
        logger.info(f"Step 4: Success! Identified {num_flags} risk flags.")
        
        return patient_profile