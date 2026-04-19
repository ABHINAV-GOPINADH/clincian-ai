from crewai import Agent, Task, Crew
from aether.config.llm_config import agent_llm 
from aether.schemas.clinical import PatientData, ClinicalHistory, PatientProfile
from aether.utils.logger import logger
from aether.config.llm_config import get_strict_clinical_llm, get_mistral

class ProfilerAgent:
    """Clinical Risk Profiler Agent - Identifies risks and cognitive indicators."""
    
    def __init__(self):
        self.llm = get_mistral()  # Using Mistral for more accurate parsing of clinical text
        
        self.agent = Agent(
            role="Clinical Data Extraction Specialist",
            goal="Extract explicitly stated clinical risks and cognitive indicators exactly as written in the source text.",
            backstory=(
                "You are a strict, literal data parser. Your job is to extract facts from clinical notes "
                "into a structured JSON format. You NEVER infer medical correlations, you NEVER invent "
                "treatment plans, and you NEVER diagnose the patient. If a piece of information is not "
                "explicitly written in the source text, you return null."
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
Perform a strict data extraction of risk flags and cognitive indicators based ONLY on the provided text.

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

EXTRACTION RULES:
1. RISK FLAGS: Extract only the risks explicitly mentioned in the text. DO NOT invent risks like "falls" unless the word "falls" or "falling" is in the text.
2. COGNITIVE INDICATORS: Extract only the specific memory or cognitive issues stated.
3. MITIGATION STRATEGIES: Leave this field null/empty UNLESS a specific treatment or action is explicitly requested in the text. Do not provide medical advice.
4. COMPLEXITY SCORE (1-10): Assign a score based solely on the number of active conditions and medications listed.
5. INFORMATION GAPS: List only what is obviously missing for a standard referral.

CRITICAL RULE: Return ONLY a valid JSON object matching the PatientProfile schema. Do not include markdown formatting like ```json.
            """.strip(),
            expected_output="Strictly extracted patient risk profile as JSON",
            agent=self.agent,
            output_pydantic=PatientProfile
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