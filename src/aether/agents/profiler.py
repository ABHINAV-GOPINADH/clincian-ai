from crewai import Agent, Task, Crew
from aether.config.llm_config import structured_llm
from aether.schemas.clinical import PatientData, ClinicalHistory, PatientProfile
from aether.utils.logger import logger
import json, re

class ProfilerAgent:
    """Clinical Risk Profiler Agent - Identifies risks and cognitive indicators via AI."""
    
    def __init__(self):
        self.llm = structured_llm
        self.agent = Agent(
            role="Clinical Risk Assessor",
            goal="Analyze clinical data to autonomously identify health risks, cognitive indicators, and calculate patient complexity.",
            backstory=(
                "You are an expert clinical data analyst. Your job is to review patient referrals, "
                "medical history, and medications to identify potential safety, medical, social, "
                "and cognitive risks. Instead of just copying text, you synthesize the data to "
                "flag underlying risks (e.g., polypharmacy, fall risks, social isolation) and "
                "calculate a holistic complexity score."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    def _try_extract_json(self, text: str) -> dict:
        """Helper to cleanly extract JSON if the LLM wraps it in markdown blocks."""
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return {}

    def create_task(self, patient_data: PatientData, clinical_history: ClinicalHistory) -> Task:
        conditions_text = "\n".join([
            f"- {c.display} ({c.status}{f', {c.severity}' if getattr(c, 'severity', None) else ''})"
            for c in clinical_history.conditions
        ]) if clinical_history.conditions else "None recorded"
        
        medications_text = "\n".join([
            f"- {m.name}{f' {m.dosage}' if getattr(m, 'dosage', None) else ''}"
            for m in clinical_history.medications
        ]) if clinical_history.medications else "None recorded"

        events_text = "\n".join([
            f"- {e.description}" for e in getattr(clinical_history, "events", [])
        ]) if getattr(clinical_history, "events", None) else "None recorded"
        
        return Task(
            description=f"""
Analyze the following patient data and extract a comprehensive clinical risk profile.

PATIENT:
{patient_data.name.first} {patient_data.name.last}, {patient_data.age}yo {patient_data.gender}

REFERRAL REASON:
{patient_data.referral_reason}

CLINICAL CONDITIONS:
{conditions_text}

MEDICATIONS:
{medications_text}

EVENTS/TIMELINE:
{events_text}

ALLERGIES:
{', '.join(clinical_history.allergies) if clinical_history.allergies else 'None documented'}

ANALYSIS INSTRUCTIONS:
1. RISK FLAGS: Identify and categorize all clinical, social, and safety risks. 
   - Look for implicit risks like polypharmacy (if 4+ medications are listed), fall risks (if mobility issues or past falls are mentioned), social risks (living alone/homeless), and sensory impairments.
   - Categorize as "cognitive", "medication", or "safety".
2. COGNITIVE INDICATORS: Identify any specific memory, confusion, or cognitive issues.
3. COMPLEXITY SCORE: Calculate a score from 0-100 based on the patient's age, condition severity, medication volume, and social instability. Output this as an integer.
4. INFORMATION GAPS: Note any obvious missing information critical to patient care.

OUTPUT FORMAT (JSON only, matching the exact schema below):
{{
  "risk_flags": [
    {{"category":"cognitive|medication|safety","severity":"low|medium|high","description":"...", "reasoning":"...", "mitigation_strategy":""}}
  ],
  "cognitive_indicators": [
    {{"domain":"memory|attention|...","concern":"...", "evidence_source":"..."}}
  ],
  "complexity_score": 0,
  "complexity_summary": {{
    "score": 0, 
    "factors": ["..."]
  }},
  "information_gaps": ["..."]
}}

Return ONLY a valid JSON object. Do not include markdown formatting or conversational text.
""".strip(),
            expected_output="Strictly extracted and assessed patient risk profile as JSON",
            agent=self.agent,
            output_pydantic=PatientProfile
        )
    
    def execute(self, patient_data: PatientData, clinical_history: ClinicalHistory) -> PatientProfile:
        logger.info("Step 1: ProfilerAgent execution started (Pure LLM Mode)")
        task = self.create_task(patient_data, clinical_history)
        
        logger.info("Step 2: Initializing CrewAI environment")
        crew = Crew(agents=[self.agent], tasks=[task], verbose=False)
        
        logger.info("Step 3: Kicking off Crew... (Waiting for LLM analysis)")
        result = crew.kickoff()

        # Extract raw JSON safely
        raw_dict = self._try_extract_json(getattr(result, "raw", "") or "") or {}
        if not raw_dict and getattr(result, "pydantic", None):
            raw_dict = result.pydantic.model_dump(mode="json")

        logger.info("Step 4: Validating LLM output against Pydantic schema")
        patient_profile = PatientProfile.model_validate(raw_dict)

        num_flags = len(patient_profile.risk_flags) if getattr(patient_profile, 'risk_flags', None) else 0
        logger.info(f"Step 5: Success! Agent autonomously identified {num_flags} risk flags.")
        
        return patient_profile