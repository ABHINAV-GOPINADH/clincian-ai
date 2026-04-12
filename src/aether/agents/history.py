from crewai import Agent, Task, Crew
from aether.schemas.clinical import PatientData, ClinicalHistory
from aether.utils.logger import logger
from aether.config.llm_config import agent_llm
import re


class ClinicalHistoryAgent:
    """Clinical Historian Agent - Structures comprehensive patient clinical history."""
    
    def __init__(self):
        self.llm = agent_llm 
        
        self.agent = Agent(
            role="Clinical Historian",
            goal="Structure comprehensive patient clinical history with SNOMED-CT coding",
            backstory=(
                "You are a senior clinical informaticist specialized in dementia care pathways. "
                "You have deep knowledge of SNOMED-CT coding, medication classification, and timeline "
                "reconstruction from narrative clinical notes."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )
    
    # CHANGED: clinical_notes is now strictly required
    def create_task(self, patient_data: PatientData, clinical_notes: str) -> Task:
        return Task(
            description=f"""
            Analyze and structure the clinical history for this patient.

            PATIENT CONTEXT:
            Name: {patient_data.name.first} {patient_data.name.last}
            NHS#: {patient_data.nhs_number}
            Age: {patient_data.age} | Gender: {patient_data.gender}

            REFERRAL REASON:
            {patient_data.referral_reason}

            RAW CLINICAL NOTES:
            {clinical_notes}

            CRITICAL EXTRACTION RULES:
                1. ONLY extract explicit medical diagnoses mentioned in the text.
                2. DO NOT hallucinate or invent SNOMED, ICD-10, or any other medical codes. If a specific code is not explicitly written next to the condition in the text, you MUST set the `code` field to `null`.
                3. NEVER use administrative codes (e.g., ODS codes like F83006, GMC numbers, NHS numbers, or postal codes) as medical diagnostic codes. 
                4. Do not guess the medical history. If it is not in the text, leave the list empty.

                Return ONLY a valid JSON object matching the ClinicalHistory schema.
            """.strip(),
            expected_output="A structured JSON object representing the patient's history.",
            agent=self.agent,
            output_pydantic=ClinicalHistory 
        )
    
    def _clean_json_response(self, result: str) -> str:
        """Clean Gemini response to extract pure JSON."""
        result = result.strip()
        if result.startswith("```"):
            result = re.sub(r'^```(?:json)?\n', '', result)
            result = re.sub(r'\n```$', '', result)
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            result = json_match.group(0)
        return result.strip()
    
    def execute(self, patient_data: PatientData, clinical_notes: str = None) -> ClinicalHistory:
        """Execute the clinical history agent."""
        logger.info("Step 1: ClinicalHistoryAgent execution started")

        # ADD THIS SAFEGUARD: Stop the pipeline if the text is missing
        if not clinical_notes or not clinical_notes.strip():
            logger.error("CRITICAL: No raw clinical notes provided. Aborting to prevent hallucination.")
            return ClinicalHistory(conditions=[], medications=[], allergies=[], past_assessments=[], timeline_events=[])
        
        logger.info("Step 2: Creating Task with prompt instructions")
        task = self.create_task(patient_data, clinical_notes)
        
        logger.info("Step 3: Initializing CrewAI environment")
        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            verbose=False,
        )
        
        logger.info("Step 4: Kicking off Crew... (Waiting for LLM response)")
        result = crew.kickoff()
        
        logger.info("Step 5: LLM response received, extracting Pydantic model")
        
        # SAFETY NET: Check if CrewAI's parser failed
        if result.pydantic is None:
            logger.warning("CrewAI native parsing returned None. Falling back to manual validation...")
            import re
            
            # Extract raw string, strip markdown if present, and force Pydantic to parse it
            raw_text = result.raw
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                raw_text = json_match.group(0)
                
            # If the keys are STILL wrong, this will throw a very clear ValidationError instead of a NoneType crash
            clinical_history = ClinicalHistory.model_validate_json(raw_text)
        else:
            clinical_history = result.pydantic 
        
        logger.info(f"Step 6: Success! Structured {len(clinical_history.conditions)} conditions and {len(clinical_history.medications)} medications.")
        
        return clinical_history