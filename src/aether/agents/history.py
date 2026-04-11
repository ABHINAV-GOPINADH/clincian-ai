from crewai import Agent, Task, Crew
from aether.schemas.clinical import PatientData, ClinicalHistory
from aether.utils.logger import logger
from aether.config.llm_config import agent_llm
import re


class ClinicalHistoryAgent:
    """Clinical Historian Agent - Structures comprehensive patient clinical history."""
    
    def __init__(self):
        self.llm = agent_llm  # CHANGED: Using Gemini
        
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
    
    def create_task(self, patient_data: PatientData, clinical_notes: str = None) -> Task:
        """Create history structuring task."""
        return Task(
            description=f"""
            Analyze and structure the clinical history for this patient.

            PATIENT CONTEXT:
            Name: {patient_data.name.first} {patient_data.name.last}
            NHS#: {patient_data.nhs_number}
            Age: {patient_data.age} | Gender: {patient_data.gender}

            REFERRAL REASON:
            {patient_data.referral_reason}

            {f"ADDITIONAL CLINICAL NOTES:\n{clinical_notes}" if clinical_notes else ""}

            Extract and structure ONLY what is explicitly stated:
            1. Medical conditions (infer reasonable SNOMED-CT text codes if applicable)
            2. Current medications with dosage 
            3. Known allergies
            4. Past psychiatric/cognitive assessments
            5. Timeline of significant clinical events

            CRITICAL RULE: You MUST return a valid JSON object using EXACTLY these key names. Do not deviate:
            - "conditions"
            - "medications"
            - "allergies"
            - "cognitive_assessments"
            - "significant_events"

            If a specific piece of information is missing, return an empty list `[]`. Do not guess.
            """.strip(),
                        expected_output="A structured JSON object matching the requested schema exactly.",
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