from crewai import Agent, Task, LLM, Crew
from aether.config.settings import settings
from aether.schemas.clinical import ReferralInput, PatientData
from aether.utils.logger import logger
from datetime import datetime

class IntakeAgent:
    """Clinical Intake Specialist Agent - Extracts structured patient data from referrals."""
    
    def __init__(self):
        # 1. Lower temperature to 0.0 for deterministic extraction
        self.llm = LLM(
            model=f"ollama/{settings.ollama_model}",
            base_url=settings.ollama_base_url,
            temperature=0.0, 
        )
        
        self.agent = Agent(
            role="Clinical Intake Specialist",
            goal="Extract structured patient demographics and referral information from GP letters",
            backstory=(
                "You are an expert clinical administrator with 15 years of experience "
                "processing NHS referrals. You excel at extracting accurate patient information while "
                "maintaining data quality and compliance with NHS data standards. "
                "You NEVER invent data. If data is missing from the text, you omit it."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )
    
    def create_task(self, referral_input: ReferralInput) -> Task:
        """Create extraction task."""
        current_year = datetime.now().year
        
        return Task(
            description=f"""
Extract complete patient demographic and referral information from the following GP referral letter.

REFERRAL LETTER:
{referral_input.referral_text}

NHS Number (if provided): {referral_input.nhs_number or 'null'}

Extraction Rules:
1. Extract ONLY the information explicitly stated in the text.
2. If information is missing from the text, you MUST return `null` for that field. Do not use placeholder strings like "Not provided" or "N/A".
3. Ensure all dates are in ISO 8601 format (YYYY-MM-DD).
4. Calculate the patient's current age based on the current year ({current_year}).

IMPORTANT: Do not invent, guess, or assume any values.
            """.strip(),
            expected_output="A JSON object containing the structured patient data.",
            agent=self.agent,
            output_pydantic=PatientData # 2. Let CrewAI handle the Pydantic validation natively
        )
    
    # 3. _clean_json_response is no longer needed! Delete it.

    def execute(self, referral_input: ReferralInput) -> PatientData:
        """Execute the intake agent."""
        logger.info("IntakeAgent: Starting extraction")
        
        task = self.create_task(referral_input)
        
        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            verbose=False,
        )
        
        # 4. kickoff() now returns a CrewOutput object where the `pydantic` attribute
        # is already validated and cast to your PatientData model.
        result = crew.kickoff()
        patient_data = result.pydantic 
        
        logger.info(f"IntakeAgent: Extracted data for patient NHS#{patient_data.nhs_number}")
        
        return patient_data