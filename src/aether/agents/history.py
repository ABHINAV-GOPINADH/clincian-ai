from crewai import Agent, Task
from langchain_anthropic import ChatAnthropic
from aether.config.settings import settings
from aether.schemas.clinical import ReferralInput, PatientData
from aether.utils.logger import logger


class IntakeAgent:
    """Clinical Intake Specialist Agent - Extracts structured patient data from referrals."""
    
    def __init__(self):
        self.llm = ChatAnthropic(
            model=settings.claude_model,
            anthropic_api_key=settings.anthropic_api_key,
            temperature=0.2,
            max_tokens=4096,
        )
        
        self.agent = Agent(
            role="Clinical Intake Specialist",
            goal="Extract structured patient demographics and referral information from GP letters",
            backstory=(
                "You are an expert clinical administrator with 15 years of experience "
                "processing NHS referrals. You excel at extracting accurate patient information while "
                "maintaining data quality and compliance with NHS data standards."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )
    
    def create_task(self, referral_input: ReferralInput) -> Task:
        """Create extraction task."""
        return Task(
            description=f"""
Extract complete patient demographic and referral information from the following GP referral letter.

REFERRAL LETTER:
{referral_input.referral_text}

NHS Number (if provided): {referral_input.nhs_number or 'Not provided'}

Extract:
1. Patient full name (first and last)
2. NHS number
3. Date of birth and calculate age
4. Gender
5. Contact information (phone, email, address if available)
6. GP practice details
7. Referral reason and date

Ensure all dates are in ISO 8601 format (YYYY-MM-DD).
If information is missing, mark it as "Not provided" but do NOT fabricate data.

Return the data as valid JSON matching the PatientData schema.
            """.strip(),
            expected_output="Structured patient data as JSON",
            agent=self.agent,
        )
    
    def execute(self, referral_input: ReferralInput) -> PatientData:
        """Execute the intake agent."""
        logger.info("IntakeAgent: Starting extraction")
        
        task = self.create_task(referral_input)
        result = task.execute()
        
        # Parse and validate with Pydantic
        patient_data = PatientData.model_validate_json(result)
        
        logger.info(f"IntakeAgent: Extracted data for patient NHS#{patient_data.nhs_number}")
        return patient_data