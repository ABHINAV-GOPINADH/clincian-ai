"""
AETHER Multi-Agent Clinical Assessment System
Entry point for the agent orchestration module
"""
import sys
import os

# 1. Get the absolute path to the 'src' directory
current_dir = os.path.dirname(os.path.abspath(__file__)) # Gets the 'aether' folder
src_dir = os.path.dirname(current_dir)                   # Gets the 'src' folder

# 2. Insert 'src' at the very front of Python's path list
sys.path.insert(0, src_dir)

# 3. NOW you can do your normal imports
from aether.orchestrator.crew import AetherCrew
# ... the rest of your imports and code ...

from aether.orchestrator.crew import AetherCrew
from aether.schemas.clinical import ReferralInput
from aether.utils.logger import logger


def main():
    """Example execution of AETHER agent system."""
    
    example_referral = ReferralInput(
        referral_text="""
REFERRAL TO: Leeds Memory Assessment Service (Old Age Psychiatry)
ODS Code: RR8MA

REFERRING CLINICIAN: Dr. Priya Nair, General Practitioner (GMC: 7654321)
Meadowbank Surgery
12 Meadowbank Road, Leeds, LS8 3HL
ODS: F83006

DATE OF REFERRAL: 02 April 2026
URGENCY: Routine

PATIENT DETAILS:
Name: Mrs. Margaret Wilson
DOB: 14 March 1948 (Age: 78)
NHS Number: 943 476 5919
Address: 14 Beechwood Close, Leeds, West Yorkshire, LS8 4PQ
Telephone: 0113 265 XXXX
Next of Kin: Claire Wilson (Daughter) - 07700 9XXXXX (Consent given to contact)

Dear Memory Assessment Team,

RE: Query Early Cognitive Decline / Mild Cognitive Impairment

Thank you for seeing this 78-year-old lady, who I am referring for a routine cognitive assessment due to an ongoing decline in her memory over the past 12 months.

History of Presenting Complaint:
Mrs. Wilson and her daughter, Claire, have both noted increasing forgetfulness and word-finding difficulties. In January 2026, her daughter raised concerns with the surgery that Mrs. Wilson had been repeating questions, missing appointments, and had left the kitchen hob on twice.

Examination & Investigations:
I reviewed her in the surgery today (02/04/2026). There are no focal neurological signs. I administered a Mini-Mental State Examination (MMSE), on which she scored 24/30 (borderline/below age-adjusted norm), specifically losing points on orientation (2), recall (3), and attention (1).

Past Medical History:

Essential hypertension (since 2012)

Type 2 diabetes mellitus (since 2015)

Primary hypothyroidism (since 2018)

Current Medications:

Amlodipine 5mg tablets - once daily

Metformin 500mg tablets - twice daily with meals

Levothyroxine sodium 50mcg tablets - once daily, morning

Aspirin 75mg dispersible tablets - once daily

Allergies:

Penicillin (Reaction: Urticaria - Moderate)

Social History:
Mrs. Wilson is a retired school teacher who has lived alone since she was widowed in 2021. Her daughter visits her twice weekly to provide informal support. She is independent with her mobility (uses a stick outdoors) and is currently still driving. She is an ex-smoker (quit in 2005) and drinks approximately 4 units of alcohol per week.

I would be very grateful for your specialist assessment, diagnostic clarification, and guidance on further management.

Yours sincerely,

Dr. Priya Nair
General Practitioner
Meadowbank Surgery
        """.strip(),
        nhs_number="123-456-7890"
    )
    
    crew = AetherCrew()
    
    try:
        logger.info("Starting AETHER agent system demonstration")
        
        result = crew.execute_with_streaming(
            example_referral,
            lambda step, data: logger.info(f"Progress: {step}", extra=data)
        )
        
        logger.info(
            f"Final result: Status={result.qa_result.overall_status}, "
            f"ProcessingTime={result.metadata.processing_time_ms}ms"
        )
        
        print("\n" + "="*80)
        print("CLINICAL BRIEF")
        print("="*80)
        print(result.clinical_brief.model_dump_json(indent=2))
        
        print("\n" + "="*80)
        print("QA RESULT")
        print("="*80)
        print(result.qa_result.model_dump_json(indent=2))
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        raise


if __name__ == "__main__":
    main()