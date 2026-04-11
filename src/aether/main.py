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
Re: John Smith, DOB 15/03/1948, NHS# 123-456-7890

Dear Colleague,

I am referring this 76-year-old gentleman for cognitive assessment.

His wife reports progressive memory difficulties over the past 18 months,
particularly with recent events and appointments. He has gotten lost
driving to familiar places twice in the past 3 months.

PMH: Type 2 diabetes, hypertension, hyperlipidemia
Medications: Metformin 500mg BD, Amlodipine 5mg OD, Atorvastatin 20mg ON

He lives with his wife who is his main carer. No formal care package in place.

I would be grateful for your expert assessment.

Kind regards,
Dr. Sarah Johnson
Greenfield Medical Centre
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