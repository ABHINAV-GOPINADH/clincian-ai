"""Test Intake Agent independently."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from aether.agents.intake import IntakeAgent
from aether.schemas.clinical import ReferralInput
from aether.utils.logger import logger

# Sample GP referral letter
SAMPLE_REFERRAL = """
Re: Mary Higgins, DOB 12/08/1955, NHS# 987-654-3210

Dear Colleague,

I am referring this 68-year-old lady for an orthopedic assessment regarding persistent right knee joint pain.

She presents with a 6-month history of progressively worsening pain and swelling in her right knee, accompanied by significant morning stiffness lasting up to an hour. The pain is exacerbated by weight-bearing activities, and she reports that her walking distance has now reduced to less than 200 yards. A recent X-ray showed moderate joint space narrowing and osteophyte formation consistent with osteoarthritis. Conservative management, including a 6-week course of physiotherapy and regular analgesia, has yielded minimal relief.

PMH: Asthma, Hypothyroidism, Mild obesity (BMI 28)
Medications: Salbutamol 100mcg inhaler PRN, Levothyroxine 75mcg OD, Ibuprofen 400mg TDS PRN, Omeprazole 20mg OD

She is a retired teacher who lives alone in a first-floor flat without lift access. The stairs are becoming increasingly difficult for her to manage, which is beginning to impact her independence.

I would be grateful for your expert evaluation to consider suitability for a corticosteroid injection or joint replacement surgery.

Kind regards,
Dr. David Chen
Riverside Health Clinic
Tel: 01987 654321
"""

def test_intake_agent():
    logger.info("="*80)
    logger.info("Testing Intake Agent")
    logger.info("="*80)
    
    try:
        # Create agent
        agent = IntakeAgent()
        
        # Create input
        referral_input = ReferralInput(
            referral_text=SAMPLE_REFERRAL,
            nhs_number="123-456-7890"
        )
        
        logger.info("\n📥 Processing referral...")
        
        # Execute
        patient_data = agent.execute(referral_input)
        
        logger.info("\n✅ Extraction successful!")
        logger.info("\n📊 Extracted Patient Data:")
        logger.info(f"   Name: {patient_data.name.first} {patient_data.name.last}")
        logger.info(f"   NHS#: {patient_data.nhs_number}")
        logger.info(f"   Age: {patient_data.age}")
        logger.info(f"   DOB: {patient_data.date_of_birth}")
        logger.info(f"   Gender: {patient_data.gender}")
        logger.info(f"   GP: {patient_data.gp_details.practice_name}")
        logger.info(f"   Referral reason: {patient_data.referral_reason[:100]}...")
        
        return patient_data
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_intake_agent()