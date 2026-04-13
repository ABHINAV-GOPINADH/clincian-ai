import sys
import os

# 1. Path Injection to ensure we can import 'aether'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)              
src_dir = os.path.join(project_root, "src")              
sys.path.insert(0, src_dir)

from aether.agents.profiler import ProfilerAgent
from aether.schemas.clinical import PatientData, ClinicalHistory
from aether.utils.logger import logger

def test_profiler_agent():
    logger.info("================================================================================")
    logger.info("🧪 Testing Clinical Profiler Agent")
    logger.info("================================================================================")

    logger.info("TEST STEP A: Preparing mock PatientData")
    mock_patient = PatientData(
        nhs_number="987-654-3210",
        name={"first": "Mary", "last": "Higgins"},
        date_of_birth="1955-08-12",
        age=68,
        gender="female",
        contact_info={"phone": None, "email": None, "address": "First-floor flat without lift access"},
        gp_details={
            "practice_name": "Riverside Health Clinic", 
            "gp_name": "Dr. David Chen", 
            "contact_number": "01987 654321"
        },
        referral_reason="Progressive memory difficulties over 6 months, recently got lost walking to the local shop. Lives alone, struggling with stairs.",
        referral_date="2024-04-10"
    )

    logger.info("TEST STEP B: Preparing mock ClinicalHistory")
    # Using a dict and Pydantic validation to easily generate the mock history
    mock_history_dict = {
        "conditions": [
            {"code": "131100006", "display": "Asthma", "status": "active", "severity": "mild"},
            {"code": "38341003", "display": "Hypertension", "status": "active", "severity": "moderate"}
        ],
        "medications": [
            {"name": "Salbutamol", "dosage": "100mcg", "frequency": "PRN"},
            {"name": "Amlodipine", "dosage": "5mg", "frequency": "Daily"}
        ],
        "allergies": ["Penicillin"],
        "past_assessments": [],
        "significant_events": [],
        "timeline_events": []  # <-- Add this missing field here!
    }
    mock_history = ClinicalHistory.model_validate(mock_history_dict)

    try:
        logger.info("TEST STEP C: Initializing ProfilerAgent instance")
        agent = ProfilerAgent()
        
        logger.info("TEST STEP D: Passing data to agent.execute()")
        profile_result = agent.execute(
            patient_data=mock_patient, 
            clinical_history=mock_history
        )

        logger.info("TEST STEP E: Execution finished. Printing results:")
        
        print("\n✅ Profiling successful!\n")
        print("📊 Extracted Patient Profile:")
        
        # We use model_dump() everywhere so it prints safely regardless of your exact schema keys
        
        print(f"\n  Complexity Score: {getattr(profile_result, 'complexity_score', 'N/A')}")
        
        risk_flags = getattr(profile_result, 'risk_flags', [])
        print(f"\n  Risk Flags Found: {len(risk_flags) if risk_flags else 0}")
        if risk_flags:
            for flag in risk_flags:
                # Use model_dump if it's an object, otherwise print as a string
                print(f"    - {flag.model_dump() if hasattr(flag, 'model_dump') else flag}")
                
        cog_indicators = getattr(profile_result, 'cognitive_indicators', getattr(profile_result, 'cognitive_assessments', []))
        print(f"\n  Cognitive Indicators Found: {len(cog_indicators) if cog_indicators else 0}")
        if cog_indicators:
            for ind in cog_indicators:
                print(f"    - {ind.model_dump() if hasattr(ind, 'model_dump') else ind}")
                
        info_gaps = getattr(profile_result, 'information_gaps', [])
        print(f"\n  Information Gaps Found: {len(info_gaps) if info_gaps else 0}")
        if info_gaps:
            for gap in info_gaps:
                print(f"    - {gap.model_dump() if hasattr(gap, 'model_dump') else gap}")
                
    except Exception as e:
        logger.error(f"❌ Test failed during execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_profiler_agent()