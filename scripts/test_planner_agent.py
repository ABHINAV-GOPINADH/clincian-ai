import sys
import os

# 1. Path Injection
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)              
src_dir = os.path.join(project_root, "src")              
sys.path.insert(0, src_dir)

# Make sure the import matches whatever you named your file! 
# (Assuming aether/agents/planner.py)
from aether.agents.planner import AssessmentPlannerAgent
from aether.schemas.clinical import PatientData, PatientProfile
from aether.utils.logger import logger

def test_planner_agent():
    logger.info("================================================================================")
    logger.info("🧪 Testing Assessment Planner Agent")
    logger.info("================================================================================")

    logger.info("TEST STEP A: Preparing mock PatientData")
    mock_patient = PatientData(
        nhs_number="987-654-3210",
        name={"first": "Mary", "last": "Higgins"},
        date_of_birth="1955-08-12",
        age=68,
        gender="female",
        contact_info={"phone": None, "email": None, "address": "First-floor flat"},
        gp_details={"practice_name": "Riverside Clinic", "gp_name": "Dr. Chen", "contact_number": "01987"},
        referral_reason="Progressive memory difficulties over 6 months.",
        referral_date="2024-04-10"
    )

    logger.info("TEST STEP B: Preparing mock PatientProfile")
    # Generating mock profile based on your previous successful run
    mock_profile_dict = {
        "complexity_summary": {"score": 6, "factors": ["Risk of falls", "Cognitive decline"]},
        "risk_flags": [
            {
                "category": "safety", 
                "severity": "high", 
                "description": "Loss of independence due to memory difficulties and stairs.", 
                "reasoning": "Home safety compromised.", 
                "mitigation_strategy": "Home assessment"
            }
        ],
        "cognitive_indicators": [
            {
                "domain": "memory", 
                "concern": "Difficulty with new learning and remembering recent events.", 
                "evidence_source": "Patient and family report."
            },
            {
                "domain": "executive", 
                "concern": "Struggling to plan daily activities.", 
                "evidence_source": "Family report."
            }
        ],
        "information_gaps": ["Medication dosing schedule"]
    }
    mock_profile = PatientProfile.model_validate(mock_profile_dict)

    try:
        logger.info("TEST STEP C: Initializing AssessmentPlannerAgent instance")
        agent = AssessmentPlannerAgent()
        
        logger.info("TEST STEP D: Passing data to agent.execute()")
        plan_result = agent.execute(
            patient_data=mock_patient, 
            patient_profile=mock_profile
        )

        logger.info("TEST STEP E: Execution finished. Printing results:")
        
        print("\n✅ Assessment Planning successful!\n")
        print("📊 Extracted Assessment Plan:")
        
        instruments = getattr(plan_result, 'instruments', [])
        print(f"\n  Instruments Planned: {len(instruments) if instruments else 0}")
        if instruments:
            for inst in instruments:
                 print(f"    - {inst.model_dump() if hasattr(inst, 'model_dump') else inst}")
                 
        print(f"\n  Overall Rationale: {getattr(plan_result, 'overall_rationale', 'None provided')}")
        print(f"\n  Estimated Total Duration: {getattr(plan_result, 'estimated_duration_minutes', 'Unknown')} minutes")
                
    except Exception as e:
        logger.error(f"❌ Test failed during execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_planner_agent()