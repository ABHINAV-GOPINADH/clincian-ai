import sys
import os

# 1. Path Injection
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)              
src_dir = os.path.join(project_root, "src")              
sys.path.insert(0, src_dir)

from aether.agents.brief_writer import BriefWriterAgent
from aether.schemas.clinical import PatientData, ClinicalHistory, PatientProfile, AssessmentPlan
from aether.utils.logger import logger

def test_brief_writer_agent():
    logger.info("================================================================================")
    logger.info("🧪 Testing Clinical Brief Writer Agent")
    logger.info("================================================================================")

    # --- MOCK 1: Patient Data ---
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

    # --- MOCK 2: Clinical History ---
    mock_history = ClinicalHistory.model_validate({
        "conditions": [{"code": "131100006", "display": "Asthma", "status": "active", "severity": "mild"}],
        "medications": [{"name": "Salbutamol", "dosage": "100mcg", "frequency": "PRN", "indication": "Asthma"}],
        "allergies": ["Penicillin"],
        "past_assessments": [],
        "significant_events": [],
        "timeline_events": []
    })

    # --- MOCK 3: Patient Profile ---
    mock_profile = PatientProfile.model_validate({
        "complexity_summary": {"score": 6, "factors": ["Risk of falls", "Cognitive decline"]},
        "risk_flags": [
            {"category": "safety", "severity": "high", "description": "Loss of independence due to stairs.", "reasoning": "Home safety compromised.", "mitigation_strategy": "Home assessment"}
        ],
        "cognitive_indicators": [
            {"domain": "memory", "concern": "Difficulty with new learning.", "evidence_source": "Patient report."}
        ],
        "information_gaps": ["Medication dosing schedule"]
    })

    # --- MOCK 4: Assessment Plan ---
    mock_plan = AssessmentPlan.model_validate({
        "instruments": [
            {"name": "MoCA", "type": "ADAS-Cog", "priority": "essential", "rationale": "Sensitive for MCI.", "estimated_duration": 10},
            {"name": "NPI", "type": "GDS", "priority": "essential", "rationale": "Assess neuropsychiatric symptoms.", "estimated_duration": 20}
        ],
        "total_estimated_duration": 30,
        "priority_order": ["MoCA", "NPI"],
        "contraindications": [],
        "special_considerations": [],
        "nice_compliance_notes": "Follows NICE NG97 guidelines for initial cognitive assessment."
    })

    try:
        logger.info("Initializing BriefWriterAgent instance...")
        agent = BriefWriterAgent()
        
        logger.info("Passing all 4 data modules to agent.execute()...")
        brief_result = agent.execute(
            patient_data=mock_patient, 
            clinical_history=mock_history,
            patient_profile=mock_profile,
            assessment_plan=mock_plan
        )

        logger.info("Execution finished. Printing results:\n")
        
        print("✅ Brief Generation successful!\n")
        print("📑 Extracted Clinical Brief:")
        print("="*80)
        
        # Safely print the dict representation of the final brief
        brief_dict = brief_result.model_dump()
        for key, value in brief_dict.items():
            key_formatted = key.replace('_', ' ').title()
            print(f"\n[{key_formatted}]:")
            
            # Formatting lists nicely
            if isinstance(value, list):
                if not value:
                    print("  None")
                else:
                    for item in value:
                        print(f"  - {item}")
            # Formatting dicts nicely
            elif isinstance(value, dict):
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"  {value}")
                
        print("="*80)
                
    except Exception as e:
        logger.error(f"❌ Test failed during execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_brief_writer_agent()