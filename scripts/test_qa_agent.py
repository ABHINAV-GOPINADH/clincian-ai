import sys
import os

# Path Injection
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)              
src_dir = os.path.join(project_root, "src")              
sys.path.insert(0, src_dir)

from aether.agents.qa import QAAgent
from aether.schemas.clinical import (
    PatientData, ClinicalHistory, PatientProfile, AssessmentPlan, ClinicalBrief
)
from aether.utils.logger import logger

def test_qa_agent():
    logger.info("================================================================================")
    logger.info("🧪 Testing Quality Assurance Agent")
    logger.info("================================================================================")

    # --- MOCK DATA GENERATION ---
    logger.info("Generating mock data pipeline for audit...")
    
    mock_patient = PatientData(
        nhs_number="987-654-3210",
        name={"first": "Mary", "last": "Higgins"},
        date_of_birth="1955-08-12",
        age=68,
        gender="female",
        contact_info={"phone": None, "email": None, "address": "First-floor flat"},
        gp_details={"practice_name": "Riverside", "gp_name": "Dr. Chen", "contact_number": "01987"},
        referral_reason="Progressive memory difficulties over 6 months.",
        referral_date="2024-04-10"
    )

    mock_history = ClinicalHistory.model_validate({
        "conditions": [{"code": "131100006", "display": "Asthma", "status": "active", "severity": "mild"}],
        "medications": [{"name": "Salbutamol", "dosage": "100mcg", "frequency": "PRN", "indication": "Asthma"}],
        "allergies": ["Penicillin"],
        "past_assessments": [],
        "significant_events": [],
        "timeline_events": []
    })

    mock_profile = PatientProfile.model_validate({
        "complexity_summary": {"score": 6, "factors": ["Risk of falls"]},
        "risk_flags": [{"category": "safety", "severity": "high", "description": "Loss of independence due to stairs.", "reasoning": "Home safety compromised.", "mitigation_strategy": "Home assessment"}],
        "cognitive_indicators": [{"domain": "memory", "concern": "Difficulty with new learning.", "evidence_source": "Patient report."}],
        "information_gaps": []
    })

    mock_plan = AssessmentPlan.model_validate({
        "instruments": [{"name": "MoCA", "type": "ADAS-Cog", "priority": "essential", "rationale": "Sensitive for MCI.", "estimated_duration": 10}],
        "total_estimated_duration": 10,
        "priority_order": ["MoCA"],
        "contraindications": [],
        "special_considerations": [],
        "nice_compliance_notes": "Follows NICE NG97 guidelines."
    })

    mock_brief = ClinicalBrief.model_validate({
        "header": {"patient_name": "Mary Higgins", "nhs_number": "987-654-3210", "date_of_birth": "1955-08-12", "assessment_date": "2026-04-11"},
        "executive_summary": "Patient referred for progressive memory difficulties and high fall risk.",
        "presenting_concerns": ["Memory difficulties"],
        "relevant_history": {"medical": ["Asthma"], "psychiatric": [], "social": []},
        "risk_summary": [{"category": "safety", "severity": "high", "description": "Loss of independence due to stairs.", "reasoning": "Home safety compromised.", "mitigation_strategy": "Home assessment"}],
        "recommended_assessments": [{"name": "MoCA", "type": "ADAS-Cog", "priority": "essential", "rationale": "Sensitive for MCI.", "estimated_duration": 10}],
        "key_considerations": [],
        "nice_guidance_alignment": "Follows NICE NG97 guidelines."
    })

    try:
        logger.info("Initializing QAAgent instance...")
        agent = QAAgent()
        
        logger.info("Passing all data modules to QA agent.execute()...")
        qa_result = agent.execute(
            patient_data=mock_patient, 
            clinical_history=mock_history,
            patient_profile=mock_profile,
            assessment_plan=mock_plan,
            clinical_brief=mock_brief
        )

        logger.info("Execution finished. Printing results:\n")
        
        print("✅ QA Validation successful!\n")
        print("🔎 Extracted QA Report:")
        print("="*80)
        
        # Safely print the dict representation of the final report
        qa_dict = qa_result.model_dump()
        for key, value in qa_dict.items():
            key_formatted = key.replace('_', ' ').title()
            print(f"\n[{key_formatted}]:")
            
            if isinstance(value, list):
                if not value:
                    print("  None")
                else:
                    for item in value:
                        print(f"  - {item}")
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
    test_qa_agent()