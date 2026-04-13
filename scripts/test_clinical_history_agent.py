import sys
import os

# 1. Get the absolute path to your 'src' directory
current_dir = os.path.dirname(os.path.abspath(__file__)) # Gets the 'scripts' folder
project_root = os.path.dirname(current_dir)              # Gets the 'clincian-ai' folder
src_dir = os.path.join(project_root, "src")              # Gets the 'src' folder

# 2. Insert 'src' at the very front of Python's path list
sys.path.insert(0, src_dir)

# 3. NOW you can do all your imports normally (Do NOT use 'src.aether')
from aether.agents.history import ClinicalHistoryAgent
from aether.schemas.clinical import PatientData
from aether.utils.logger import logger

def test_clinical_history_agent():
    logger.info("================================================================================")
    logger.info("🧪 Testing Clinical History Agent")
    logger.info("================================================================================")

    logger.info("TEST STEP A: Preparing mock PatientData and Notes")
    mock_patient = PatientData(
        nhs_number="987-654-3210",
        name={"first": "Mary", "last": "Higgins"},
        date_of_birth="1955-08-12",
        age=68,
        gender="female",
        contact_info={"phone": None, "email": None, "address": "First-floor flat"},
        gp_details={
            "practice_name": "Riverside Health Clinic", 
            "gp_name": "Dr. David Chen", 
            "contact_number": "01987 654321"
        },
        referral_reason="Persistent right knee joint pain for orthopedic assessment.",
        referral_date="2024-04-10"
    )

    mock_notes = """
    Patient seen by community physio on 15/02/2024. Noted significant crepitus in right knee. 
    Trial of naproxen was stopped in 2022 due to GI upset. 
    Currently managing with Ibuprofen but complains of mild heartburn. 
    Diagnosed with Asthma in childhood, well controlled. 
    Thyroid function tests normal as of Jan 2024. 
    No known drug allergies (NKDA) except the naproxen intolerance.
    """

    try:
        logger.info("TEST STEP B: Initializing ClinicalHistoryAgent instance")
        agent = ClinicalHistoryAgent()
        
        logger.info("TEST STEP C: Passing data to agent.execute()")
        history_result = agent.execute(
            patient_data=mock_patient, 
            clinical_notes=mock_notes.strip()
        )

        logger.info("TEST STEP D: Execution finished. Printing results:")
        
        print("\n✅ Extraction successful!\n")
        print("📊 Extracted Clinical History:")
        
        print(f"\n  Conditions Found: {len(history_result.conditions)}")
        for cond in history_result.conditions:
            print(f"    - {cond.model_dump()}")
            
        print(f"\n  Medications Found: {len(history_result.medications)}")
        for med in history_result.medications:
            print(f"    - {med.model_dump()}")
            
        # FIX: Just print the list directly since it's a list of strings
        print(f"\n  Allergies: {history_result.allergies}")
        
        # Use getattr to safely check for events, handling potential schema variations
        events = getattr(history_result, 'significant_events', getattr(history_result, 'timeline_events', []))
        print(f"\n  Significant Events/Timeline: {len(events) if events else 0}")
        if events:
            for event in events:
                 # Safety check: Use model_dump if it's an object, otherwise print as a string
                 if hasattr(event, 'model_dump'):
                     print(f"    - {event.model_dump()}")
                 else:
                     print(f"    - {event}")
                 
        # ------------------------------------------
        
    except Exception as e:
        logger.error(f"❌ Test failed during execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_clinical_history_agent()