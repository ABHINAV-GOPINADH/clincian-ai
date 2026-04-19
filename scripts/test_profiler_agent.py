import sys
import os
from typing import Dict, List, Optional, Any

# 1. Path Injection to ensure we can import 'aether'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)              
src_dir = os.path.join(project_root, "src")              
sys.path.insert(0, src_dir)

from aether.agents.profiler import ProfilerAgent
from aether.schemas.clinical import PatientData, ClinicalHistory
from aether.utils.logger import logger

class TestCase:
    """Container for a test case with expected verification criteria"""
    def __init__(self, name: str, patient_data: PatientData, clinical_history: ClinicalHistory,
                 expected_min_risk_flags: int = 0,
                 expected_min_cognitive_indicators: int = 0,
                 expected_complexity_range: tuple = (0, 100),
                 expected_risk_types: Optional[List[str]] = None,
                 description: str = ""):
        self.name = name
        self.patient_data = patient_data
        self.clinical_history = clinical_history
        self.expected_min_risk_flags = expected_min_risk_flags
        self.expected_min_cognitive_indicators = expected_min_cognitive_indicators
        self.expected_complexity_range = expected_complexity_range
        self.expected_risk_types = expected_risk_types or []
        self.description = description

def create_test_cases() -> List[TestCase]:
    """Create 5 varied test cases with different risk profiles"""
    
    test_cases = []
    
    # Test Case 1: Elderly patient with multiple comorbidities and memory issues
    test_cases.append(TestCase(
        name="Elderly with Multiple Comorbidities",
        patient_data=PatientData(
            nhs_number="987-654-3210",
            name={"first": "Mary", "last": "Higgins"},
            date_of_birth="1935-08-12",
            age=88,
            gender="female",
            contact_info={"phone": "07700 900123", "email": None, "address": "First-floor flat without lift access"},
            gp_details={
                "practice_name": "Riverside Health Clinic", 
                "gp_name": "Dr. David Chen", 
                "contact_number": "01987 654321"
            },
            referral_reason="Progressive memory difficulties over 6 months, recently got lost walking to the local shop. Lives alone, struggling with stairs. Two falls in the last month.",
            referral_date="2024-04-10"
        ),
        clinical_history=ClinicalHistory(
            conditions=[
                {"code": "131100006", "display": "Asthma", "status": "active", "severity": "moderate"},
                {"code": "38341003", "display": "Hypertension", "status": "active", "severity": "moderate"},
                {"code": "44054006", "display": "Type 2 Diabetes", "status": "active", "severity": "moderate"},
                {"code": "399211009", "display": "Osteoarthritis", "status": "active", "severity": "severe"},
                {"code": "371125006", "display": "Chronic Kidney Disease Stage 3", "status": "active", "severity": "moderate"}
            ],
            medications=[
                {"name": "Salbutamol", "dosage": "100mcg", "frequency": "PRN"},
                {"name": "Amlodipine", "dosage": "5mg", "frequency": "Daily"},
                {"name": "Metformin", "dosage": "500mg", "frequency": "Twice daily"},
                {"name": "Paracetamol", "dosage": "1g", "frequency": "Four times daily"}
            ],
            allergies=["Penicillin", "NSAIDs"],
            past_assessments=[
                {"date": "2023-11-15", "type": "MMSE", "score": 24, "interpretation": "Mild cognitive impairment", "outcome": "Mild cognitive impairment"}
            ],
            significant_events=[
                {"date": "2024-03-10", "description": "Fall at home, no fracture", "severity": "moderate"},
                {"date": "2024-03-25", "description": "Fall in garden, minor laceration", "severity": "moderate"}
            ],
            timeline_events=[]
        ),
        expected_min_risk_flags=4,  # Expect multiple risk flags (falls, polypharmacy, living alone, cognitive decline)
        expected_min_cognitive_indicators=2,
        expected_complexity_range=(70, 95),  # High complexity due to multiple factors
        expected_risk_types=["falls", "cognition", "social", "medication"],
        description="Elderly patient with multiple comorbidities, recent falls, and cognitive concerns"
    ))
    
    # Test Case 2: Younger patient with mental health and substance use issues
    test_cases.append(TestCase(
        name="Mental Health with Substance Use",
        patient_data=PatientData(
            nhs_number="123-456-7890",
            name={"first": "James", "last": "Wilson"},
            date_of_birth="1988-03-25",
            age=36,
            gender="male",
            contact_info={"phone": "07700 123456", "email": None, "address": "Temporary accommodation, homeless shelter"},
            gp_details={
                "practice_name": "City Centre Practice", 
                "gp_name": "Dr. Sarah Johnson", 
                "contact_number": "01234 567890"
            },
            referral_reason="Increasing paranoia, social withdrawal, poor self-care. History of alcohol dependence. Recently stopped taking antipsychotic medication.",
            referral_date="2024-04-12"
        ),
        clinical_history=ClinicalHistory(
            conditions=[
                {"code": "191736004", "display": "Schizophrenia", "status": "active", "severity": "severe"},
                {"code": "66590003", "display": "Alcohol dependence syndrome", "status": "active", "severity": "severe"},
                {"code": "35489007", "display": "Depressive disorder", "status": "active", "severity": "moderate"}
            ],
            medications=[
                {"name": "Olanzapine", "dosage": "10mg", "frequency": "Daily", "status": "discontinued"},
                {"name": "Acamprosate", "dosage": "666mg", "frequency": "Three times daily"}
            ],
            allergies=[],
            past_assessments=[
                {"date": "2023-09-10", "type": "CAGE", "score": 3, "interpretation": "High risk of alcohol dependence", "outcome": "High risk of alcohol dependence"}

            ],
            significant_events=[
                {"date": "2024-02-15", "description": "A&E attendance for alcohol intoxication", "severity": "moderate"},
                {"date": "2024-03-20", "description": "Lost temporary accommodation due to behavior", "severity": "severe"}
            ],
            timeline_events=[]
        ),
        expected_min_risk_flags=3,  # Social risks, medication non-compliance, substance use
        expected_min_cognitive_indicators=1,  # Paranoia as cognitive/psychiatric indicator
        expected_complexity_range=(60, 85),
        expected_risk_types=["social", "substance", "medication", "mental_health"],
        description="Younger patient with serious mental illness, substance use, and social instability"
    ))
    
    # Test Case 3: Patient with complex medication regimen and frailty
    test_cases.append(TestCase(
        name="Polypharmacy and Frailty",
        patient_data=PatientData(
            nhs_number="555-444-3333",
            name={"first": "Robert", "last": "Smith"},
            date_of_birth="1942-11-30",
            age=81,
            gender="male",
            contact_info={"phone": "07700 555444", "email": None, "address": "Bungalow with step-free access"},
            gp_details={
                "practice_name": "Parkview Medical Centre", 
                "gp_name": "Dr. Emma Wilson", 
                "contact_number": "01865 334455"
            },
            referral_reason="Multiple hospital admissions in past year for falls, chest infections, and medication complications. Carer support at home 4 hours daily.",
            referral_date="2024-04-08"
        ),
        clinical_history=ClinicalHistory(
            conditions=[
                {"code": "38341003", "display": "Hypertension", "status": "active", "severity": "moderate"},
                {"code": "72892002", "display": "Heart failure", "status": "active", "severity": "moderate"},
                {"code": "13645005", "display": "COPD", "status": "active", "severity": "severe"},
                {"code": "302870006", "display": "Atrial fibrillation", "status": "active", "severity": "moderate"},
                {"code": "399211009", "display": "Osteoporosis", "status": "active", "severity": "severe"}
            ],
            medications=[
                {"name": "Furosemide", "dosage": "40mg", "frequency": "Daily"},
                {"name": "Bisoprolol", "dosage": "5mg", "frequency": "Daily"},
                {"name": "Apixaban", "dosage": "5mg", "frequency": "Twice daily"},
                {"name": "Tiotropium", "dosage": "18mcg", "frequency": "Daily"},
                {"name": "Salbutamol", "dosage": "100mcg", "frequency": "PRN"},
                {"name": "Alendronic acid", "dosage": "70mg", "frequency": "Weekly"},
                {"name": "Paracetamol", "dosage": "1g", "frequency": "Four times daily"},
                {"name": "Codeine", "dosage": "30mg", "frequency": "Four times daily"}
            ],
            allergies=["Sulfonamides"],
            past_assessments=[
                {"date": "2024-01-15", "type": "Frailty Score", "score": 6, "interpretation": "Moderately frail", "outcome": "Moderately frail"}
            ],
            significant_events=[
                {"date": "2023-12-10", "description": "Hospital admission for fall with hip fracture", "severity": "severe"},
                {"date": "2024-02-20", "description": "Hospital admission for COPD exacerbation", "severity": "severe"},
                {"date": "2024-03-05", "description": "Medication review - reduced Apixaban dose due to bleeding risk", "severity": "moderate"}
            ],
            timeline_events=[]
        ),
        expected_min_risk_flags=5,  # Polypharmacy, frailty, falls, medication interactions, multiple comorbidities
        expected_min_cognitive_indicators=0,  # Not primarily cognitive
        expected_complexity_range=(75, 95),
        expected_risk_types=["medication", "falls", "frailty", "hospital_admission"],
        description="Elderly patient with polypharmacy, frailty, and frequent hospital admissions"
    ))
    
    # Test Case 4: Patient with early cognitive concerns and mild risks
    test_cases.append(TestCase(
        name="Early Cognitive Concerns",
        patient_data=PatientData(
            nhs_number="222-333-4444",
            name={"first": "Margaret", "last": "Brown"},
            date_of_birth="1958-07-14",
            age=65,
            gender="female",
            contact_info={"phone": "07700 222333", "email": "margaret.brown@email.com", "address": "Two-storey house, stairs to bedroom"},
            gp_details={
                "practice_name": "Willow Tree Surgery", 
                "gp_name": "Dr. Thomas Reed", 
                "contact_number": "01452 789012"
            },
            referral_reason="Subjective memory complaints over 12 months. Family concerned about repetitive questioning. Still independent with ADLs.",
            referral_date="2024-04-05"
        ),
        clinical_history=ClinicalHistory(
            conditions=[
                {"code": "38341003", "display": "Hypertension", "status": "active", "severity": "mild"},
                {"code": "46635009", "display": "Hypothyroidism", "status": "active", "severity": "mild"}
            ],
            medications=[
                {"name": "Lisinopril", "dosage": "10mg", "frequency": "Daily"},
                {"name": "Levothyroxine", "dosage": "50mcg", "frequency": "Daily"}
            ],
            allergies=[],
            past_assessments=[
                {"date": "2024-01-20", "type": "MMSE", "score": 28, "interpretation": "Normal range", "outcome": "Normal range"}
            ],
            significant_events=[],
            timeline_events=[]
        ),
        expected_min_risk_flags=1,  # Mild cognitive concerns
        expected_min_cognitive_indicators=1,  # Memory complaints
        expected_complexity_range=(30, 60),  # Lower complexity
        expected_risk_types=["cognition"],
        description="Patient with early cognitive concerns, minimal comorbidities, good social support"
    ))
    
    # Test Case 5: Patient with sensory impairments and social isolation
    test_cases.append(TestCase(
        name="Sensory Impairments with Social Isolation",
        patient_data=PatientData(
            nhs_number="777-888-9999",
            name={"first": "Arthur", "last": "Jones"},
            date_of_birth="1929-12-03",
            age=94,
            gender="male",
            contact_info={"phone": None, "email": None, "address": "Ground floor flat, rarely leaves home"},
            gp_details={
                "practice_name": "Hillside Medical Practice", 
                "gp_name": "Dr. Lisa Carter", 
                "contact_number": "01632 456789"
            },
            referral_reason="Severe hearing and visual impairment. Lives alone, daughter visits weekly. Recent weight loss and reduced mobility.",
            referral_date="2024-04-15"
        ),
        clinical_history=ClinicalHistory(
            conditions=[
                {"code": "15188001", "display": "Age-related macular degeneration", "status": "active", "severity": "severe"},
                {"code": "3463005", "display": "Sensorineural hearing loss", "status": "active", "severity": "severe"},
                {"code": "266569009", "display": "Osteoarthritis of knee", "status": "active", "severity": "severe"}
            ],
            medications=[
                {"name": "Paracetamol", "dosage": "1g", "frequency": "Four times daily"},
                {"name": "Ibuprofen", "dosage": "400mg", "frequency": "Three times daily"}
            ],
            allergies=[],
            past_assessments=[
                {"date": "2023-10-10", "type": "Visual Acuity", "score": "6/60", "interpretation": "Severely impaired", "outcome": "Severely impaired"},
                {"date": "2023-10-10", "type": "Hearing Test", "result": "Severe bilateral loss", "interpretation": "Requires hearing aids", "outcome": "Severe bilateral loss"}
            ],
            significant_events=[
                {"date": "2024-03-01", "description": "Stopped using hearing aids (broken)", "severity": "moderate"},
                {"date": "2024-03-15", "description": "Meals on Wheels service started", "severity": "mild"}
            ],
            timeline_events=[]
        ),
        expected_min_risk_flags=3,  # Sensory impairment, social isolation, reduced mobility
        expected_min_cognitive_indicators=0,  # Not primarily cognitive
        expected_complexity_range=(65, 85),
        expected_risk_types=["sensory", "social", "mobility"],
        description="Very elderly patient with severe sensory impairments, social isolation, and reduced mobility"
    ))
    
    return test_cases

def verify_test_result(test_case: TestCase, profile_result: Any) -> Dict[str, bool]:
    """Verify the agent's output against expected criteria"""
    verification_results = {}
    
    # Get result attributes (using getattr for safety)
    complexity_score = getattr(profile_result, 'complexity_score', 0)
    risk_flags = getattr(profile_result, 'risk_flags', [])
    cognitive_indicators = getattr(profile_result, 'cognitive_indicators', 
                                 getattr(profile_result, 'cognitive_assessments', []))
    info_gaps = getattr(profile_result, 'information_gaps', [])
    
    # 1. Verify complexity score is within expected range
    verification_results['complexity_in_range'] = (
        test_case.expected_complexity_range[0] <= complexity_score <= test_case.expected_complexity_range[1]
    )
    
    # 2. Verify minimum number of risk flags
    actual_risk_count = len(risk_flags) if risk_flags else 0
    verification_results['min_risk_flags'] = actual_risk_count >= test_case.expected_min_risk_flags
    
    # 3. Verify minimum number of cognitive indicators
    actual_cog_count = len(cognitive_indicators) if cognitive_indicators else 0
    verification_results['min_cognitive_indicators'] = actual_cog_count >= test_case.expected_min_cognitive_indicators
    
    # 4. Check for expected risk types (if specified)
    if test_case.expected_risk_types:
        risk_text = str(risk_flags).lower() if risk_flags else ""
        type_matches = []
        for risk_type in test_case.expected_risk_types:
            # Check if risk type is mentioned in any risk flag
            type_found = any(risk_type.lower() in str(flag).lower() for flag in risk_flags) if risk_flags else False
            type_matches.append(type_found)
        
        # Require at least half of expected risk types to be found
        verification_results['expected_risk_types'] = sum(type_matches) >= len(test_case.expected_risk_types) / 2
    
    return verification_results

def print_test_results(test_case: TestCase, profile_result: Any, verification_results: Dict[str, bool]):
    """Print formatted test results"""
    print(f"\n{'='*80}")
    print(f"📋 TEST CASE: {test_case.name}")
    print(f"{'='*80}")
    print(f"📝 Description: {test_case.description}")
    
    # Get result attributes
    complexity_score = getattr(profile_result, 'complexity_score', 0)
    risk_flags = getattr(profile_result, 'risk_flags', [])
    cognitive_indicators = getattr(profile_result, 'cognitive_indicators', 
                                 getattr(profile_result, 'cognitive_assessments', []))
    info_gaps = getattr(profile_result, 'information_gaps', [])
    
    print(f"\n📊 PROFILING RESULTS:")
    print(f"  • Complexity Score: {complexity_score} (expected range: {test_case.expected_complexity_range})")
    print(f"  • Risk Flags Found: {len(risk_flags) if risk_flags else 0} (expected min: {test_case.expected_min_risk_flags})")
    print(f"  • Cognitive Indicators: {len(cognitive_indicators) if cognitive_indicators else 0} (expected min: {test_case.expected_min_cognitive_indicators})")
    print(f"  • Information Gaps: {len(info_gaps) if info_gaps else 0}")
    
    if risk_flags and len(risk_flags) > 0:
        print(f"\n  🚨 Risk Flags Detected:")
        for i, flag in enumerate(risk_flags, 1):
            flag_text = flag.model_dump() if hasattr(flag, 'model_dump') else str(flag)
            print(f"    {i}. {flag_text}")
    
    if cognitive_indicators and len(cognitive_indicators) > 0:
        print(f"\n  🧠 Cognitive Indicators:")
        for i, indicator in enumerate(cognitive_indicators, 1):
            ind_text = indicator.model_dump() if hasattr(indicator, 'model_dump') else str(indicator)
            print(f"    {i}. {ind_text}")
    
    print(f"\n✅ VERIFICATION RESULTS:")
    all_passed = True
    for check_name, passed in verification_results.items():
        status = "PASS" if passed else "FAIL"
        icon = "✓" if passed else "✗"
        print(f"  {icon} {check_name}: {status}")
        if not passed:
            all_passed = False
    
    overall_status = "PASSED" if all_passed else "FAILED"
    status_icon = "✅" if all_passed else "❌"
    print(f"\n{status_icon} TEST CASE {overall_status}")
    
    return all_passed

def run_test_suite():
    """Run all test cases"""
    logger.info("="*80)
    logger.info("🧪 STARTING PROFILER AGENT TEST SUITE")
    logger.info("="*80)
    
    # Create test cases
    test_cases = create_test_cases()
    
    # Initialize agent once (more efficient)
    logger.info("Initializing ProfilerAgent...")
    agent = ProfilerAgent()
    
    # Run all tests
    total_cases = len(test_cases)
    passed_cases = 0
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\nRunning Test Case {i}/{total_cases}: {test_case.name}")
        
        try:
            # Execute profiling
            profile_result = agent.execute(
                patient_data=test_case.patient_data, 
                clinical_history=test_case.clinical_history
            )
            
            # Verify results
            verification_results = verify_test_result(test_case, profile_result)
            
            # Print results
            case_passed = print_test_results(test_case, profile_result, verification_results)
            
            if case_passed:
                passed_cases += 1
                
        except Exception as e:
            logger.error(f"❌ Test case '{test_case.name}' failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"\n❌ TEST CASE FAILED: {test_case.name}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("📊 TEST SUITE SUMMARY")
    print(f"{'='*80}")
    print(f"Total Test Cases: {total_cases}")
    print(f"Passed: {passed_cases}")
    print(f"Failed: {total_cases - passed_cases}")
    
    if passed_cases == total_cases:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"⚠️  {total_cases - passed_cases} test(s) failed")
    
    return passed_cases == total_cases

if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)