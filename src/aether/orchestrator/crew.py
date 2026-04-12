import time
import uuid
from typing import Callable, Optional
from crewai import Crew, Process
from aether.agents.intake import IntakeAgent
from aether.agents.history import ClinicalHistoryAgent
from aether.agents.profiler import ProfilerAgent
from aether.agents.planner import AssessmentPlannerAgent
from aether.agents.brief_writer import BriefWriterAgent
from aether.agents.qa import QAAgent
from aether.orchestrator.context_store import context_store
from aether.schemas.clinical import ReferralInput, OrchestratorOutput, OrchestratorMetadata
from aether.utils.logger import logger


class AetherCrew:
    """AETHER Multi-Agent Clinical Assessment Orchestrator."""
    
    def __init__(self):
        self.intake_agent = IntakeAgent()
        self.history_agent = ClinicalHistoryAgent()
        self.profiler_agent = ProfilerAgent()
        self.planner_agent = AssessmentPlannerAgent()
        self.brief_writer_agent = BriefWriterAgent()
        self.qa_agent = QAAgent()
    
    def execute(self, referral_input: ReferralInput) -> OrchestratorOutput:
        """Execute the full AETHER workflow."""
        start_time = time.time()
        encounter_id = referral_input.encounter_id or str(uuid.uuid4())
        
        logger.info(f"🚀 Starting AETHER workflow for encounter: {encounter_id}")
        
        # Initialize context
        context_store.initialize_context(encounter_id)
        
        try:
            # STEP 1: Intake - Extract patient data
            logger.info("📥 STEP 1: Patient data extraction")
            patient_data = self.intake_agent.execute(referral_input)
            context_store.add_audit_entry(
                encounter_id, "IntakeAgent", "extracted_patient_data", patient_data.model_dump()
            )
            
            # STEP 2: Clinical History - Structure medical history
            logger.info("📚 STEP 2: Clinical history structuring")
            clinical_history = self.history_agent.execute(patient_data, referral_input.referral_text)
            context_store.update_context(encounter_id, patient_history=clinical_history)
            context_store.add_audit_entry(
                encounter_id, "ClinicalHistoryAgent", "structured_history", clinical_history.model_dump()
            )
            
            # STEP 3: Profiler - Risk and cognitive analysis
            logger.info("👤 STEP 3: Risk profiling and cognitive analysis")
            patient_profile = self.profiler_agent.execute(patient_data, clinical_history)
            context_store.update_context(
                encounter_id,
                patient_profile=patient_profile,
                risk_flags=patient_profile.risk_flags
            )
            context_store.add_audit_entry(
                encounter_id, "ProfilerAgent", "generated_profile", patient_profile.model_dump()
            )
            
            # STEP 4: Planner - Assessment battery design
            logger.info("📋 STEP 4: Assessment battery planning")
            assessment_plan = self.planner_agent.execute(patient_data, patient_profile)
            context_store.update_context(encounter_id, assessment_plan=assessment_plan)
            context_store.add_audit_entry(
                encounter_id, "AssessmentPlannerAgent", "designed_plan", assessment_plan.model_dump()
            )
            
            # STEP 5: Brief Writer - Synthesize clinical brief
            logger.info("📝 STEP 5: Clinical brief composition")
            clinical_brief = self.brief_writer_agent.execute(
                patient_data, clinical_history, patient_profile, assessment_plan
            )
            context_store.add_audit_entry(
                encounter_id, "BriefWriterAgent", "composed_brief", clinical_brief.model_dump()
            )
            
            # STEP 6: QA - Quality validation
            logger.info("✅ STEP 6: Quality assurance validation")
            qa_result = self.qa_agent.execute(
                patient_data, clinical_history, patient_profile, assessment_plan, clinical_brief
            )
            context_store.add_audit_entry(
                encounter_id, "QAAgent", "validated_output", qa_result.model_dump()
            )
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"✨ AETHER workflow completed in {processing_time_ms}ms - "
                f"Status: {qa_result.overall_status}"
            )
            
            # Construct final output
            output = OrchestratorOutput(
                patient_data=patient_data,
                clinical_history=clinical_history,
                patient_profile=patient_profile,
                assessment_plan=assessment_plan,
                clinical_brief=clinical_brief,
                qa_result=qa_result,
                metadata=OrchestratorMetadata(
                    processing_time_ms=processing_time_ms,
                    agent_execution_order=[
                        "IntakeAgent",
                        "ClinicalHistoryAgent",
                        "ProfilerAgent",
                        "AssessmentPlannerAgent",
                        "BriefWriterAgent",
                        "QAAgent",
                    ]
                )
            )
            
            # Clear context after successful completion
            context_store.clear_context(encounter_id)
            
            return output
            
        except Exception as e:
            logger.error(f"❌ AETHER workflow failed for encounter {encounter_id}: {e}")
            context_store.clear_context(encounter_id)
            raise
    
    def execute_with_streaming(
        self,
        referral_input: ReferralInput,
        on_progress: Callable[[str, dict], None]
    ) -> OrchestratorOutput:
        """Execute with progress callbacks for streaming."""
        start_time = time.time()
        encounter_id = referral_input.encounter_id or str(uuid.uuid4())
        
        context_store.initialize_context(encounter_id)
        
        try:
            # STEP 1
            on_progress("intake", {"status": "running", "message": "Extracting patient data..."})
            patient_data = self.intake_agent.execute(referral_input)
            on_progress("intake", {"status": "complete", "data": patient_data.model_dump()})
            
            # STEP 2
            on_progress("history", {"status": "running", "message": "Structuring clinical history..."})
            clinical_history = self.history_agent.execute(patient_data,referral_input.referral_text)
            on_progress("history", {"status": "complete", "data": clinical_history.model_dump()})
            
            # STEP 3
            on_progress("profiler", {"status": "running", "message": "Analyzing risk profile..."})
            patient_profile = self.profiler_agent.execute(patient_data, clinical_history)
            on_progress("profiler", {"status": "complete", "data": patient_profile.model_dump()})
            
            # STEP 4
            on_progress("planner", {"status": "running", "message": "Designing assessment battery..."})
            assessment_plan = self.planner_agent.execute(patient_data, patient_profile)
            on_progress("planner", {"status": "complete", "data": assessment_plan.model_dump()})
            
            # STEP 5
            on_progress("brief", {"status": "running", "message": "Composing clinical brief..."})
            clinical_brief = self.brief_writer_agent.execute(
                patient_data, clinical_history, patient_profile, assessment_plan
            )
            on_progress("brief", {"status": "complete", "data": clinical_brief.model_dump()})
            
            # STEP 6
            on_progress("qa", {"status": "running", "message": "Validating output quality..."})
            qa_result = self.qa_agent.execute(
                patient_data, clinical_history, patient_profile, assessment_plan, clinical_brief
            )
            on_progress("qa", {"status": "complete", "data": qa_result.model_dump()})
            
            output = OrchestratorOutput(
                patient_data=patient_data,
                clinical_history=clinical_history,
                patient_profile=patient_profile,
                assessment_plan=assessment_plan,
                clinical_brief=clinical_brief,
                qa_result=qa_result,
                metadata=OrchestratorMetadata(
                    processing_time_ms=int((time.time() - start_time) * 1000),
                    agent_execution_order=[
                        "IntakeAgent",
                        "ClinicalHistoryAgent",
                        "ProfilerAgent",
                        "AssessmentPlannerAgent",
                        "BriefWriterAgent",
                        "QAAgent",
                    ]
                )
            )
            
            context_store.clear_context(encounter_id)
            on_progress("complete", {"status": "complete", "data": output.model_dump()})
            
            return output
            
        except Exception as e:
            on_progress("error", {"status": "error", "error": str(e)})
            context_store.clear_context(encounter_id)
            raise