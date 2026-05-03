# aether/orchestrator/graph.py
"""
AETHER Multi-Agent Clinical Assessment Graph Orchestrator
LangGraph-based state machine with native state management
"""
import time
import uuid
from typing import Callable, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from aether.agents.intake import IntakeAgent
from aether.agents.history import ClinicalHistoryAgent
from aether.agents.profiler import ProfilerAgent
from aether.agents.planner import AssessmentPlannerAgent
from aether.agents.brief_writer import BriefWriterAgent
from aether.agents.qa import QAAgent
from aether.orchestrator.state import AetherState, AuditEntry, RiskFlag
from aether.schemas.clinical import (
    ReferralInput, 
    OrchestratorOutput, 
    OrchestratorMetadata
)
from aether.utils.logger import logger


class AetherGraph:
    """AETHER Multi-Agent Clinical Assessment Orchestrator using LangGraph."""
    
    def __init__(self, use_persistence: bool = False, db_path: str = "aether_checkpoints.db"):
        """
        Initialize AETHER graph.
        
        Args:
            use_persistence: If True, uses SQLite for checkpoint persistence
            db_path: Path to SQLite database for checkpoints
        """
        # Initialize all agents
        self.intake_agent = IntakeAgent()
        self.history_agent = ClinicalHistoryAgent()
        self.profiler_agent = ProfilerAgent()
        self.planner_agent = AssessmentPlannerAgent()
        self.brief_writer_agent = BriefWriterAgent()
        self.qa_agent = QAAgent()
        
        # Choose checkpointer
        if use_persistence:
            checkpointer = SqliteSaver.from_conn_string(db_path)
            logger.info(f"Using SQLite persistence: {db_path}")
        else:
            checkpointer = MemorySaver()
            logger.info("Using in-memory state management")
        
        # Build the graph
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=checkpointer)
        
        # Progress callback (for streaming)
        self.progress_callback: Optional[Callable[[str, dict], None]] = None
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine for clinical workflow."""
        workflow = StateGraph(AetherState)
        
        # Add nodes for each agent step
        workflow.add_node("intake", self._intake_node)
        workflow.add_node("history", self._history_node)
        workflow.add_node("profiler", self._profiler_node)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("brief_writer", self._brief_writer_node)
        workflow.add_node("qa", self._qa_node)
        
        # Define the workflow edges (sequential pipeline)
        workflow.set_entry_point("intake")
        workflow.add_edge("intake", "history")
        workflow.add_edge("history", "profiler")
        workflow.add_edge("profiler", "planner")
        workflow.add_edge("planner", "brief_writer")
        workflow.add_edge("brief_writer", "qa")
        workflow.add_edge("qa", END)
        
        return workflow
    
    def _create_audit_entry(
        self, 
        state: AetherState, 
        agent: str, 
        action: str, 
        data: dict
    ) -> AuditEntry:
        """Create a single audit entry."""
        return AuditEntry(
            timestamp=time.time(),
            encounter_id=state["encounter_id"],
            agent=agent,
            action=action,
            data=data,
            step_number=len(state.get("audit_log", []))
        )
    
    def _create_risk_flag(
        self,
        flag_type: str,
        severity: str,
        description: str,
        identified_by: str
    ) -> RiskFlag:
        """Create a risk flag entry."""
        return RiskFlag(
            flag_type=flag_type,
            severity=severity,
            description=description,
            identified_by=identified_by,
            timestamp=time.time()
        )
    
    def _record_step_timing(
        self, 
        state: AetherState, 
        step_name: str, 
        duration_ms: float
    ) -> dict[str, float]:
        """Record timing for a step."""
        timings = state.get("step_timings", {}).copy()
        timings[step_name] = duration_ms
        return timings
    
    def _intake_node(self, state: AetherState) -> dict:
        """STEP 1: Patient data extraction."""
        step_start = time.time()
        logger.info("📥 STEP 1: Patient data extraction")
        
        if self.progress_callback:
            self.progress_callback("intake", {
                "status": "running", 
                "message": "Extracting patient data...",
                "encounter_id": state["encounter_id"]
            })
        
        try:
            patient_data = self.intake_agent.execute(state["referral_input"])
            
            step_duration = (time.time() - step_start) * 1000
            
            if self.progress_callback:
                self.progress_callback("intake", {
                    "status": "complete", 
                    "data": patient_data.model_dump(),
                    "duration_ms": step_duration
                })
            
            # Create audit entry
            audit_entry = self._create_audit_entry(
                state, 
                "IntakeAgent", 
                "extracted_patient_data", 
                patient_data.model_dump()
            )
            
            # Return state updates (LangGraph merges these)
            return {
                "patient_data": patient_data,
                "current_step": "intake",
                "agent_execution_order": ["IntakeAgent"],
                "steps_completed": ["intake"],
                "audit_log": [audit_entry],
                "step_timings": self._record_step_timing(state, "intake", step_duration)
            }
            
        except Exception as e:
            logger.error(f"Intake node failed: {e}")
            return {"error": str(e)}
    
    def _history_node(self, state: AetherState) -> dict:
        """STEP 2: Clinical history structuring."""
        step_start = time.time()
        logger.info("📚 STEP 2: Clinical history structuring")
        
        if self.progress_callback:
            self.progress_callback("history", {
                "status": "running", 
                "message": "Structuring clinical history..."
            })
        
        try:
            clinical_history = self.history_agent.execute(
                state["patient_data"],
                state["referral_input"].referral_text
            )
            
            step_duration = (time.time() - step_start) * 1000
            
            if self.progress_callback:
                self.progress_callback("history", {
                    "status": "complete", 
                    "data": clinical_history.model_dump(),
                    "duration_ms": step_duration
                })
            
            audit_entry = self._create_audit_entry(
                state,
                "ClinicalHistoryAgent",
                "structured_history",
                clinical_history.model_dump()
            )
            
            return {
                "clinical_history": clinical_history,
                "current_step": "history",
                "agent_execution_order": ["ClinicalHistoryAgent"],
                "steps_completed": ["history"],
                "audit_log": [audit_entry],
                "step_timings": self._record_step_timing(state, "history", step_duration)
            }
            
        except Exception as e:
            logger.error(f"History node failed: {e}")
            return {"error": str(e)}
    
    def _profiler_node(self, state: AetherState) -> dict:
        """STEP 3: Risk profiling and cognitive analysis."""
        step_start = time.time()
        logger.info("👤 STEP 3: Risk profiling and cognitive analysis")
        
        if self.progress_callback:
            self.progress_callback("profiler", {
                "status": "running", 
                "message": "Analyzing risk profile..."
            })
        
        try:
            patient_profile = self.profiler_agent.execute(
                state["patient_data"],
                state["clinical_history"]
            )
            
            step_duration = (time.time() - step_start) * 1000
            
            if self.progress_callback:
                self.progress_callback("profiler", {
                    "status": "complete", 
                    "data": patient_profile.model_dump(),
                    "duration_ms": step_duration
                })
            
            audit_entry = self._create_audit_entry(
                state,
                "ProfilerAgent",
                "generated_profile",
                patient_profile.model_dump()
            )
            
            # Convert risk flags from profile to RiskFlag objects
            risk_flags = [
                self._create_risk_flag(
                    flag_type=flag,
                    severity=self._determine_severity(flag, patient_profile),
                    description=f"Risk identified: {flag}",
                    identified_by="ProfilerAgent"
                )
                for flag in patient_profile.risk_flags
            ]
            
            return {
                "patient_profile": patient_profile,
                "current_step": "profiler",
                "agent_execution_order": ["ProfilerAgent"],
                "steps_completed": ["profiler"],
                "audit_log": [audit_entry],
                "risk_flags": risk_flags,
                "step_timings": self._record_step_timing(state, "profiler", step_duration)
            }
            
        except Exception as e:
            logger.error(f"Profiler node failed: {e}")
            return {"error": str(e)}
    
    def _determine_severity(self, flag: str, profile) -> str:
        """Determine severity of a risk flag based on context."""
        # Simple heuristic - can be made more sophisticated
        high_risk_keywords = ["fall", "suicide", "severe", "critical"]
        medium_risk_keywords = ["moderate", "concern", "monitoring"]
        
        flag_lower = flag.lower()
        
        if any(keyword in flag_lower for keyword in high_risk_keywords):
            return "high"
        elif any(keyword in flag_lower for keyword in medium_risk_keywords):
            return "medium"
        else:
            return "low"
    
    def _planner_node(self, state: AetherState) -> dict:
        """STEP 4: Assessment battery planning."""
        step_start = time.time()
        logger.info("📋 STEP 4: Assessment battery planning")
        
        if self.progress_callback:
            self.progress_callback("planner", {
                "status": "running", 
                "message": "Designing assessment battery..."
            })
        
        try:
            assessment_plan = self.planner_agent.execute(
                state["patient_data"],
                state["patient_profile"]
            )
            
            step_duration = (time.time() - step_start) * 1000
            
            if self.progress_callback:
                self.progress_callback("planner", {
                    "status": "complete", 
                    "data": assessment_plan.model_dump(),
                    "duration_ms": step_duration
                })
            
            audit_entry = self._create_audit_entry(
                state,
                "AssessmentPlannerAgent",
                "designed_plan",
                assessment_plan.model_dump()
            )
            
            return {
                "assessment_plan": assessment_plan,
                "current_step": "planner",
                "agent_execution_order": ["AssessmentPlannerAgent"],
                "steps_completed": ["planner"],
                "audit_log": [audit_entry],
                "step_timings": self._record_step_timing(state, "planner", step_duration)
            }
            
        except Exception as e:
            logger.error(f"Planner node failed: {e}")
            return {"error": str(e)}
    
    def _brief_writer_node(self, state: AetherState) -> dict:
        """STEP 5: Clinical brief composition."""
        step_start = time.time()
        logger.info("📝 STEP 5: Clinical brief composition")
        
        if self.progress_callback:
            self.progress_callback("brief", {
                "status": "running", 
                "message": "Composing clinical brief..."
            })
        
        try:
            clinical_brief = self.brief_writer_agent.execute(
                state["patient_data"],
                state["clinical_history"],
                state["patient_profile"],
                state["assessment_plan"]
            )
            
            step_duration = (time.time() - step_start) * 1000
            
            if self.progress_callback:
                self.progress_callback("brief", {
                    "status": "complete", 
                    "data": clinical_brief.model_dump(),
                    "duration_ms": step_duration
                })
            
            audit_entry = self._create_audit_entry(
                state,
                "BriefWriterAgent",
                "composed_brief",
                clinical_brief.model_dump()
            )
            
            return {
                "clinical_brief": clinical_brief,
                "current_step": "brief_writer",
                "agent_execution_order": ["BriefWriterAgent"],
                "steps_completed": ["brief_writer"],
                "audit_log": [audit_entry],
                "step_timings": self._record_step_timing(state, "brief_writer", step_duration)
            }
            
        except Exception as e:
            logger.error(f"Brief writer node failed: {e}")
            return {"error": str(e)}
    
    def _qa_node(self, state: AetherState) -> dict:
        """STEP 6: Quality assurance validation."""
        step_start = time.time()
        logger.info("✅ STEP 6: Quality assurance validation")
        
        if self.progress_callback:
            self.progress_callback("qa", {
                "status": "running", 
                "message": "Validating output quality..."
            })
        
        try:
            qa_result = self.qa_agent.execute(
                state["patient_data"],
                state["clinical_history"],
                state["patient_profile"],
                state["assessment_plan"],
                state["clinical_brief"]
            )
            
            step_duration = (time.time() - step_start) * 1000
            
            if self.progress_callback:
                self.progress_callback("qa", {
                    "status": "complete", 
                    "data": qa_result.model_dump(),
                    "duration_ms": step_duration
                })
            
            audit_entry = self._create_audit_entry(
                state,
                "QAAgent",
                "validated_output",
                qa_result.model_dump()
            )
            
            # Add QA-specific risk flags if validation fails
            qa_risk_flags = []
            if qa_result.overall_status == "FAIL":
                qa_risk_flags.append(
                    self._create_risk_flag(
                        flag_type="qa_validation_failure",
                        severity="high",
                        description=f"QA validation failed: {len(qa_result.failed_checks)} checks failed",
                        identified_by="QAAgent"
                    )
                )
            
            return {
                "qa_result": qa_result,
                "current_step": "qa",
                "agent_execution_order": ["QAAgent"],
                "steps_completed": ["qa"],
                "audit_log": [audit_entry],
                "risk_flags": qa_risk_flags,
                "step_timings": self._record_step_timing(state, "qa", step_duration)
            }
            
        except Exception as e:
            logger.error(f"QA node failed: {e}")
            return {"error": str(e)}
    
    def execute(self, referral_input: ReferralInput) -> OrchestratorOutput:
        """Execute the full AETHER workflow without streaming."""
        start_time = time.time()
        encounter_id = referral_input.encounter_id or str(uuid.uuid4())
        
        logger.info(f"🚀 Starting AETHER workflow for encounter: {encounter_id}")
        
        try:
            # Initialize state
            initial_state: AetherState = {
                "referral_input": referral_input,
                "encounter_id": encounter_id,
                "start_time": start_time,
                "patient_data": None,
                "clinical_history": None,
                "patient_profile": None,
                "assessment_plan": None,
                "clinical_brief": None,
                "qa_result": None,
                "audit_log": [],
                "risk_flags": [],
                "current_step": "init",
                "steps_completed": [],
                "error": None,
                "agent_execution_order": [],
                "step_timings": {}
            }
            
            # Execute the graph
            config = {"configurable": {"thread_id": encounter_id}}
            final_state = self.app.invoke(initial_state, config)
            
            # Check for errors
            if final_state.get("error"):
                raise Exception(final_state["error"])
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"✨ AETHER workflow completed in {processing_time_ms}ms - "
                f"Status: {final_state['qa_result'].overall_status}"
            )
            
            # Log audit summary
            self._log_audit_summary(final_state)
            
            # Construct final output
            output = OrchestratorOutput(
                patient_data=final_state["patient_data"],
                clinical_history=final_state["clinical_history"],
                patient_profile=final_state["patient_profile"],
                assessment_plan=final_state["assessment_plan"],
                clinical_brief=final_state["clinical_brief"],
                qa_result=final_state["qa_result"],
                metadata=OrchestratorMetadata(
                    processing_time_ms=processing_time_ms,
                    agent_execution_order=final_state["agent_execution_order"]
                )
            )
            
            return output
            
        except Exception as e:
            logger.error(f"❌ AETHER workflow failed for encounter {encounter_id}: {e}")
            raise
    
    def execute_with_streaming(
        self,
        referral_input: ReferralInput,
        on_progress: Callable[[str, dict], None]
    ) -> OrchestratorOutput:
        """Execute with progress callbacks for streaming."""
        self.progress_callback = on_progress
        
        start_time = time.time()
        encounter_id = referral_input.encounter_id or str(uuid.uuid4())
        
        try:
            # Initialize state
            initial_state: AetherState = {
                "referral_input": referral_input,
                "encounter_id": encounter_id,
                "start_time": start_time,
                "patient_data": None,
                "clinical_history": None,
                "patient_profile": None,
                "assessment_plan": None,
                "clinical_brief": None,
                "qa_result": None,
                "audit_log": [],
                "risk_flags": [],
                "current_step": "init",
                "steps_completed": [],
                "error": None,
                "agent_execution_order": [],
                "step_timings": {}
            }
            
            # Execute with streaming
            config = {"configurable": {"thread_id": encounter_id}}
            
            final_state = None
            for event in self.app.stream(initial_state, config, stream_mode="values"):
                final_state = event
                # Progress callbacks are handled within each node
            
            if not final_state:
                raise Exception("Graph execution produced no final state")
            
            # Check for errors
            if final_state.get("error"):
                on_progress("error", {"status": "error", "error": final_state["error"]})
                raise Exception(final_state["error"])
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Log audit summary
            self._log_audit_summary(final_state)
            
            # Construct output
            output = OrchestratorOutput(
                patient_data=final_state["patient_data"],
                clinical_history=final_state["clinical_history"],
                patient_profile=final_state["patient_profile"],
                assessment_plan=final_state["assessment_plan"],
                clinical_brief=final_state["clinical_brief"],
                qa_result=final_state["qa_result"],
                metadata=OrchestratorMetadata(
                    processing_time_ms=processing_time_ms,
                    agent_execution_order=final_state["agent_execution_order"]
                )
            )
            
            on_progress("complete", {
                "status": "complete", 
                "data": output.model_dump(),
                "audit_summary": self._get_audit_summary(final_state)
            })
            
            return output
            
        except Exception as e:
            on_progress("error", {"status": "error", "error": str(e)})
            raise
        finally:
            self.progress_callback = None
    
    def _log_audit_summary(self, state: AetherState):
        """Log audit trail summary."""
        audit_log = state.get("audit_log", [])
        risk_flags = state.get("risk_flags", [])
        step_timings = state.get("step_timings", {})
        
        logger.info(f"📊 Audit Summary for {state['encounter_id']}:")
        logger.info(f"   - Total audit entries: {len(audit_log)}")
        logger.info(f"   - Risk flags identified: {len(risk_flags)}")
        logger.info(f"   - Steps completed: {', '.join(state.get('steps_completed', []))}")
        
        if step_timings:
            logger.info("   - Step timings:")
            for step, duration in step_timings.items():
                logger.info(f"     • {step}: {duration:.2f}ms")
        
        if risk_flags:
            logger.info("   - Risk flags:")
            for flag in risk_flags:
                logger.info(f"     • [{flag.severity.upper()}] {flag.flag_type}: {flag.description}")
    
    def _get_audit_summary(self, state: AetherState) -> dict:
        """Get audit summary as dict."""
        return {
            "encounter_id": state["encounter_id"],
            "total_audit_entries": len(state.get("audit_log", [])),
            "risk_flags_count": len(state.get("risk_flags", [])),
            "steps_completed": state.get("steps_completed", []),
            "step_timings": state.get("step_timings", {}),
            "risk_flags": [
                {
                    "type": flag.flag_type,
                    "severity": flag.severity,
                    "description": flag.description,
                    "identified_by": flag.identified_by
                }
                for flag in state.get("risk_flags", [])
            ]
        }
    
    def get_state_history(self, encounter_id: str) -> list[AetherState]:
        """
        Retrieve state history for an encounter.
        Requires persistent checkpointer.
        """
        config = {"configurable": {"thread_id": encounter_id}}
        history = []
        
        try:
            for state in self.app.get_state_history(config):
                history.append(state.values)
            return history
        except Exception as e:
            logger.error(f"Failed to retrieve state history: {e}")
            return []
    
    def resume_from_checkpoint(self, encounter_id: str) -> OrchestratorOutput:
        """
        Resume execution from last checkpoint.
        Requires persistent checkpointer.
        """
        config = {"configurable": {"thread_id": encounter_id}}
        
        try:
            # Get the last state
            current_state = self.app.get_state(config)
            
            if not current_state:
                raise ValueError(f"No checkpoint found for encounter: {encounter_id}")
            
            logger.info(f"Resuming from step: {current_state.values['current_step']}")
            
            # Continue execution
            final_state = self.app.invoke(None, config)
            
            # Build output
            return OrchestratorOutput(
                patient_data=final_state["patient_data"],
                clinical_history=final_state["clinical_history"],
                patient_profile=final_state["patient_profile"],
                assessment_plan=final_state["assessment_plan"],
                clinical_brief=final_state["clinical_brief"],
                qa_result=final_state["qa_result"],
                metadata=OrchestratorMetadata(
                    processing_time_ms=int((time.time() - final_state["start_time"]) * 1000),
                    agent_execution_order=final_state["agent_execution_order"]
                )
            )
            
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")
            raise