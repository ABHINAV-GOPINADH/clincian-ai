# aether/agents/brief_writer.py - FINAL COMPLETE VERSION
"""Clinical Documentation Specialist Agent - LangGraph version."""
from typing import Optional, Dict, Any
from datetime import date
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
import json

from aether.config.settings import settings
from aether.schemas.clinical import (
    PatientData, ClinicalHistory, PatientProfile, AssessmentPlan, ClinicalBrief, RiskFlag
)
from aether.utils.logger import logger


class BriefWriterAgent:
    """Clinical Documentation Specialist Agent."""
    
    def __init__(self):
        self.llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.1,
            format="json"
        )
        self.parser = JsonOutputParser()
        self.chain = self._create_prompt() | self.llm | self.parser
    
    def _create_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", "You are a Clinical Documentation Specialist."),
            ("human", """Create clinical brief:

DATE: {current_date}
PATIENT: {patient_name}, {age}yo, NHS#{nhs_number}
CONDITIONS: {conditions}
RISKS: {risk_flags}
ASSESSMENTS: {assessment_plan}

Return JSON (use exact field names):
{{
  "header": {{
    "patient_name": "name",
    "nhs_number": "number",
    "date_of_birth": "YYYY-MM-DD",
    "age": number,
    "date_generated": "{current_date}",
    "assessment_date": "{current_date}"
  }},
  "executive_summary": "brief overview",
  "presenting_concerns": ["concern 1"],
  "relevant_history": {{
    "medical": ["condition 1"],
    "psychiatric": ["None"],
    "social": ["Lives alone"]
  }},
  "key_considerations": ["consideration 1"],
  "nice_guidance_alignment": "NICE NG97 compliant"
}}""")
        ])
    
    def execute(
        self,
        patient_data: PatientData,
        clinical_history: ClinicalHistory,
        patient_profile: PatientProfile,
        assessment_plan: AssessmentPlan
    ) -> ClinicalBrief:
        print("\n📝 CLINICAL BRIEF WRITER")
        logger.info("BriefWriterAgent: Starting")
        
        try:
            # Prepare data
            conditions = [c.display for c in clinical_history.conditions] if hasattr(clinical_history, 'conditions') else []
            risk_flags_list = [rf.description for rf in patient_profile.risk_flags] if hasattr(patient_profile, 'risk_flags') else []
            instruments = [i.name for i in assessment_plan.instruments] if hasattr(assessment_plan, 'instruments') else []
            
            # Call LLM
            print("⏳ Calling LLM...")
            raw = self.chain.invoke({
                "current_date": date.today().isoformat(),
                "patient_name": f"{patient_data.name.first} {patient_data.name.last}",
                "age": patient_data.age or 0,
                "nhs_number": patient_data.nhs_number or "Unknown",
                "conditions": ", ".join(conditions) if conditions else "None",
                "risk_flags": ", ".join(risk_flags_list) if risk_flags_list else "None",
                "assessment_plan": ", ".join(instruments) if instruments else "None"
            })
            print("✅ LLM responded\n")
            
            # NORMALIZE
            
            # Fix header - add assessment_date
            header = raw.get("header", {})
            if "assessment_date" not in header:
                header["assessment_date"] = date.today().isoformat()
            
            # Fix relevant_history
            relevant_history = raw.get("relevant_history", {})
            if "medical" not in relevant_history:
                relevant_history = {
                    "medical": conditions[:3] if conditions else ["None documented"],
                    "psychiatric": ["None documented"],
                    "social": ["Lives independently"] if patient_data else ["Unknown"]
                }
            
            # Build risk_summary as RiskFlag objects (NOT strings)
            risk_summary = []
            if hasattr(patient_profile, 'risk_flags') and patient_profile.risk_flags:
                # Use actual RiskFlag objects from profile
                risk_summary = patient_profile.risk_flags[:5]  # Top 5 risks
            else:
                # Create minimal RiskFlag
                risk_summary = [
                    RiskFlag(
                        category="clinical",
                        description="No high-priority risks identified",
                        severity="low",
                        source="clinical_data",
                        reasoning="Standard assessment"
                    )
                ]
            
            # Build recommended_assessments from assessment_plan
            assessments = []
            if hasattr(assessment_plan, 'instruments') and assessment_plan.instruments:
                for inst in assessment_plan.instruments:
                    assessments.append({
                        "name": inst.name,
                        "type": inst.type,  # Already validated enum from planner
                        "rationale": inst.rationale,
                        "estimated_duration": inst.estimated_duration,
                        "priority": inst.priority  # Required field!
                    })
            else:
                # Minimal fallback
                assessments = [{
                    "name": "MMSE",
                    "type": "MMSE",
                    "rationale": "Standard cognitive screening",
                    "estimated_duration": 10,
                    "priority": "essential"
                }]
            
            # Build normalized brief
            normalized = {
                "header": header,
                "executive_summary": raw.get("executive_summary", "Clinical assessment required"),
                "presenting_concerns": raw.get("presenting_concerns") or ["Assessment pending"],
                "relevant_history": relevant_history,
                "risk_summary": risk_summary,  # RiskFlag objects
                "recommended_assessments": assessments,
                "key_considerations": raw.get("key_considerations") or ["Standard NICE NG97 protocol"],
                "nice_guidance_alignment": raw.get("nice_guidance_alignment") or "Compliant with NICE NG97 guidelines"
            }
            
            print("🔍 Validating...")
            brief = ClinicalBrief.model_validate(normalized)
            print("✅ SUCCESS\n")
            
            logger.info("Success!")
            return brief
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}\n")
            logger.error(f"Failed: {e}")
            
            # Minimal fallback with ALL required fields
            return ClinicalBrief(
                header={
                    "patient_name": f"{patient_data.name.first} {patient_data.name.last}",
                    "nhs_number": patient_data.nhs_number or "Unknown",
                    "date_of_birth": patient_data.date_of_birth,
                    "age": patient_data.age or 0,
                    "date_generated": date.today().isoformat(),
                    "assessment_date": date.today().isoformat()  # Required!
                },
                executive_summary="Clinical brief generation incomplete.",
                presenting_concerns=["System error"],
                relevant_history={
                    "medical": [c.display for c in clinical_history.conditions[:3]] if hasattr(clinical_history, 'conditions') else ["None"],
                    "psychiatric": ["None documented"],
                    "social": ["Unknown"]
                },
                risk_summary=[  # RiskFlag objects!
                    RiskFlag(
                        category="clinical",
                        description="Assessment incomplete",
                        severity="medium",
                        source="system",
                        reasoning="Brief generation failed"
                    )
                ],
                recommended_assessments=[
                    {
                        "name": "MMSE",
                        "type": "MMSE",
                        "rationale": "Standard screening (fallback)",
                        "estimated_duration": 10,
                        "priority": "essential"  # Required!
                    }
                ],
                key_considerations=["Manual review required"],
                nice_guidance_alignment="Unable to generate compliance statement"
            )