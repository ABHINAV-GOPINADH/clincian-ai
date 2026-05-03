# aether/agents/qa.py - FINAL COMPLETE VERSION
"""Clinical Quality Assurance Agent - LangGraph version with RAG."""
from typing import Optional, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
import json

from aether.config.settings import settings
from aether.schemas.clinical import (
    PatientData, ClinicalHistory, PatientProfile, AssessmentPlan, ClinicalBrief, QAResult,
    ClinicalAccuracy, NICECompliance, DataCompleteness, SafetyChecks
)
from aether.tools.rag_tool import nice_rag
from aether.utils.logger import logger


class QAAgent:
    """Clinical Quality Assurance Specialist Agent."""
    
    def __init__(self):
        self.llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.0,
            format="json"
        )
        self.parser = JsonOutputParser()
        self.chain = self._create_prompt() | self.llm | self.parser
    
    def _create_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", "You are a Clinical Quality Assurance Specialist."),
            ("human", """Perform QA validation on this clinical assessment:

PATIENT: {patient_name}
CLINICAL HISTORY: {history_summary}
RISK PROFILE: {risk_summary}
ASSESSMENT PLAN: {plan_summary}
CLINICAL BRIEF: {brief_summary}

NICE NG97 STANDARDS:
{nice_compliance}

Return JSON with EXACT field names:
{{
  "overall_status": "green/amber/red",
  "overall_score": number (0-100),
  "clinical_accuracy": number (0-100),
  "nice_compliance": number (0-100),
  "data_completeness": number (0-100),
  "safety_checks": number (0-100),
  "validation_checks": [
    {{
      "check_name": "name",
      "status": "PASS/FAIL/WARNING",
      "score": number,
      "details": "explanation"
    }}
  ],
  "failed_checks": ["check name if failed"],
  "warnings": ["warning if any"],
  "recommendations": ["recommendation"],
  "nice_compliance_status": "COMPLIANT/PARTIAL/NON_COMPLIANT"
}}

CRITICAL:
- overall_status MUST be lowercase: green/amber/red
- green (≥90), amber (70-89), red (<70)
- Provide individual scores for: clinical_accuracy, nice_compliance, data_completeness, safety_checks""")
        ])
    
    def _retrieve_nice_compliance_standards(self) -> str:
        """Retrieve NICE NG97 quality standards using RAG."""
        print("\n🔍 RETRIEVING NICE NG97 QUALITY STANDARDS...")
        
        try:
            docs = nice_rag.retrieve_guidance(
                "NICE NG97 dementia assessment mandatory requirements quality standards", 
                top_k=3
            )
            
            if docs:
                standards = "\n\n---\n\n".join([doc.page_content for doc in docs])
                print(f"  ✅ Retrieved {len(docs)} quality standards")
                return standards
            else:
                return self._get_fallback_standards()
        except Exception as e:
            print(f"  ❌ RAG error: {e}")
            return self._get_fallback_standards()
    
    def _get_fallback_standards(self) -> str:
        return "NICE NG97: Use validated cognitive instruments, comprehensive risk assessment, patient safety priority, complete documentation."
    
    def execute(
        self,
        patient_data: PatientData,
        clinical_history: ClinicalHistory,
        patient_profile: PatientProfile,
        assessment_plan: AssessmentPlan,
        clinical_brief: ClinicalBrief
    ) -> QAResult:
        print("\n" + "="*80)
        print("✅ QUALITY ASSURANCE AGENT")
        print("="*80)
        logger.info("QAAgent: Starting validation")
        
        try:
            # Get NICE standards
            nice_compliance = self._retrieve_nice_compliance_standards()
            
            # Prepare summaries
            history_summary = f"{len(clinical_history.conditions) if hasattr(clinical_history, 'conditions') else 0} conditions"
            risk_summary = f"{len(patient_profile.risk_flags) if hasattr(patient_profile, 'risk_flags') else 0} risks"
            plan_summary = f"{len(assessment_plan.instruments) if hasattr(assessment_plan, 'instruments') else 0} instruments"
            brief_summary = clinical_brief.executive_summary if hasattr(clinical_brief, 'executive_summary') else "Brief generated"
            
            # Call LLM
            print("⏳ Calling LLM...")
            raw = self.chain.invoke({
                "patient_name": f"{patient_data.name.first} {patient_data.name.last}",
                "history_summary": history_summary,
                "risk_summary": risk_summary,
                "plan_summary": plan_summary,
                "brief_summary": brief_summary[:200],
                "nice_compliance": nice_compliance
            })
            print("✅ LLM responded\n")
            
            print("🔍 RAW:", json.dumps(raw, indent=2), "\n")
            
            # NORMALIZE
            
            # Fix overall_status to lowercase
            overall_status = str(raw.get("overall_status", "amber")).lower()
            if overall_status not in ["green", "amber", "red"]:
                overall_status = "amber"
            
            # Get scores
            overall_score = int(raw.get("overall_score", 75))
            clinical_accuracy = int(raw.get("clinical_accuracy", 80))
            nice_compliance_score = int(raw.get("nice_compliance", 80))
            data_completeness = int(raw.get("data_completeness", 80))
            safety_checks = int(raw.get("safety_checks", 80))
            
            # Auto-adjust status based on score
            if overall_score >= 90:
                overall_status = "green"
            elif overall_score < 70:
                overall_status = "red"
            else:
                overall_status = "amber"
            
            # Normalize validation_checks
            validation_checks = []
            for check in (raw.get("validation_checks") or []):
                status = str(check.get("status", "PASS")).upper()
                if status not in ["PASS", "FAIL", "WARNING"]:
                    status = "PASS"
                
                validation_checks.append({
                    "check_name": str(check.get("check_name", "Unknown")),
                    "status": status,
                    "score": int(check.get("score", 100)),
                    "details": str(check.get("details", "No details"))
                })
            
            # Normalize NICE compliance status
            nice_status = str(raw.get("nice_compliance_status", "COMPLIANT")).upper()
            if nice_status not in ["COMPLIANT", "PARTIAL", "NON_COMPLIANT"]:
                nice_status = "COMPLIANT"
            
            # Build normalized result
            normalized = {
                "overall_status": overall_status,
                "clinical_accuracy": ClinicalAccuracy(score=clinical_accuracy, issues=[]),
                "nice_compliance": NICECompliance(compliant=(nice_status == "COMPLIANT"), gaps=[]),
                "data_completeness": DataCompleteness(percentage=data_completeness, missing_fields=[]),
                "safety_checks": SafetyChecks(passed=(safety_checks >= 80), flags=[]),
                "recommendations": raw.get("recommendations") or []
            }
            
            print(f"✅ QA SUMMARY:")
            print(f"  Status: {overall_status.upper()} ({overall_score}/100)")
            print(f"  Clinical Accuracy: {clinical_accuracy}/100")
            print(f"  NICE Compliance: {nice_compliance_score}/100")
            print(f"  Data Completeness: {data_completeness}/100")
            print(f"  Safety Checks: {safety_checks}/100")
            print()
            
            # Validate
            print("🔍 Validating...")
            qa_result = QAResult.model_validate(normalized)
            print("✅ SUCCESS\n" + "="*80 + "\n")
            
            logger.info(f"QA complete: {qa_result.overall_status} ({qa_result.overall_score}/100)")
            
            return qa_result
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}\n")
            logger.error(f"QA failed: {e}")
            
            # Fallback
            return QAResult(
                overall_status="amber",  # lowercase!
                clinical_accuracy=ClinicalAccuracy(score=75, issues=[]),
                nice_compliance=NICECompliance(compliant=False, gaps=[]),
                data_completeness=DataCompleteness(percentage=75, missing_fields=[]),
                safety_checks=SafetyChecks(passed=False, flags=[]),
                recommendations=["Manual review recommended"]
            )