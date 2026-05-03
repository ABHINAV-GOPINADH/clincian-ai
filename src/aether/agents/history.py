# aether/agents/history.py
"""Clinical History Agent - LangGraph version with complete field normalization."""
from typing import Optional, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
import json

from aether.config.settings import settings
from aether.schemas.clinical import PatientData, ClinicalHistory
from aether.utils.logger import logger


class ClinicalHistoryAgent:
    """Clinical Historian - extracts structured clinical history."""
    
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
        """Create extraction prompt."""
        
        return ChatPromptTemplate.from_messages([
            ("system", """You are a Clinical Informaticist. Extract clinical history data.

CRITICAL: Use these EXACT formats:
- status: "active", "resolved", or "chronic"
- significance: "low", "medium", or "high"
- type: actual test name (never null)"""),
            
            ("human", """Extract from these notes:

PATIENT: {patient_name}
NOTES: {clinical_notes}

Return JSON:
{{
  "conditions": [{{"code": null, "display": "name", "onset_date": null, "status": "active"}}],
  "medications": [{{"name": "drug", "dosage": "amount", "frequency": "how often", "start_date": null}}],
  "allergies": ["allergen name"],
  "past_assessments": [{{"type": "test name", "date": null, "outcome": "result"}}],
  "timeline_events": [{{"date": null, "event": "what happened", "significance": "low"}}]
}}""")
        ])
    
    def execute(self, patient_data: PatientData, clinical_notes: Optional[str] = None) -> ClinicalHistory:
        """Execute extraction with full normalization."""
        
        print("\n" + "="*80)
        print("📚 CLINICAL HISTORY AGENT")
        print("="*80)
        
        if not clinical_notes or not clinical_notes.strip():
            print("❌ No clinical notes - returning empty")
            return ClinicalHistory(conditions=[], medications=[], allergies=[], past_assessments=[], timeline_events=[])
        
        print(f"Patient: {patient_data.name.first} {patient_data.name.last}")
        print(f"Notes: {len(clinical_notes)} chars\n")
        
        try:
            # Get LLM response
            print("⏳ Calling LLM...")
            raw = self.chain.invoke({
                "patient_name": f"{patient_data.name.first} {patient_data.name.last}",
                "clinical_notes": clinical_notes
            })
            print("✅ LLM responded\n")
            
            print("🔍 RAW OUTPUT:")
            print(json.dumps(raw, indent=2))
            print()
            
            # NORMALIZE WITH DEFAULTS
            normalized = {
                "conditions": [
                    {
                        "code": c.get("code"),
                        "display": c.get("display") or c.get("name"),
                        "onset_date": c.get("onset_date"),
                        "status": (c.get("status") or "active").lower()  # FIX 1: Default status
                    }
                    for c in (raw.get("conditions") or [])
                ],
                "medications": [
                    {
                        "name": m.get("name"),
                        "dosage": m.get("dosage") or "Unknown",
                        "frequency": m.get("frequency") or "As directed",
                        "start_date": m.get("start_date")
                    }
                    for m in (raw.get("medications") or [])
                ],
                "allergies": [
                    a if isinstance(a, str) else a.get("allergen") or a.get("name")
                    for a in (raw.get("allergies") or [])
                    if a
                ],
                "past_assessments": [
                    {
                        "type": a.get("type") or a.get("test_name") or "Clinical assessment",  # FIX 2: Default type
                        "date": a.get("date"),
                        "outcome": a.get("outcome") or a.get("result") or "Not recorded"
                    }
                    for a in (raw.get("past_assessments") or [])
                ],
                "timeline_events": [
                    {
                        "date": e.get("date") or (f"{e['year']}-01-01" if e.get("year") else None),
                        "event": e.get("event") or e.get("description"),
                        "significance": self._map_significance(e.get("significance"))  # FIX 3: Map significance
                    }
                    for e in (raw.get("timeline_events") or [])
                ]
            }
            
            print("✅ NORMALIZED:")
            print(f"  Conditions: {len(normalized['conditions'])}")
            print(f"  Medications: {len(normalized['medications'])}")
            print(f"  Allergies: {len(normalized['allergies'])}")
            print(f"  Assessments: {len(normalized['past_assessments'])}")
            print(f"  Timeline: {len(normalized['timeline_events'])}")
            print()
            
            # Validate
            print("🔍 Validating...")
            history = ClinicalHistory.model_validate(normalized)
            print("✅ VALIDATION PASSED!\n")
            
            print("="*80)
            print("✅ SUCCESS")
            print("="*80 + "\n")
            
            return history
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}\n")
            logger.error(f"Extraction failed: {e}")
            return ClinicalHistory(conditions=[], medications=[], allergies=[], past_assessments=[], timeline_events=[])
    
    def _map_significance(self, value: Any) -> str:
        """Map any significance value to low/medium/high."""
        if not value:
            return "low"
        
        value_str = str(value).lower()
        
        # Map to high
        if any(word in value_str for word in ["high", "severe", "critical", "major", "important", "significant"]):
            return "high"
        
        # Map to medium
        if any(word in value_str for word in ["medium", "moderate"]):
            return "medium"
        
        # Default to low
        return "low"