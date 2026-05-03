# aether/agents/profiler.py
"""Clinical Risk Profiler Agent - Complete schema compliance."""
from typing import Optional, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
import json

from aether.config.settings import settings
from aether.schemas.clinical import (
    PatientData, ClinicalHistory, PatientProfile,
    RiskFlag, CognitiveIndicator, ComplexitySummary
)
from aether.utils.logger import logger


class ProfilerAgent:
    """Clinical Risk Profiler - extracts risks and cognitive indicators."""
    
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
            ("system", "Extract ONLY explicitly stated risks from clinical data."),
            ("human", """Extract risk profile:

PATIENT: {patient_name}, {age}yo {gender}
CONDITIONS: {conditions_text}
MEDICATIONS: {medications_text}

Return JSON:
{{"risk_flags": ["risk 1"], "cognitive_indicators": ["indicator 1"]}}""")
        ])
    
    def execute(self, patient_data: PatientData, clinical_history: ClinicalHistory) -> PatientProfile:
        print("\n" + "="*80)
        print("👤 PROFILER AGENT")
        print("="*80)
        logger.info("ProfilerAgent: Starting")
        
        try:
            # Prepare text
            conditions_text = "\n".join([f"- {c.display}" for c in clinical_history.conditions]) or "None"
            medications_text = "\n".join([f"- {m.name}" for m in clinical_history.medications]) or "None"
            
            # Get LLM response
            raw = self.chain.invoke({
                "patient_name": f"{patient_data.name.first} {patient_data.name.last}",
                "age": str(patient_data.age or "Unknown"),
                "gender": str(patient_data.gender or "Unknown"),
                "conditions_text": conditions_text,
                "medications_text": medications_text
            })
            
            print("🔍 RAW:", json.dumps(raw, indent=2), "\n")
            
            # Build RiskFlag objects
            risk_flags = []
            for risk_text in (raw.get("risk_flags") or []):
                if not risk_text:
                    continue
                
                risk_lower = risk_text.lower()
                
                # Map to valid enum category
                if any(w in risk_lower for w in ["fall", "mobility", "balance"]):
                    category = "safety"
                elif any(w in risk_lower for w in ["medication", "drug"]):
                    category = "medication"
                elif any(w in risk_lower for w in ["memory", "cognitive", "dementia"]):
                    category = "cognitive"
                elif any(w in risk_lower for w in ["social", "isolation"]):
                    category = "social"
                else:
                    category = "clinical"
                
                risk_flags.append(RiskFlag(
                    category=category,
                    description=risk_text,
                    severity="medium",  # Default
                    source="clinical_data",
                    reasoning=f"Identified from clinical assessment: {risk_text}"
                ))
            
            print(f"🚩 RISKS: {len(risk_flags)}")
            
            # Build CognitiveIndicator objects
            cognitive_indicators = []
            for cog_text in (raw.get("cognitive_indicators") or []):
                if not cog_text:
                    continue
                
                cog_lower = cog_text.lower()
                
                # Map to valid enum domain (CANNOT be 'general')
                if any(w in cog_lower for w in ["memory", "recall", "forget"]):
                    domain = "memory"
                elif any(w in cog_lower for w in ["attention", "concentration"]):
                    domain = "attention"
                elif any(w in cog_lower for w in ["language", "speech", "word"]):
                    domain = "language"
                elif any(w in cog_lower for w in ["executive", "planning", "organization"]):
                    domain = "executive"
                elif any(w in cog_lower for w in ["visuospatial", "visual", "spatial"]):
                    domain = "visuospatial"
                else:
                    # Default to memory if can't categorize
                    domain = "memory"
                
                cognitive_indicators.append(CognitiveIndicator(
                    domain=domain,
                    description=cog_text,
                    severity="mild",  # Default
                    observed_by="referrer",
                    concern=cog_text,  # Required field
                    evidence_source="referral_letter"  # Required field
                ))
            
            print(f"🧠 COGNITIVE: {len(cognitive_indicators)}\n")
            
            # Calculate complexity
            complexity_score = min(10, (len(clinical_history.conditions) + len(clinical_history.medications)) // 2 + 1)
            
            # Build factors list for ComplexitySummary
            factors = []
            if clinical_history.conditions:
                factors.append(f"{len(clinical_history.conditions)} active conditions")
            if clinical_history.medications:
                factors.append(f"{len(clinical_history.medications)} medications")
            if risk_flags:
                factors.append(f"{len(risk_flags)} risk flags")
            if cognitive_indicators:
                factors.append(f"{len(cognitive_indicators)} cognitive concerns")
            
            if not factors:
                factors = ["Minimal clinical complexity"]
            
            # Build ComplexitySummary with all required fields
            complexity_summary = ComplexitySummary(
                score=complexity_score,
                rationale=", ".join(factors),
                primary_concerns=[rf.description for rf in risk_flags[:3]],
                factors=factors  # Required field!
            )
            
            print(f"📊 COMPLEXITY: {complexity_score}/10")
            print(f"Factors: {factors}\n")
            
            # Build final profile
            profile = PatientProfile(
                risk_flags=risk_flags,
                cognitive_indicators=cognitive_indicators,
                functional_status=raw.get("functional_status"),
                social_context=raw.get("social_context"),
                complexity_score=complexity_score,
                complexity_summary=complexity_summary,
                information_gaps=raw.get("information_gaps") or [],
                recommended_assessments=raw.get("recommended_assessments") or []
            )
            
            print("✅ SUCCESS\n" + "="*80 + "\n")
            
            return profile
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}\n")
            logger.error(f"Profiling failed: {e}")
            
            # Return minimal valid profile
            complexity_score = min(10, (len(clinical_history.conditions) + len(clinical_history.medications)) // 2 + 1)
            
            return PatientProfile(
                risk_flags=[],
                cognitive_indicators=[],
                functional_status=None,
                social_context=None,
                complexity_score=complexity_score,
                complexity_summary=ComplexitySummary(
                    score=complexity_score,
                    rationale="Assessment incomplete",
                    primary_concerns=[],
                    factors=["Assessment failed"]
                ),
                information_gaps=["Profiling incomplete"],
                recommended_assessments=[]
            )