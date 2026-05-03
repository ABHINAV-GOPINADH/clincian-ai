# aether/agents/planner.py - FINAL CORRECTED VERSION
"""Neuropsychological Assessment Planner Agent - LangGraph with RAG."""
from typing import Optional, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
import json
import re

from aether.config.settings import settings
from aether.schemas.clinical import PatientData, PatientProfile, AssessmentPlan
from aether.tools.rag_tool import nice_rag
from aether.utils.logger import logger


class AssessmentPlannerAgent:
    """Neuropsychological Assessment Planner with RAG."""
    
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
            ("system", "You are a Clinical Psychologist designing NICE NG97-compliant assessment batteries."),
            ("human", """Design assessment plan for:

PATIENT: {patient_name}, {age}yo
RISK FLAGS: {risk_flags_text}
COGNITIVE CONCERNS: {cognitive_concerns_text}

NICE NG97 GUIDANCE:
{nice_guidance}

AVAILABLE INSTRUMENTS (use EXACT names):
- MMSE (Mini-Mental State Examination)
- MoCA (Montreal Cognitive Assessment)
- ACE-III (Addenbrooke's Cognitive Examination)
- ADAS-Cog (Alzheimer's Disease Assessment Scale)
- ADL (Activities of Daily Living)
- IADL (Instrumental Activities of Daily Living)
- CDR (Clinical Dementia Rating)
- GDS (Geriatric Depression Scale)
- NPI (Neuropsychiatric Inventory)

Return JSON:
{{
  "instruments": [
    {{
      "name": "Full instrument name",
      "type": "MMSE/MoCA/ACE-III/ADAS-Cog/ADL/IADL/CDR/GDS/NPI",
      "rationale": "why selected",
      "priority": "essential/recommended/optional",
      "estimated_duration": 15
    }}
  ],
  "special_considerations": ["consideration 1"],
  "contraindications": ["any contraindications"],
  "nice_compliance_notes": "compliance statement"
}}

CRITICAL: 
- type must be EXACT: MMSE, MoCA, ACE-III, ADAS-Cog, ADL, IADL, CDR, GDS, or NPI
- estimated_duration must be INTEGER (minutes only)""")
        ])
    
    def _retrieve_nice_guidance(self, patient_profile: PatientProfile) -> str:
        """Retrieve NICE NG97 guidance using RAG."""
        print("\n🔍 RETRIEVING NICE NG97 GUIDANCE...")
        
        try:
            query = "NICE NG97 dementia assessment instruments cognitive testing recommendations"
            docs = nice_rag.retrieve_guidance(query, top_k=5)
            
            if docs:
                guidance = "\n\n---\n\n".join([doc.page_content for doc in docs])
                print(f"  ✅ Retrieved {len(docs)} excerpts")
                return guidance
            else:
                return self._get_fallback_guidance()
        except Exception as e:
            print(f"  ❌ RAG error: {e}")
            return self._get_fallback_guidance()
    
    def _get_fallback_guidance(self) -> str:
        return """NICE NG97: Use ACE-III, MoCA, or MMSE for cognitive assessment."""
    
    def _extract_duration(self, duration_str: str) -> int:
        """Extract integer minutes from duration string."""
        try:
            # Extract first number from string like "10-15 minutes"
            nums = re.findall(r'\d+', str(duration_str))
            if nums:
                return int(nums[0])
            return 15  # Default
        except:
            return 15
    
    def _map_to_valid_type(self, inst_type: str, inst_name: str) -> str:
        """Map instrument type to valid enum value."""
        # Valid types: ADAS-Cog, MMSE, MoCA, ACE-III, ADL, IADL, CDR, GDS, NPI
        
        inst_type_upper = str(inst_type).upper()
        inst_name_upper = str(inst_name).upper()
        
        # Try exact match first
        valid_types = ["ADAS-Cog", "MMSE", "MoCA", "ACE-III", "ADL", "IADL", "CDR", "GDS", "NPI"]
        for vtype in valid_types:
            if vtype.upper() == inst_type_upper:
                return vtype
        
        # Try to infer from name
        if "MMSE" in inst_name_upper or "MINI-MENTAL" in inst_name_upper:
            return "MMSE"
        elif "MOCA" in inst_name_upper or "MONTREAL" in inst_name_upper:
            return "MoCA"
        elif "ACE" in inst_name_upper or "ADDENBROOKE" in inst_name_upper:
            return "ACE-III"
        elif "ADAS" in inst_name_upper:
            return "ADAS-Cog"
        elif "ADL" in inst_name_upper and "INSTRUMENTAL" not in inst_name_upper:
            return "ADL"
        elif "IADL" in inst_name_upper or "INSTRUMENTAL" in inst_name_upper:
            return "IADL"
        elif "CDR" in inst_name_upper or "DEMENTIA RATING" in inst_name_upper:
            return "CDR"
        elif "GDS" in inst_name_upper or "DEPRESSION" in inst_name_upper:
            return "GDS"
        elif "NPI" in inst_name_upper or "NEUROPSYCHIATRIC" in inst_name_upper:
            return "NPI"
        
        # Default fallback
        return "MMSE"
    
    def execute(self, patient_data: PatientData, patient_profile: PatientProfile) -> AssessmentPlan:
        print("\n" + "="*80)
        print("📋 ASSESSMENT PLANNER")
        print("="*80)
        logger.info("AssessmentPlannerAgent: Starting")
        
        try:
            # Prepare text
            risk_flags_text = "None"
            if hasattr(patient_profile, 'risk_flags') and patient_profile.risk_flags:
                risk_flags_text = "\n".join([f"- {rf.description}" for rf in patient_profile.risk_flags])
            
            cognitive_concerns_text = "None"
            if hasattr(patient_profile, 'cognitive_indicators') and patient_profile.cognitive_indicators:
                cognitive_concerns_text = "\n".join([f"- {ci.concern}" for ci in patient_profile.cognitive_indicators])
            
            # Get RAG guidance
            nice_guidance = self._retrieve_nice_guidance(patient_profile)
            
            # Call LLM
            print("\n⏳ Calling LLM...")
            raw = self.chain.invoke({
                "patient_name": f"{patient_data.name.first} {patient_data.name.last}",
                "age": str(patient_data.age or "Unknown"),
                "risk_flags_text": risk_flags_text,
                "cognitive_concerns_text": cognitive_concerns_text,
                "nice_guidance": nice_guidance
            })
            print("✅ LLM responded\n")
            
            print("🔍 RAW:", json.dumps(raw, indent=2), "\n")
            
            # Normalize instruments
            instruments = []
            for inst in (raw.get("instruments") or []):
                # Map priority
                priority = str(inst.get("priority", "recommended")).lower()
                if priority not in ["essential", "recommended", "optional"]:
                    if priority in ["high", "critical"]:
                        priority = "essential"
                    elif priority in ["low"]:
                        priority = "optional"
                    else:
                        priority = "recommended"
                
                # Map type to valid enum
                inst_type = self._map_to_valid_type(
                    inst.get("type", ""),
                    inst.get("name", "")
                )
                
                # Extract duration as integer
                duration = self._extract_duration(inst.get("estimated_duration", 15))
                
                instruments.append({
                    "name": str(inst.get("name", "Unknown")),
                    "type": inst_type,
                    "rationale": str(inst.get("rationale", "Standard assessment")),
                    "priority": priority,
                    "estimated_duration": duration
                })
            
            print(f"🎯 INSTRUMENTS: {len(instruments)}")
            for i, inst in enumerate(instruments):
                print(f"  {i+1}. [{inst['priority'].upper()}] {inst['name']} ({inst['type']}) - {inst['estimated_duration']}min")
            
            # Build priority order
            essential = [i["name"] for i in instruments if i["priority"] == "essential"]
            recommended = [i["name"] for i in instruments if i["priority"] == "recommended"]
            optional = [i["name"] for i in instruments if i["priority"] == "optional"]
            priority_order = essential + recommended + optional
            
            # Calculate total duration as integer
            total_estimated_duration = sum(inst["estimated_duration"] for inst in instruments)
            
            # Build normalized plan
            normalized = {
                "instruments": instruments,
                "total_estimated_duration": total_estimated_duration,
                "priority_order": priority_order,
                "special_considerations": raw.get("special_considerations") or ["Standard NICE NG97 protocol"],
                "contraindications": raw.get("contraindications") or [],
                "nice_compliance_notes": raw.get("nice_compliance_notes") or "Assessment plan per NICE NG97"
            }
            
            print(f"\n⏱️  TOTAL: {total_estimated_duration} minutes")
            print(f"📋 SPECIAL CONSIDERATIONS: {normalized['special_considerations']}")
            print()
            
            # Validate
            print("🔍 Validating...")
            plan = AssessmentPlan.model_validate(normalized)
            print("✅ VALIDATION PASSED\n")
            
            print("="*80 + "\n")
            
            logger.info(f"Success! {len(plan.instruments)} instruments, {total_estimated_duration}min")
            
            return plan
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}\n")
            logger.error(f"Planning failed: {e}")
            
            # Minimal fallback
            return AssessmentPlan(
                instruments=[
                    {
                        "name": "Mini-Mental State Examination",
                        "type": "MMSE",
                        "rationale": "Standard cognitive screening",
                        "priority": "essential",
                        "estimated_duration": 10
                    }
                ],
                total_estimated_duration=10,
                priority_order=["Mini-Mental State Examination"],
                special_considerations=["Minimal assessment plan (fallback)"],
                contraindications=[],
                nice_compliance_notes="Fallback assessment plan"
            )