# aether/agents/intake.py
"""
Clinical Intake Specialist Agent - Extracts demographic data from referrals.
Simplified with debugging and proper default handling.
"""
from typing import Optional, Dict, Any
from datetime import datetime, date
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
from pydantic import ValidationError
import json

from aether.config.settings import settings
from aether.schemas.clinical import ReferralInput, PatientData
from aether.utils.logger import logger


class IntakeAgent:
    """Clinical Intake Specialist Agent - Extracts ONLY demographic data."""
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        
        self.llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.0,
            format="json"
        )
        
        self.parser = JsonOutputParser()
        self.prompt = self._create_prompt()
        self.chain = self.prompt | self.llm | self.parser
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create extraction prompt."""
        
        system = """You are a Clinical Intake Data Extraction Specialist for the NHS.

Extract ONLY demographic data explicitly stated in the referral letter.

RULES:
1. Use null for missing data (NOT "Not provided" or "N/A")
2. Dates in YYYY-MM-DD format
3. Return ONLY valid JSON
4. Do NOT invent data"""

        user = """Extract patient demographics from this referral:

{referral_text}

Additional NHS Number: {nhs_number}

Return JSON with these EXACT field names:
{{
  "name": {{"first": "first name", "last": "last name", "middle": null, "title": null}},
  "date_of_birth": "YYYY-MM-DD",
  "age": number,
  "nhs_number": "10-digit number",
  "gender": "male/female/other/unknown",
  "contact_info": {{"address": "full address", "phone": "phone", "email": null}},
  "gp_details": {{"practice_name": "practice name", "gp_name": "gp name", "contact_number": "phone"}},
  "referral_date": "YYYY-MM-DD",
  "urgency": "routine/urgent/emergency",
  "referring_clinician_name": "clinician name",
  "referring_organization": "organization",
  "receiving_service": "service",
  "referral_reason": "reason for referral"
}}

CRITICAL: Use the EXACT field names shown above. Return ONLY the JSON."""

        return ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", user)
        ])
    
    def _normalize_json(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize LLM output and provide safe defaults with DEBUG prints."""
        
        print("\n" + "="*80)
        print("🔍 DEBUG: RAW LLM OUTPUT")
        print("="*80)
        print(json.dumps(json_data, indent=2))
        print("="*80 + "\n")
        
        logger.info(f"Raw JSON keys: {list(json_data.keys())}")
        
        # Extract name with all variations
        name_data = (
            json_data.get("name") or 
            json_data.get("PatientName") or 
            json_data.get("patient_name") or
            {}
        )
        
        # If name is a string like "Mary Higgins", split it
        if isinstance(name_data, str):
            parts = name_data.split()
            name_data = {
                "first": parts[0] if len(parts) > 0 else None,
                "last": parts[-1] if len(parts) > 1 else None,
                "middle": parts[1] if len(parts) > 2 else None,
                "title": None
            }
        
        print("📛 Name extraction:")
        print(f"  Raw: {json_data.get('name') or json_data.get('PatientName') or json_data.get('patient_name')}")
        print(f"  Normalized: {name_data}")
        
        # Extract gender with variations
        gender_raw = (
            json_data.get("gender") or 
            json_data.get("Gender") or 
            json_data.get("sex") or
            json_data.get("Sex")
        )
        
        if gender_raw and isinstance(gender_raw, str):
            gender = gender_raw.lower()
            # Map common variations
            if gender in ["m", "male"]:
                gender = "male"
            elif gender in ["f", "female"]:
                gender = "female"
        else:
            gender = "unknown"
        
        print(f"⚥ Gender extraction:")
        print(f"  Raw: {gender_raw}")
        print(f"  Normalized: {gender}")
        
        # Extract age
        age_raw = json_data.get("age") or json_data.get("Age")
        age = int(age_raw) if age_raw is not None else None
        
        print(f"📅 Age extraction:")
        print(f"  Raw: {age_raw}")
        print(f"  Normalized: {age}")
        
        # Extract address
        address_raw = (
            json_data.get("address") or
            json_data.get("Address")
        )
        
        # If address is a dict, convert to string
        if isinstance(address_raw, dict):
            address_parts = [
                address_raw.get("street"),
                address_raw.get("city"),
                address_raw.get("post_code")
            ]
            address = ", ".join([p for p in address_parts if p])
            if not address:
                address = None
        else:
            address = address_raw
        
        print(f"🏠 Address extraction:")
        print(f"  Raw: {address_raw}")
        print(f"  Normalized: {address}")
        
        # Build normalized structure
        normalized = {
            "name": {
                "first": name_data.get("first"),
                "last": name_data.get("last"),
                "middle": name_data.get("middle"),
                "title": name_data.get("title")
            },
            "date_of_birth": (
                json_data.get("date_of_birth") or 
                json_data.get("dob") or 
                json_data.get("DateOfBirth")
            ),
            "age": age,
            "nhs_number": (
                json_data.get("nhs_number") or 
                json_data.get("NHSNumber") or
                json_data.get("nhs_no")
            ),
            "gender": gender,
            "contact_info": {
                "address": address,
                "phone": (
                    json_data.get("phone") or 
                    json_data.get("telephone") or
                    json_data.get("contact_number")
                ),
                "email": json_data.get("email")
            },
            "gp_details": {
                "practice_name": (
                    json_data.get("gp_practice") or 
                    json_data.get("practice_name") or
                    "Unknown Practice"  # Default
                ),
                "gp_name": json_data.get("gp_name"),
                "contact_number": json_data.get("gp_phone") or json_data.get("gp_contact")
            },
            "referral_date": (
                json_data.get("referral_date") or 
                json_data.get("ReferralDate") or
                date.today().isoformat()  # Default to today
            ),
            "urgency": (
                json_data.get("urgency") or 
                json_data.get("Urgency") or
                "routine"  # Default
            ),
            "referring_clinician_name": (
                json_data.get("referring_clinician_name") or
                json_data.get("clinician_name") or
                json_data.get("gp_name")
            ),
            "referring_organization": (
                json_data.get("referring_organization") or
                json_data.get("organization") or
                json_data.get("gp_practice")
            ),
            "receiving_service": json_data.get("receiving_service"),
            "referral_reason": json_data.get("referral_reason")
        }
        
        print("\n" + "="*80)
        print("✅ DEBUG: NORMALIZED OUTPUT")
        print("="*80)
        print(json.dumps(normalized, indent=2, default=str))
        print("="*80 + "\n")
        
        logger.info(f"Normalized keys: {list(normalized.keys())}")
        
        return normalized
    
    def execute(self, referral_input: ReferralInput) -> PatientData:
        """Execute intake extraction with debugging."""
        
        print("\n" + "="*80)
        print("🚀 INTAKE AGENT STARTED")
        print("="*80)
        print(f"Referral text length: {len(referral_input.referral_text)} chars")
        print(f"NHS Number provided: {referral_input.nhs_number}")
        print("="*80 + "\n")
        
        logger.info("IntakeAgent: Starting extraction")
        
        # Simple security check
        if any(pattern in referral_input.referral_text.lower() 
               for pattern in ["ignore previous", "system:", "jailbreak"]):
            raise ValueError("Potential prompt injection detected")
        
        for attempt in range(self.max_retries):
            try:
                print(f"\n{'='*80}")
                print(f"🔄 ATTEMPT {attempt + 1}/{self.max_retries}")
                print(f"{'='*80}\n")
                
                logger.info(f"IntakeAgent: Attempt {attempt + 1}/{self.max_retries}")
                
                # Get JSON from LLM
                print("⏳ Calling LLM...")
                json_data = self.chain.invoke({
                    "referral_text": referral_input.referral_text,
                    "nhs_number": referral_input.nhs_number or "null"
                })
                print("✅ LLM response received\n")
                
                # Normalize and add defaults
                normalized = self._normalize_json(json_data)
                
                # Validate with Pydantic
                print("🔍 Validating with Pydantic...")
                patient_data = PatientData.model_validate(normalized)
                print("✅ Pydantic validation passed!\n")
                
                # Basic validation
                if patient_data.age and (patient_data.age < 0 or patient_data.age > 120):
                    raise ValueError(f"Invalid age: {patient_data.age}")
                
                print(f"\n{'='*80}")
                print("✅ SUCCESS!")
                print(f"{'='*80}")
                print(f"Patient: {patient_data.name.first} {patient_data.name.last}")
                print(f"NHS Number: {patient_data.nhs_number}")
                print(f"Age: {patient_data.age}")
                print(f"Gender: {patient_data.gender}")
                print(f"{'='*80}\n")
                
                logger.info(f"✅ IntakeAgent: Success! NHS#{patient_data.nhs_number}")
                return patient_data
                
            except ValidationError as e:
                print(f"\n❌ VALIDATION ERROR:")
                print(f"{'='*80}")
                print(e)
                print(f"{'='*80}\n")
                
                logger.warning(f"❌ Attempt {attempt + 1} validation error: {e}")
                if attempt == self.max_retries - 1:
                    raise
                    
            except Exception as e:
                print(f"\n❌ ERROR:")
                print(f"{'='*80}")
                print(f"Type: {type(e).__name__}")
                print(f"Message: {e}")
                print(f"{'='*80}\n")
                
                logger.warning(f"❌ Attempt {attempt + 1} error: {e}")
                if attempt == self.max_retries - 1:
                    raise
        
        raise RuntimeError("All extraction attempts failed")