# AETHER Clinical Assessment Agents

AETHER is a multi-agent clinical assessment module for processing GP referral letters and producing structured dementia assessment outputs. It uses CrewAI agents, Pydantic schemas, local LLM configuration through Ollama, and a Pinecone-backed RAG tool for NICE NG97 dementia guidance.

The project is designed around a clinical workflow: extract patient and referral details, structure the clinical history, identify risks and cognitive indicators, plan an evidence-aligned assessment battery, write a clinical brief, and run quality assurance checks.

> This software is a clinical decision-support prototype. It does not replace qualified clinical judgement, local governance, or patient-specific review.

## What It Does

- Extracts structured demographics and referral information from GP referral text.
- Builds a clinical history with conditions, medications, allergies, previous assessments, and timeline events.
- Profiles clinical, medication, cognitive, social, and safety risks.
- Designs a NICE NG97-aligned dementia assessment battery using validated instruments such as MMSE, MoCA, ACE-III, ADL, IADL, CDR, GDS, and NPI.
- Generates a concise clinical brief for downstream review.
- Validates the final output for clinical accuracy, data completeness, safety, and NICE compliance.

## Agent Workflow

The orchestrator runs the following agents in sequence:

1. `IntakeAgent` extracts core patient demographics and referral metadata.
2. `ClinicalHistoryAgent` structures the patient history from the referral text.
3. `ProfilerAgent` identifies risks, cognitive indicators, complexity, and information gaps.
4. `AssessmentPlannerAgent` designs a NICE NG97-informed assessment plan.
5. `BriefWriterAgent` synthesizes the output into a clinical brief.
6. `QAAgent` checks the assessment plan and brief for safety, completeness, and guideline alignment.

The main orchestration entry point is `src/aether/orchestrator/crew.py`.

## Project Structure

```text
src/aether/
  agents/          CrewAI specialist agents
  config/          Settings and LLM configuration
  orchestrator/    End-to-end workflow coordination and context store
  schemas/         Pydantic clinical data models
  tools/           NICE NG97 RAG and Pinecone utilities
  utils/           Logging helpers
  main.py          Example execution script

scripts/
  create_index.py              Create the Pinecone index
  ingest_nice_guidelines.py    Load NICE guidance content
  ingest_sample_data.py        Load sample RAG data
  test_*                       Agent and integration test scripts
```

## Requirements

- Python 3.11+
- Ollama running locally
- A local Ollama model, default: `llama3.2:3b`
- Pinecone API key
- NICE NG97 guideline data ingested into the configured Pinecone index

Dependencies are listed in both `pyproject.toml` and `requirements.txt`.

## Environment Variables

Create a `.env` file in the project root:

```env
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434

PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=nice-ng97-guidelines
PINECONE_ENVIRONMENT=us-east-1

LOG_LEVEL=INFO
HUGGINGFACE_API_KEY=
```

`PINECONE_API_KEY` is required by the current settings loader. Ollama does not require an API key.

## Setup

Install dependencies with Poetry:

```bash
poetry install
```

Or with pip:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Start Ollama and make sure the configured model is available:

```bash
ollama pull llama3.2:3b
ollama serve
```

Create and populate the Pinecone index:

```bash
python scripts/create_index.py
python scripts/ingest_nice_guidelines.py
```

For sample data ingestion, use:

```bash
python scripts/ingest_sample_data.py
```

## Running The Example

Run the built-in demonstration referral:

```bash
python src/aether/main.py
```

The script creates a sample `ReferralInput`, runs the full AETHER workflow, and prints the generated clinical brief and QA result as JSON.

## Programmatic Usage

```python
from aether.orchestrator.crew import AetherCrew
from aether.schemas.clinical import ReferralInput

referral = ReferralInput(
    referral_text="Raw GP referral letter text goes here",
    nhs_number="943 476 5919",
)

crew = AetherCrew()
result = crew.execute(referral)

print(result.clinical_brief.model_dump_json(indent=2))
print(result.qa_result.model_dump_json(indent=2))
```

For UI or API integrations, use `execute_with_streaming()` to receive progress callbacks for each workflow stage.

## Testing And Utilities

The `scripts/` directory contains focused scripts for checking individual agents and infrastructure:

```bash
python scripts/test_intake_agent.py
python scripts/test_clinical_history_agent.py
python scripts/test_profiler_agent.py
python scripts/test_planner_agent.py
python scripts/test_brief_writer_agent.py
python scripts/test_qa_agent.py
python scripts/test_rag.py
```

Pinecone utility scripts are also available:

```bash
python scripts/list_indexes.py
python scripts/get_index_details.py
python scripts/find_region.py
```

## Core Outputs

The pipeline returns an `OrchestratorOutput` containing:

- `patient_data`
- `clinical_history`
- `patient_profile`
- `assessment_plan`
- `clinical_brief`
- `qa_result`
- `metadata`

All major outputs are validated with Pydantic models in `src/aether/schemas/clinical.py`.

## Clinical Safety Notes

- Agent outputs should be reviewed by an appropriately qualified clinician before use.
- The system is intended to support dementia assessment planning, not to diagnose.
- The RAG layer depends on the quality and currency of ingested NICE NG97 guidance.
- The agents are instructed not to invent missing clinical data, but all extracted information should still be checked against the original referral.
- Patient-identifiable data must be handled according to local information governance and privacy requirements.
