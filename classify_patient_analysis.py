#!/usr/bin/env python3

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate


VALID_TASK_TYPES = {"binary", "multiclass"}
VALID_BACKENDS = {"ollama", "huggingface", "huggingface_endpoint"}


# ---------------------------------------------------------------------------
# LLM backend factory
# ---------------------------------------------------------------------------

def create_llm(backend, model_name, max_new_tokens=2048):
    """Create a runnable LLM for the requested backend."""

    if backend == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_name,
            temperature=0,
            format="json",
        )

    if backend == "huggingface":
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            pipeline,
        )
        from langchain_huggingface import HuggingFacePipeline

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        quant_config = None
        if torch.cuda.is_available():
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            quantization_config=quant_config,
            trust_remote_code=True,
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            return_full_text=False,
        )

        return HuggingFacePipeline(pipeline=pipe)

    if backend == "huggingface_endpoint":
        from langchain_huggingface import HuggingFaceEndpoint

        return HuggingFaceEndpoint(
            repo_id=model_name,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
        )

    raise ValueError(f"Unknown backend: {backend}")


# ---------------------------------------------------------------------------
# Robust JSON parsing
# ---------------------------------------------------------------------------

def extract_json_object(text):
    """Extract the first balanced JSON object from ``text``."""

    if not isinstance(text, str):
        text = str(text)

    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response.")

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]

        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:index + 1]
                return json.loads(candidate)

    raise ValueError("Unbalanced JSON object in model response.")


def get_response_text(response):
    """Return a plain string from a LangChain LLM response."""

    content = getattr(response, "content", response)

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            else:
                parts.append(str(part))
        content = "".join(parts)

    return str(content)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Task / input loading
# ---------------------------------------------------------------------------

def load_tasks(path):
    tasks = load_json(path)

    if not isinstance(tasks, list) or not tasks:
        raise ValueError(
            "Classification configuration must be a non-empty JSON list."
        )

    seen_ids = set()

    for task in tasks:
        for key in ("id", "task_type", "classes", "instruction"):
            if key not in task:
                raise ValueError(
                    f"Classification task is missing '{key}': {task}"
                )

        if task["id"] in seen_ids:
            raise ValueError(
                f"Duplicate classification task ID: {task['id']}"
            )

        seen_ids.add(task["id"])

        if task["task_type"] not in VALID_TASK_TYPES:
            raise ValueError(
                f"Task '{task['id']}' has invalid task_type. "
                f"Use one of {sorted(VALID_TASK_TYPES)}."
            )

        classes = task["classes"]

        if not isinstance(classes, list) or len(classes) < 2:
            raise ValueError(
                f"Task '{task['id']}' must define at least two classes."
            )

        if len(set(classes)) != len(classes):
            raise ValueError(
                f"Task '{task['id']}' contains duplicate classes."
            )

        if task["task_type"] == "binary" and len(classes) != 2:
            raise ValueError(
                f"Binary task '{task['id']}' must have exactly two classes."
            )

        task.setdefault("allow_abstain", True)

    return tasks


def find_analysis_records(data):
    """Extract Stage 1 analysis records from common output structures."""

    records = []

    record_id_keys = ("id", "analysis_id", "question_id")
    answer_keys = ("answer", "result", "response", "value")

    def is_record(obj):
        return (
            isinstance(obj, dict)
            and any(k in obj for k in record_id_keys)
            and any(k in obj for k in answer_keys)
        )

    def record_id(obj, fallback=None):
        for key in record_id_keys:
            if obj.get(key) is not None:
                return str(obj[key])
        return fallback

    def get_answer(obj):
        for key in answer_keys:
            if key in obj:
                return obj[key]
        return None

    def visit(obj, parent_key=None):
        if isinstance(obj, list):
            for item in obj:
                visit(item, parent_key)
            return

        if not isinstance(obj, dict):
            return

        if is_record(obj):
            records.append({
                "id": record_id(obj, parent_key),
                "answer": get_answer(obj),
                "answer_type": obj.get("answer_type"),
                "evidence": obj.get("evidence", []),
            })
            return

        for key, value in obj.items():
            if isinstance(value, dict) and any(
                answer_key in value for answer_key in answer_keys
            ):
                records.append({
                    "id": record_id(value, str(key)),
                    "answer": get_answer(value),
                    "answer_type": value.get("answer_type"),
                    "evidence": value.get("evidence", []),
                })
            else:
                visit(value, str(key))

    visit(data)

    unique = {}
    for record in records:
        unique[record["id"]] = record

    return list(unique.values())


def prepare_stage1_input(path):
    data = load_json(path)

    if not isinstance(data, dict):
        raise ValueError(
            "Each Stage 1 patient file must contain a JSON object."
        )

    patient_id = str(data.get("patient_id", path.stem))
    analyses = find_analysis_records(data)

    if not analyses:
        raise ValueError(
            f"No Stage 1 analysis records found in {path}. "
            "Expected analysis records with an ID and an answer."
        )

    return patient_id, analyses


# ---------------------------------------------------------------------------
# Classifier chain
# ---------------------------------------------------------------------------

CLASSIFIER_SYSTEM_PROMPT = """
You are performing a research classification based on structured
analysis of a patient's interview.

Use ONLY the Stage 1 analysis records supplied in this request.
Do not invent symptoms, medical history, test results, or clinical labels.
Do not infer that a symptom is absent because it was not discussed.
Distinguish subjective patient reports from objective cognitive testing.
Do not diagnose a patient.

Return ONLY a valid JSON object. No markdown, no code fences, no prose
outside the JSON. Structure:

{
  "predicted_class": "one of the permitted classes, or insufficient_evidence",
  "class_scores": {
    "class_name": 0
  },
  "confidence": 0,
  "rationale": "brief explanation",
  "supporting_analysis_ids": ["analysis_id"],
  "limitations": ["limitation"]
}

Requirements:
- Include every permitted class in class_scores.
- Give each class a relative score between 0 and 100.
- Scores are subjective model estimates, NOT calibrated probabilities.
- confidence is your subjective confidence in this classification from 0 to 100.
  It is not a clinical or statistically calibrated confidence measure.
- supporting_analysis_ids must contain only IDs present in the supplied records.
- If evidence is inadequate, use insufficient_evidence when abstention is allowed.
- When abstention is not allowed, select the best-supported class but explain
  the uncertainty in limitations.
- Do not use the patient's own stated confidence as your classification confidence.
""".strip()


CLASSIFIER_HUMAN_PROMPT = """
Classification task:
{task_json}

Permitted classes:
{classes_json}

Abstention allowed:
{allow_abstain}

Stage 1 analysis records:
{analysis_json}

Classify this patient for this task only.
""".strip()


def build_classifier(backend, model_name):
    llm = create_llm(backend, model_name)

    prompt = ChatPromptTemplate.from_messages([
        ("system", CLASSIFIER_SYSTEM_PROMPT),
        ("human", CLASSIFIER_HUMAN_PROMPT),
    ])

    return prompt | llm


# ---------------------------------------------------------------------------
# Score normalisation
# ---------------------------------------------------------------------------

def normalise_scores(raw_scores, classes):
    """Convert non-negative class scores to a relative 0-100 distribution."""

    scores = {}

    for class_name in classes:
        try:
            value = float(raw_scores.get(class_name, 0))
        except (TypeError, ValueError):
            value = 0.0

        scores[class_name] = max(0.0, min(100.0, value))

    total = sum(scores.values())

    if total <= 0:
        return {class_name: None for class_name in classes}

    return {
        class_name: round(value / total * 100, 2)
        for class_name, value in scores.items()
    }


# ---------------------------------------------------------------------------
# Classification per task
# ---------------------------------------------------------------------------

def classify_one_task(chain, task, analyses):
    allowed_classes = task["classes"]
    allow_abstain = bool(task.get("allow_abstain", True))

    task_payload = {
        "id": task["id"],
        "task_type": task["task_type"],
        "instruction": task["instruction"],
    }

    response = chain.invoke({
        "task_json": json.dumps(task_payload, ensure_ascii=False),
        "classes_json": json.dumps(allowed_classes, ensure_ascii=False),
        "allow_abstain": str(allow_abstain),
        "analysis_json": json.dumps(analyses, ensure_ascii=False),
    })

    raw_text = get_response_text(response)
    result = extract_json_object(raw_text)

    predicted = str(result.get("predicted_class", "")).strip()

    if predicted not in allowed_classes:
        if allow_abstain and predicted == "insufficient_evidence":
            pass
        else:
            raise ValueError(
                f"Model returned invalid class '{predicted}' "
                f"for task '{task['id']}'."
            )

    try:
        confidence = float(result.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(100.0, confidence))

    valid_ids = {item["id"] for item in analyses}
    supporting_ids = result.get("supporting_analysis_ids", [])

    if not isinstance(supporting_ids, list):
        supporting_ids = []

    supporting_ids = [
        str(item) for item in supporting_ids if str(item) in valid_ids
    ]

    raw_scores = result.get("class_scores", {})
    if not isinstance(raw_scores, dict):
        raw_scores = {}

    scores = normalise_scores(raw_scores, allowed_classes)

    return {
        "task_id": task["id"],
        "task_type": task["task_type"],
        "predicted_class": predicted,
        "class_scores_percent_uncalibrated": scores,
        "model_confidence_percent_uncalibrated": round(confidence, 2),
        "rationale": str(result.get("rationale", "")),
        "supporting_analysis_ids": supporting_ids,
        "limitations": result.get("limitations", []),
    }


def process_patient(path, tasks, chain):
    patient_id, analyses = prepare_stage1_input(path)

    results = []

    for task in tasks:
        result = classify_one_task(chain, task, analyses)
        results.append(result)

    return {
        "patient_id": patient_id,
        "source_file": path.name,
        "classification_method": "local LLM using Stage 1 analysis",
        "calibration_note": (
            "Class scores and model confidence are uncalibrated model "
            "estimates; they are not validated clinical probabilities."
        ),
        "classifications": results,
    }


# ---------------------------------------------------------------------------
# Combined CSV
# ---------------------------------------------------------------------------

def write_combined_csv(path, patient_results):
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "patient_id",
        "source_file",
        "task_id",
        "task_type",
        "predicted_class",
        "class_scores_percent_uncalibrated",
        "model_confidence_percent_uncalibrated",
        "rationale",
        "supporting_analysis_ids",
        "limitations",
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for patient in patient_results:
            for result in patient["classifications"]:
                writer.writerow({
                    "patient_id": patient["patient_id"],
                    "source_file": patient["source_file"],
                    "task_id": result["task_id"],
                    "task_type": result["task_type"],
                    "predicted_class": result["predicted_class"],
                    "class_scores_percent_uncalibrated": json.dumps(
                        result["class_scores_percent_uncalibrated"],
                        ensure_ascii=False,
                    ),
                    "model_confidence_percent_uncalibrated":
                        result["model_confidence_percent_uncalibrated"],
                    "rationale": result["rationale"],
                    "supporting_analysis_ids": json.dumps(
                        result["supporting_analysis_ids"],
                        ensure_ascii=False,
                    ),
                    "limitations": json.dumps(
                        result["limitations"],
                        ensure_ascii=False,
                    ),
                })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Classify patients using Stage 1 Q&A analysis JSON files."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing Stage 1 patient JSON files.",
    )
    parser.add_argument(
        "--tasks",
        required=True,
        help="Path to classification_tasks.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="./classification_results",
        help="Output directory (default: ./classification_results).",
    )
    parser.add_argument(
        "--model",
        default="llama3.1:8b",
        help="Model name or HF repo ID (default: llama3.1:8b).",
    )
    parser.add_argument(
        "--backend",
        choices=sorted(VALID_BACKENDS),
        default="ollama",
        help=(
            "LLM backend: 'ollama' (local GGUF), "
            "'huggingface' (local transformers), or "
            "'huggingface_endpoint' (HF Inference API). "
            "Default: ollama."
        ),
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Input filename pattern (default: *.json).",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")

    try:
        tasks = load_tasks(Path(args.tasks))
    except Exception as exc:
        parser.error(f"Cannot load classification tasks: {exc}")

    files = sorted(input_dir.glob(args.pattern))

    if not files:
        parser.error(
            f"No JSON files matching '{args.pattern}' found in {input_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    individual_dir = output_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input files: {len(files)}")
    print(f"Classification tasks: {len(tasks)}")
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model}")
    print(f"Output directory: {output_dir}\n")

    chain = build_classifier(args.backend, args.model)

    successful = []
    errors = []

    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] Processing {path.name}")

        try:
            result = process_patient(path, tasks, chain)

            output_path = individual_dir / f"{path.stem}_classification.json"
            save_json(output_path, result)

            successful.append(result)
            print(f"  Saved: {output_path}")

        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            errors.append({
                "file": path.name,
                "error": str(exc),
            })

    write_combined_csv(output_dir / "all_patients.csv", successful)

    save_json(output_dir / "errors.json", errors)

    print("\nFinished.")
    print(f"Input files:       {len(files)}")
    print(f"Successful files:  {len(successful)}")
    print(f"Failed files:      {len(errors)}")
    print(f"Results:           {output_dir.resolve()}")


if __name__ == "__main__":
    main()

'''
# Core (needed for all backends)
pip install -U langchain-core

# Ollama backend
pip install -U langchain-ollama
ollama pull llama3.1:8b

# Hugging Face local backend
pip install -U langchain-huggingface transformers accelerate bitsandbytes torch

# Hugging Face endpoint backend (also needs an HF token)
export HUGGINGFACEHUB_API_TOKEN=hf_...


Usage Examples
Ollama (as before):

bash
python analyse_patient_qa.py \
    --input-dir ./patients \
    --analysis-questions ./analysis_questions.json \
    --output-dir ./results/stage1_llama3.1 \
    --backend ollama \
    --model llama3.1:8b
Hugging Face local (transformers, 4-bit on GPU):

bash
python analyse_patient_qa.py \
    --input-dir ./patients \
    --analysis-questions ./analysis_questions.json \
    --output-dir ./results/stage1_qwen_hf \
    --backend huggingface \
    --model Qwen/Qwen2.5-7B-Instruct
Hugging Face endpoint (serverless):

bash
python analyse_patient_qa.py \
    --input-dir ./patients \
    --analysis-questions ./analysis_questions.json \
    --output-dir ./results/stage1_hf_endpoint \
    --backend huggingface_endpoint \
    --model meta-llama/Llama-3.1-8B-Instruct



Updated Comparison Loop
bash
# Ollama sweep (as before)
for model in llama3.1:8b qwen3:8b gemma3:4b mistral:7b deepseek-r1:8b
do
    safe_name=$(echo "$model" | tr ':/' '__')

    python analyse_patient_qa.py \
        --input-dir ./patients \
        --analysis-questions ./analysis_questions.json \
        --output-dir "./results/stage1_${safe_name}" \
        --backend ollama \
        --model "$model"

    python classify_patient_analysis.py \
        --input-dir "./results/stage1_${safe_name}/individual" \
        --tasks ./classification_tasks.json \
        --output-dir "./results/stage2_${safe_name}" \
        --backend ollama \
        --model "$model"
done

# Hugging Face sweep (parallel experiment arm)
for model in \
    meta-llama/Llama-3.1-8B-Instruct \
    Qwen/Qwen2.5-7B-Instruct \
    mistralai/Mistral-7B-Instruct-v0.3 \
    google/gemma-2-9b-it \
    microsoft/Phi-3.5-mini-instruct
do
    safe_name=$(echo "$model" | tr ':/' '__')

    python analyse_patient_qa.py \
        --input-dir ./patients \
        --analysis-questions ./analysis_questions.json \
        --output-dir "./results_hf/stage1_${safe_name}" \
        --backend huggingface \
        --model "$model"

    python classify_patient_analysis.py \
        --input-dir "./results_hf/stage1_${safe_name}/individual" \
        --tasks ./classification_tasks.json \
        --output-dir "./results_hf/stage2_${safe_name}" \
        --backend huggingface \
        --model "$model"
done


Recommended Models Table
Model	Ollama Tag	HF Repo ID	Params	Family	Notes
Llama 3.1 8B Instruct	llama3.1:8b	meta-llama/Llama-3.1-8B-Instruct	8B	Meta	Baseline
Llama 3.2 3B Instruct	llama3.2:3b	meta-llama/Llama-3.2-3B-Instruct	3B	Meta	Lightweight
Qwen 2.5 7B Instruct	qwen2.5:7b	Qwen/Qwen2.5-7B-Instruct	7B	Alibaba	Strong structured output
Qwen 2.5 3B Instruct	qwen2.5:3b	Qwen/Qwen2.5-3B-Instruct	3B	Alibaba	Small + structured
Qwen 3 8B	qwen3:8b	Qwen/Qwen3-8B	8B	Alibaba	Reasoning
Gemma 2 9B IT	gemma2:9b	google/gemma-2-9b-it	9B	Google	Strong instruction following
Gemma 3 4B IT	gemma3:4b	google/gemma-3-4b-it	4B	Google	Small modern
Mistral 7B Instruct v0.3	mistral:7b	mistralai/Mistral-7B-Instruct-v0.3	7B	Mistral	Independent family
DeepSeek R1 Distill Llama 8B	deepseek-r1:8b	deepseek-ai/DeepSeek-R1-Distill-Llama-8B	8B	DeepSeek	Reasoning distill
MedGemma 4B IT	medgemma:4b	google/medgemma-4b-it	4B	Google	Medical-domain
Phi-3.5 Mini Instruct	phi3.5	microsoft/Phi-3.5-mini-instruct	3.8B	Microsoft	Efficient
Phi-4 Mini Instruct	phi4-mini	microsoft/Phi-4-mini-instruct	~4B	Microsoft	Newer efficient
Granite 3.1 8B	granite3.1:8b	ibm-granite/granite-3.1-8b-instruct	8B	IBM	Enterprise
SmolLM3 3B	—	HuggingFaceTB/SmolLM3-3B	3B	HF	Long context
GPT-OSS 20B	gpt-oss:20b	—	20B	OpenAI OSS	Larger comparison (Ollama only)
Scientific Caveats for Your Comparison
Backend is a confound. GGUF (Ollama) vs SafeTensors (HF) vs API have different quantisation, tokenizers, and sampling defaults. Report the backend in every result row if you want a defensible comparison.

JSON reliability differs. Qwen and Mistral families are generally more JSON-compliant than small Gemma/Llama variants. The extract_json_object() fallback saves most cases, but track parse failure rate per model.

Quantisation matters. A 4-bit Ollama llama3.1:8b and an FP16 HF meta-llama/Llama-3.1-8B-Instruct are not the same model empirically. Either standardise quantisation or treat it as an explicit variable.

Endpoint rate limits. HF serverless endpoints throttle; wrap calls in retry logic (e.g. tenacity) if you go that route at scale.

HuggingFaceEndpoint skips your format="json" shortcut. All JSON enforcement falls to the prompt + parser. Expect a higher validation-failure rate than Ollama.

'''