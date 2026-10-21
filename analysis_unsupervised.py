#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from common import (
    log, bulk_load_patients, patient_text, CachedEmbedder,
)

# ---------- schema ----------
class EvidenceItem(BaseModel):
    qa_id: str
    field: Optional[str] = None
    quote: Optional[str] = None

class AnalysisResult(BaseModel):
    id: str
    answer: str
    assessment_confidence: float = Field(ge=0, le=100)
    evidence: list[EvidenceItem] = Field(default_factory=list)

class AnalysisBatch(BaseModel):
    results: list[AnalysisResult]

def _clean_results(results: list[dict]) -> list[dict]:
    for r in results:
        kept = []
        for ev in (r.get("evidence") or []):
            if ev.get("field") in {"question", "answer"} and ev.get("quote"):
                kept.append(ev)
        r["evidence"] = kept
    return results

def _fill_missing(results: list[dict], task_ids: list[str], pid: str) -> list[dict]:
    """Fill any missing task with Don't know, reorder to match task_ids."""
    got = {r["id"]: r for r in results if r.get("id") in task_ids}
    missing = [tid for tid in task_ids if tid not in got]
    if missing:
        log(f"  INCOMPLETE {pid}: missing {len(missing)} tasks: {missing}")
        for tid in missing:
            got[tid] = {
                "id": tid,
                "answer": "Don't know",
                "assessment_confidence": 0.0,
                "evidence": [],
            }
    return [got[tid] for tid in task_ids]

# ---------- prompts ----------
SYSTEM_PROMPT = """You analyse patient interview Q&A records.

For EVERY task you are given, return an object with exactly these fields:
- id: the task id as a string
- answer: the answer as a string. For yes_no_dont_know tasks use exactly Yes, No, or Don't know.
- assessment_confidence: a number from 0 to 100 indicating YOUR confidence in the answer
- evidence: a list of objects

Every evidence object MUST contain exactly three keys: qa_id, field, quote.
- field must be either question or answer
- quote must be an exact substring of the patient's question or answer
If you cannot supply all three keys for an item, do NOT emit that item at all.

Rules:
- You MUST return exactly one result per task, in the order given.
- Never omit a task, even if the answer is unclear — use Don't know instead.
- Do NOT leave evidence empty if any patient Q&A is relevant to the task.
- Return only valid JSON. No prose outside the JSON.
"""

HUMAN_PROMPT = """Patient ID: {patient_id}

Source Q&As:
{source_qas}

Analysis tasks ({n_tasks} total):
{tasks}

Return EXACTLY {n_tasks} results — one for every task above, in the same order.
Do not skip any task. If a task cannot be answered from the Q&As,
return answer "Don't know" with an empty evidence list.
"""

def build_chain(model):
    llm = ChatOllama(
        model=model,
        temperature=0,
        num_predict=1400,
        num_ctx=4096,
    )
    structured = llm.with_structured_output(AnalysisBatch)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])
    return prompt | structured

# ---------- cache ----------
def _is_cached_usable(cached: dict, recheck_empty: bool) -> bool:
    results = cached.get("results", [])
    if not results:
        return False
    if recheck_empty:
        all_empty = all(
            r.get("assessment_confidence", 0) == 0.0
            and not r.get("evidence")
            for r in results
        )
        if all_empty:
            return False
    return True

# ---------- clustering ----------
def run_kmeans(texts_by_pid, n_clusters, embedder):
    pids = list(texts_by_pid.keys())
    texts = [texts_by_pid[p] for p in pids]
    vecs = embedder.embed_many(texts)
    k = min(n_clusters, len(pids))
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(vecs)
    coords = (PCA(n_components=2, random_state=42).fit_transform(vecs)
              if len(pids) >= 2 else np.zeros((len(pids), 2)))
    return {
        "pids": pids,
        "cluster_labels": km.labels_.tolist(),
        "coords": coords.tolist(),
        "n_clusters": k,
    }

def cluster_overall(patient_answers, n_clusters, embedder):
    texts = {pid: " | ".join(
        f"{a['id']}: {a['answer']} ({a['assessment_confidence']:.0f})"
        for a in results)
        for pid, results in patient_answers.items()}
    return run_kmeans(texts, n_clusters, embedder)

def cluster_per_question(patient_answers, task_ids, n_clusters, embedder):
    per_task = {}
    for tid in task_ids:
        pid_text = {}
        for pid, results in patient_answers.items():
            for item in results:
                if item["id"] == tid:
                    pid_text[pid] = (f"{item['answer']} "
                                     f"(confidence {item['assessment_confidence']:.0f})")
                    break
        if len(pid_text) < 2:
            per_task[tid] = {"skipped": True,
                             "reason": "fewer than 2 patients have this task"}
        else:
            log(f"Clustering task {tid} over {len(pid_text)} patients")
            per_task[tid] = run_kmeans(pid_text, n_clusters, embedder)
    return per_task

# ---------- main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input-dir", required=True)
    p.add_argument("-q", "--analysis-questions", required=True)
    p.add_argument("-o", "--output-dir", required=True)
    p.add_argument("-m", "--model", default="llama3.2:3b")
    p.add_argument("--n-clusters", type=int, default=3)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--recheck-empty", action="store_true")
    args = p.parse_args()

    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qfile = Path(args.analysis_questions)

    tasks = json.loads(qfile.read_text())
    task_ids = [t["id"] for t in tasks]
    n_tasks = len(tasks)
    log(f"Tasks ({n_tasks}): {task_ids}")

    patients = bulk_load_patients(in_dir, qfile)
    if not patients:
        raise SystemExit("No patients found")

    chain = build_chain(args.model)
    embedder = CachedEmbedder("nomic-embed-text",
                              out_dir / "embeddings_cache.json")

    patient_answers = {}
    processed = skipped = failed = 0

    for i, (pid, qas) in enumerate(patients.items(), 1):
        if args.limit is not None and processed >= args.limit:
            log(f"Limit reached ({args.limit}). Stopping.")
            break

        out_file = out_dir / f"{pid}.json"

        if out_file.exists() and not args.force:
            try:
                cached = json.loads(out_file.read_text())
                if _is_cached_usable(cached, args.recheck_empty):
                    patient_answers[pid] = cached["results"]
                    skipped += 1
                    log(f"[{i}/{len(patients)}] cached: {pid}")
                    continue
                else:
                    log(f"[{i}/{len(patients)}] stale cache: {pid}")
            except Exception as e:
                log(f"[{i}/{len(patients)}] unreadable cache {pid}: {e}")

        log(f"[{i}/{len(patients)}] LLM: {pid}")
        try:
            resp = chain.invoke({
                "patient_id": pid,
                "source_qas": json.dumps(qas, ensure_ascii=False),
                "tasks": json.dumps(tasks, ensure_ascii=False),
                "n_tasks": n_tasks,
            })
            results = _clean_results([r.model_dump() for r in resp.results])
            results = _fill_missing(results, task_ids, pid)
            patient_answers[pid] = results
            out_file.write_text(json.dumps(
                {"patient_id": pid, "results": results},
                indent=2, ensure_ascii=False))
            processed += 1
            r0 = results[0] if results else {}
            log(f"  ok ({len(results)}/{n_tasks}): "
                f"conf={r0.get('assessment_confidence')} "
                f"evid_len={len(r0.get('evidence', []))}")
        except Exception as e:
            failed += 1
            log(f"  FAIL {pid}: {e}")

    log(f"Processed: {processed}  Skipped: {skipped}  "
        f"Failed: {failed}  Total loaded: {len(patient_answers)}")

    if len(patient_answers) < 2:
        log("Need >=2 patients for clustering.")
        return

    log("Overall clustering (allQs)")
    overall = cluster_overall(patient_answers, args.n_clusters, embedder)
    (out_dir / "clusters_overall.json").write_text(json.dumps(overall, indent=2))

    log("Per-question clustering")
    per_q = cluster_per_question(patient_answers, task_ids,
                                 args.n_clusters, embedder)
    per_q["allQs"] = overall
    (out_dir / "clusters_per_question.json").write_text(json.dumps(per_q, indent=2))

    log("Done.")

if __name__ == "__main__":
    main()