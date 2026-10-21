#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from sklearn.model_selection import StratifiedKFold

from common import (
    log, speaker_key, bulk_load_patients, patient_text,
    CachedEmbedder, cosine_top_k,
)

# ---------- schema ----------
class EvidenceItem(BaseModel):
    qa_id: str
    field: Optional[str] = None
    quote: Optional[str] = None

class LabeledResult(BaseModel):
    id: str
    answer: str
    assessment_confidence: float = Field(ge=0, le=100)
    predicted_label: str
    evidence: list[EvidenceItem] = Field(default_factory=list)

class LabeledBatch(BaseModel):
    results: list[LabeledResult]

def _clean_results(results: list[dict]) -> list[dict]:
    for r in results:
        kept = []
        for ev in (r.get("evidence") or []):
            if ev.get("field") in {"question", "answer"} and ev.get("quote"):
                kept.append(ev)
        r["evidence"] = kept
    return results

def _fill_missing(results: list[dict], task_ids: list[str], pid: str) -> list[dict]:
    got = {r["id"]: r for r in results if r.get("id") in task_ids}
    missing = [tid for tid in task_ids if tid not in got]
    if missing:
        log(f"  INCOMPLETE {pid}: missing {len(missing)} tasks: {missing}")
        for tid in missing:
            got[tid] = {
                "id": tid,
                "answer": "Don't know",
                "assessment_confidence": 0.0,
                "predicted_label": "Don't know",
                "evidence": [],
            }
    return [got[tid] for tid in task_ids]

# ---------- prompts ----------
SYSTEM_PROMPT = """You analyse a patient interview Q&A record.

You are given labeled examples from similar patients (reference_examples).
Use them ONLY to calibrate the scale of your answer. Do NOT copy their
content into the current patient's evidence.

For each task, produce:
- id: the task id
- answer: text answer (Yes/No/Don't know for yes_no_dont_know tasks)
- assessment_confidence: number from 0 to 100
- predicted_label: the label you would assign using the same scheme as the examples
- evidence: list of objects

Every evidence object MUST contain exactly three keys: qa_id, field, quote.
- field must be either question or answer
- quote must be an exact substring of THIS patient's Q&As
If you cannot supply all three keys for an item, do NOT emit that item at all.

Rules:
- You MUST return exactly one result per task, in the order given.
- Never omit a task, even if the answer is unclear — use Don't know instead.
- Return only valid JSON matching the schema."""

HUMAN_PROMPT = """Patient ID: {patient_id}

This patient's Q&As:
{source_qas}

Labeled reference examples (label field: {label_field}):
{reference_examples}

Analysis tasks ({n_tasks} total):
{tasks}

Return EXACTLY {n_tasks} results — one for every task above, in the same order.
Do not skip any task. If a task cannot be answered, return answer "Don't know"
and predicted_label "Don't know" with an empty evidence list.
"""

def build_chain(model):
    llm = ChatOllama(
        model=model,
        temperature=0,
        num_predict=1400,
        num_ctx=4096,
    )
    structured = llm.with_structured_output(LabeledBatch)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])
    return prompt | structured

# ---------- CSV labels ----------
def _norm(h): return re.sub(r"[^a-z0-9]", "", h.strip().lower())
def _clean_diag(v):
    if v is None: return None
    v = v.strip()
    return None if (not v or v.lower() == "unknown") else v
def _clean_scale(v, lo, hi):
    if v is None: return None
    v = v.strip()
    if not v or v.upper() == "XX": return None
    try: n = int(float(v))
    except ValueError: return None
    return n if lo <= n <= hi else None

def load_labels_csv(path):
    labels = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        hm = {_norm(h): h for h in (r.fieldnames or [])}
        pid_k  = hm.get("patientid") or hm.get("patient")
        diag_k = hm.get("diagnosis")
        phq_k  = hm.get("phq9") or hm.get("phq")
        gad_k  = hm.get("gad7") or hm.get("gad")
        for row in r:
            pid = (row.get(pid_k) or "").strip()
            if not pid: continue
            labels[pid] = {
                "Diagnosis": _clean_diag(row.get(diag_k)) if diag_k else None,
                "PhQ9": _clean_scale(row.get(phq_k), 0, 27) if phq_k else None,
                "GAD7": _clean_scale(row.get(gad_k), 0, 21) if gad_k else None,
            }
    return labels

def _resolve_ids(vals):
    if not vals: return vals
    if len(vals) == 1 and Path(vals[0]).is_file():
        return [l.strip() for l in Path(vals[0]).read_text().splitlines()
                if l.strip() and not l.startswith("#")]
    return vals

# ---------- metrics ----------
def metrics_for(gold, pred, label_field):
    if not gold: return {"n_eval": 0}
    correct = sum(1 for g, p in zip(gold, pred) if str(g) == str(p))
    out = {"n_eval": len(gold), "correct": correct,
           "accuracy": round(correct / len(gold), 4), "mae": None}
    if label_field in {"PhQ9", "GAD7"}:
        try:
            g = np.array([float(x) for x in gold])
            p = np.array([float(x) for x in pred])
            out["mae"] = round(float(np.mean(np.abs(g - p))), 3)
        except ValueError:
            pass
    return out

# ---------- predict ----------
def predict_session(chain, embedder, train_vecs, train_meta,
                    pid, qas, label_field, tasks, task_ids,
                    k_examples):
    q_vec = embedder.embed_one(patient_text(qas))
    top = cosine_top_k(q_vec, train_vecs, k_examples)
    examples = [train_meta[i] for i in top]
    resp = chain.invoke({
        "patient_id": pid,
        "source_qas": json.dumps(qas, ensure_ascii=False),
        "reference_examples": json.dumps(examples, ensure_ascii=False),
        "label_field": label_field,
        "tasks": json.dumps(tasks, ensure_ascii=False),
        "n_tasks": len(tasks),
    })
    results = _clean_results([r.model_dump() for r in resp.results])
    results = _fill_missing(results, task_ids, pid)
    return results

# ---------- main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input-dir", required=True)
    p.add_argument("-q", "--analysis-questions", required=True)
    p.add_argument("-o", "--output-dir", required=True)
    p.add_argument("-m", "--model", default="llama3.2:3b")
    p.add_argument("--labels", required=True)
    p.add_argument("--label-field", required=True,
                   choices=["Diagnosis", "PhQ9", "GAD7"])
    p.add_argument("--train-ids", nargs="*", default=None)
    p.add_argument("--eval-ids",  nargs="*", default=None)
    p.add_argument("--cv", type=int, default=None)
    p.add_argument("--eval-fraction", type=float, default=0.3)
    p.add_argument("--k-examples", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    p.add_argument("--recheck-empty", action="store_true")
    args = p.parse_args()

    args.eval_ids  = _resolve_ids(args.eval_ids)
    args.train_ids = _resolve_ids(args.train_ids)

    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qfile = Path(args.analysis_questions)

    tasks = json.loads(qfile.read_text())
    task_ids = [t["id"] for t in tasks]
    n_tasks = len(tasks)
    log(f"Tasks ({n_tasks}): {task_ids}")

    labels = load_labels_csv(Path(args.labels))
    patients = bulk_load_patients(in_dir, qfile)

    labeled_pids = sorted(pid for pid in patients
                          if pid in labels
                          and labels[pid].get(args.label_field) is not None)
    log(f"Labeled patients: {len(labeled_pids)} for {args.label_field}")
    if not labeled_pids:
        raise SystemExit("No labeled patients")

    speaker_to_pids: dict[str, list[str]] = {}
    for pid in labeled_pids:
        speaker_to_pids.setdefault(speaker_key(pid), []).append(pid)
    speakers = sorted(speaker_to_pids)
    log(f"Labeled speakers: {len(speakers)}")

    speaker_label = {}
    for spk in speakers:
        vals = {labels[p][args.label_field] for p in speaker_to_pids[spk]}
        if len(vals) > 1:
            log(f"WARN: speaker {spk} inconsistent {args.label_field}: {vals}")
        non_none = [v for v in vals if v is not None]
        speaker_label[spk] = non_none[0] if non_none else None
    speakers = [s for s in speakers if speaker_label[s] is not None]
    log(f"Speakers with usable labels: {len(speakers)}")

    embedder = CachedEmbedder("nomic-embed-text",
                              out_dir / "embeddings_cache.json")
    chain = build_chain(args.model)
    all_rows: list[dict] = []

    def _is_cached_usable(cached):
        results = cached.get("results", [])
        if not results:
            return False
        if args.recheck_empty:
            return not all(
                r.get("assessment_confidence", 0) == 0.0
                and not r.get("evidence")
                for r in results
            )
        return True

    def run_fold(train_ids, eval_ids, fold_tag):
        train_texts = [patient_text(patients[p]) for p in train_ids]
        train_vecs  = embedder.embed_many(train_texts)
        train_meta  = [{"patient_id": p,
                        "speaker_id": speaker_key(p),
                        "label": labels[p][args.label_field],
                        "snippet": t[:400]}
                       for p, t in zip(train_ids, train_texts)]
        rows = []
        for pid in eval_ids:
            out_file = out_dir / f"{pid}.json"
            if out_file.exists() and not args.force:
                try:
                    cached = json.loads(out_file.read_text())
                    if _is_cached_usable(cached):
                        log(f"  [{fold_tag}] cached: {pid}")
                        for r in cached["results"]:
                            rows.append({
                                "patient_id": pid,
                                "speaker_id": speaker_key(pid),
                                "task_id": r["id"],
                                "fold": fold_tag,
                                "gold": str(labels[pid][args.label_field]),
                                "pred": r["predicted_label"],
                                "answer": r["answer"],
                                "confidence": r["assessment_confidence"],
                                "evidence": json.dumps(r["evidence"],
                                                       ensure_ascii=False),
                            })
                        continue
                except Exception as e:
                    log(f"  [{fold_tag}] unreadable cache {pid}: {e}")

            log(f"  [{fold_tag}] {pid}")
            try:
                results = predict_session(chain, embedder, train_vecs,
                                          train_meta, pid, patients[pid],
                                          args.label_field, tasks, task_ids,
                                          args.k_examples)
                out_file.write_text(json.dumps(
                    {"patient_id": pid, "speaker_id": speaker_key(pid),
                     "fold": fold_tag, "results": results},
                    indent=2, ensure_ascii=False))
                for r in results:
                    rows.append({
                        "patient_id": pid,
                        "speaker_id": speaker_key(pid),
                        "task_id": r["id"],
                        "fold": fold_tag,
                        "gold": str(labels[pid][args.label_field]),
                        "pred": r["predicted_label"],
                        "answer": r["answer"],
                        "confidence": r["assessment_confidence"],
                        "evidence": json.dumps(r["evidence"],
                                               ensure_ascii=False),
                    })
            except Exception as e:
                log(f"  FAIL {pid}: {e}")
        return rows

    mode = None

    if args.cv and not args.eval_ids:
        if args.label_field == "Diagnosis":
            ystr = np.array([str(speaker_label[s]) for s in speakers])
        else:
            vals = np.array([int(speaker_label[s]) for s in speakers])
            ystr = np.digitize(vals, [4, 9], right=True)

        _, counts = np.unique(ystr, return_counts=True)
        k_eff = min(args.cv, int(counts.min()))
        if k_eff < 2:
            raise SystemExit(f"Cannot do CV: smallest class {counts.min()} speakers")
        log(f"Speaker-independent stratified {k_eff}-fold CV")

        skf = StratifiedKFold(n_splits=k_eff, shuffle=True, random_state=args.seed)
        per_fold = []
        for fi, (tr_idx, ev_idx) in enumerate(skf.split(speakers, ystr), 1):
            train_speakers = [speakers[i] for i in tr_idx]
            eval_speakers  = [speakers[i] for i in ev_idx]
            assert not (set(train_speakers) & set(eval_speakers))
            train_ids = [p for s in train_speakers for p in speaker_to_pids[s]]
            eval_ids  = [p for s in eval_speakers  for p in speaker_to_pids[s]]
            log(f"Fold {fi}: train {len(train_speakers)}spk/{len(train_ids)}ses "
                f"-> eval {len(eval_speakers)}spk/{len(eval_ids)}ses")
            fold_rows = run_fold(train_ids, eval_ids, f"fold_{fi}")
            all_rows.extend(fold_rows)
            per_fold.append({
                "fold": fi,
                "n_train_speakers": len(train_speakers),
                "n_eval_speakers":  len(eval_speakers),
                "n_train_sessions": len(train_ids),
                "n_eval_sessions":  len(eval_ids),
                **metrics_for([r["gold"] for r in fold_rows],
                              [r["pred"] for r in fold_rows],
                              args.label_field),
            })

        accs = [m["accuracy"] for m in per_fold if "accuracy" in m]
        maes = [m["mae"] for m in per_fold if m.get("mae") is not None]
        (out_dir / "cv_folds.json").write_text(json.dumps({
            "mode": f"speaker_independent_cv_{k_eff}",
            "label_field": args.label_field,
            "k_folds": k_eff,
            "n_speakers_total": len(speakers),
            "per_fold": per_fold,
            "accuracy_mean": round(float(np.mean(accs)), 4) if accs else None,
            "accuracy_std":  round(float(np.std(accs)), 4) if accs else None,
            "mae_mean": round(float(np.mean(maes)), 3) if maes else None,
            "mae_std":  round(float(np.std(maes)), 3) if maes else None,
        }, indent=2))
        mode = f"speaker_independent_cv_{k_eff}"

    elif args.eval_ids:
        eval_ids = [p for p in args.eval_ids if p in labeled_pids]
        train_ids = (args.train_ids if args.train_ids
                     else [p for p in labeled_pids if p not in eval_ids])
        leaked = {speaker_key(p) for p in train_ids} & {speaker_key(p) for p in eval_ids}
        if leaked:
            raise SystemExit(f"Speaker leakage: {sorted(leaked)}")
        log(f"Fixed split: {len(train_ids)}ses train / {len(eval_ids)}ses eval")
        all_rows = run_fold(train_ids, eval_ids, "fixed")
        mode = "fixed"

    else:
        rng = np.random.default_rng(args.seed)
        shuffled = list(speakers); rng.shuffle(shuffled)
        n_eval_spk = max(1, int(round(len(shuffled) * args.eval_fraction)))
        eval_spk = set(shuffled[:n_eval_spk])
        train_spk = set(shuffled[n_eval_spk:])
        eval_ids  = [p for s in eval_spk  for p in speaker_to_pids[s]]
        train_ids = [p for s in train_spk for p in speaker_to_pids[s]]
        log(f"Speaker holdout: train {len(train_spk)}spk/{len(train_ids)}ses, "
            f"eval {len(eval_spk)}spk/{len(eval_ids)}ses")
        all_rows = run_fold(train_ids, eval_ids, "holdout")
        mode = "speaker_holdout"

    per_task = {}
    for tid in task_ids + ["allQs"]:
        sel = all_rows if tid == "allQs" else [r for r in all_rows if r["task_id"] == tid]
        if sel:
            per_task[tid] = metrics_for(
                [r["gold"] for r in sel], [r["pred"] for r in sel],
                args.label_field)
    (out_dir / "metrics_per_task.json").write_text(json.dumps({
        "mode": mode,
        "label_field": args.label_field,
        "per_task": per_task,
        "pooled": metrics_for([r["gold"] for r in all_rows],
                              [r["pred"] for r in all_rows],
                              args.label_field),
    }, indent=2))

    with (out_dir / "predictions.csv").open(
            "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=[
            "patient_id", "speaker_id", "task_id", "fold",
            "gold", "pred", "answer", "confidence", "evidence"])
        w.writeheader(); w.writerows(all_rows)

    log(f"Done. Outputs in {out_dir}")

if __name__ == "__main__":
    main()