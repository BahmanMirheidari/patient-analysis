# common.py
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from langchain_ollama import OllamaEmbeddings

def speaker_key(pid: str) -> str:
    """
    patient_id = <real_id_part1>_<real_id_part2>_<date>_<time>
    Return the speaker-level key so multiple sessions of the same person
    stay together.
    """
    parts = str(pid).split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return str(pid)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bulk_load_patients(input_dir: Path, questions_file: Path) -> Dict[str, List[dict]]:
    """
    Read every *.json in input_dir exactly once. Returns {pid: [qa, ...]}.
    Uses file name (sans .json) as fallback pid.
    """
    patients = {}
    files = sorted(p for p in input_dir.glob("*.json")
                   if p.resolve() != questions_file.resolve())
    log(f"Found {len(files)} patient JSON files")
    for i, pf in enumerate(files, 1):
        try:
            data = json.loads(pf.read_text(encoding="utf-8"))
            pid = str(data.get("patient_id", pf.stem))
            qas = data.get("QAs", {})
            out = []
            for k in sorted(qas, key=lambda x: int(re.search(r"\d+", x).group())):
                if not re.fullmatch(r"Q\d+", str(k), re.I):
                    continue
                a_key = f"A{re.search(r'\d+', k).group()}"
                q_val, a_val = qas.get(k, ""), qas.get(a_key, "")
                if isinstance(q_val, dict):
                    a_val = q_val.get("answer", a_val)
                    q_val = q_val.get("question", "")
                out.append({"qa_id": k,
                            "question": str(q_val).strip(),
                            "answer": str(a_val).strip()})
            if out:
                patients[pid] = out
        except Exception as e:
            log(f"  SKIP {pf.name}: {e}")
        if i % 20 == 0:
            log(f"  parsed {i}/{len(files)}")
    log(f"Loaded {len(patients)} patients")
    return patients


def patient_text(qas: List[dict]) -> str:
    return " ".join(f"{q['question']} {q['answer']}" for q in qas)


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class CachedEmbedder:
    """Embeds texts via Ollama, caching results to disk by content hash."""

    def __init__(self, model: str, cache_path: Path):
        self.model = model
        self.cache_path = cache_path
        self.cache: Dict[str, List[float]] = {}
        if cache_path.exists():
            try:
                self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
                log(f"Embedding cache: {len(self.cache)} entries loaded")
            except Exception as e:
                log(f"Embedding cache corrupt, ignoring: {e}")
                self.cache = {}
        self._client = OllamaEmbeddings(model=model)

    def _save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self.cache, ensure_ascii=False), encoding="utf-8"
        )

    def embed_one(self, text: str) -> List[float]:
        h = _hash(text)
        if h in self.cache:
            return self.cache[h]
        vec = self._client.embed_query(text)
        self.cache[h] = vec
        return vec

    def embed_many(self, texts: List[str]) -> np.ndarray:
        """Embed a batch, only calling Ollama for cache misses."""
        out = []
        misses, miss_idx = [], []
        for i, t in enumerate(texts):
            h = _hash(t)
            if h in self.cache:
                out.append(self.cache[h])
            else:
                out.append(None)
                misses.append(t)
                miss_idx.append(i)
        if misses:
            log(f"Embedding {len(misses)} new texts via Ollama (cache misses)")
            new_vecs = self._client.embed_documents(misses)
            for idx, vec in zip(miss_idx, new_vecs):
                out[idx] = vec
                self.cache[_hash(texts[idx])] = vec
            self._save()
        return np.array(out)


def cosine_top_k(query_vec, matrix, k):
    q = np.array(query_vec)
    sims = matrix @ q / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q) + 1e-9)
    return np.argsort(-sims)[:k]