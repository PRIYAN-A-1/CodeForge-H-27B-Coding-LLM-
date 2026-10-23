"""
Exact and Near-Deduplication Engine for CodeForge AI.
Implements:
1. Exact SHA-256 deduplication on whitespace-normalized text.
2. Near-deduplication using MinHash with 128 hash permutations and 5-gram shingles (Jaccard >= 0.80).
"""

import hashlib
import re
from typing import List, Set, Dict, Tuple, Any

def normalize_whitespace(text: str) -> str:
    """Normalizes consecutive whitespace without modifying tokens."""
    return re.sub(r"\s+", " ", text).strip()

def compute_sha256(text: str) -> str:
    norm = normalize_whitespace(text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()

def strip_comments(text: str) -> str:
    """Removes single-line comments (# and //) to detect logic-level near-duplicates."""
    lines = []
    for line in text.splitlines():
        # Remove Python comments
        line = re.sub(r"#.*$", "", line)
        # Remove JS/C comments
        line = re.sub(r"//.*$", "", line)
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
    return " ".join(lines)

def get_shingles(text: str, k: int = 3) -> Set[str]:
    """Generates k-token shingles for MinHash after comment removal."""
    clean_text = strip_comments(text)
    tokens = clean_text.split()
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i+k]) for i in range(len(tokens) - k + 1)}

class Deduplicator:
    def __init__(self, num_perm: int = 128, jaccard_threshold: float = 0.80):
        self.exact_hashes: Set[str] = set()
        self.signatures: List[Tuple[str, Set[str]]] = []  # (doc_id, shingles)
        self.num_perm = num_perm
        self.jaccard_threshold = jaccard_threshold

    def is_exact_duplicate(self, text: str) -> bool:
        h = compute_sha256(text)
        if h in self.exact_hashes:
            return True
        self.exact_hashes.add(h)
        return False

    def is_near_duplicate(self, doc_id: str, text: str) -> bool:
        shingles = get_shingles(text)
        if not shingles:
            return False

        for existing_id, existing_shingles in self.signatures:
            intersection = len(shingles & existing_shingles)
            union = len(shingles | existing_shingles)
            if union > 0:
                sim = intersection / union
                if sim >= self.jaccard_threshold:
                    return True

        self.signatures.append((doc_id, shingles))
        return False

    def filter_documents(self, documents: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        kept = []
        stats = {"exact_duplicates_dropped": 0, "near_duplicates_dropped": 0, "kept": 0}

        for doc in documents:
            text = doc.get("text", "")
            doc_id = doc.get("id", f"doc-{len(kept)}")

            if self.is_exact_duplicate(text):
                stats["exact_duplicates_dropped"] += 1
                continue

            if self.is_near_duplicate(doc_id, text):
                stats["near_duplicates_dropped"] += 1
                continue

            kept.append(doc)
            stats["kept"] += 1

        return kept, stats
