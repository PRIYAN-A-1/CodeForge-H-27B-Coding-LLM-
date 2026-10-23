"""
Fill-in-the-Middle (FIM) and uint16 Packing Pipeline for CodeForge-H.
Section I specification:
- 50% of code documents undergo FIM transformation.
- Split at two random positions into Prefix (A), Suffix (C), and Middle (B):
  <fim_prefix>A<fim_suffix>C<fim_middle>B<|endoftext|>
- Same-repo multi-file concatenation with <filename> tags.
- Packing sequences to seq_len (2048) with <|endoftext|> separators.
- Export to uint16 memmap array format.
"""

import random
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
from ml.data.tokenization.tokenizer import tokenizer

def apply_fim(code: str, fim_rate: float = 0.50, seed: int = 1337) -> str:
    """
    Applies FIM transformation to code document with specified probability.
    """
    rng = random.Random(seed)
    if rng.random() > fim_rate or len(code) < 30:
        return code

    # Split into 3 chunks: A (prefix), B (middle), C (suffix)
    idx1 = rng.randint(5, max(6, len(code) // 2))
    idx2 = rng.randint(idx1, max(idx1 + 1, len(code) - 5))

    a = code[:idx1]
    b = code[idx1:idx2]
    c = code[idx2:]

    return f"<fim_prefix>{a}<fim_suffix>{c}<fim_middle>{b}<|endoftext|>"

def pack_sequences(
    tokenized_docs: List[List[int]],
    seq_len: int = 2048,
    eod_token_id: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Packs variable-length tokenized sequences into fixed-size chunks of length seq_len.
    Returns (packed_input_ids as uint16 array, document_ids array).
    """
    flat_tokens = []
    flat_doc_ids = []

    for doc_id, doc in enumerate(tokenized_docs):
        flat_tokens.extend(doc)
        flat_tokens.append(eod_token_id)
        flat_doc_ids.extend([doc_id] * (len(doc) + 1))

    total_chunks = len(flat_tokens) // seq_len
    if total_chunks == 0:
        # Pad up to seq_len
        pad_len = seq_len - len(flat_tokens)
        flat_tokens.extend([eod_token_id] * pad_len)
        flat_doc_ids.extend([0] * pad_len)
        total_chunks = 1

    trimmed_tokens = flat_tokens[:total_chunks * seq_len]
    trimmed_doc_ids = flat_doc_ids[:total_chunks * seq_len]

    input_ids = np.array(trimmed_tokens, dtype=np.uint16).reshape((total_chunks, seq_len))
    doc_ids = np.array(trimmed_doc_ids, dtype=np.int32).reshape((total_chunks, seq_len))

    return input_ids, doc_ids
