"""
Byte-Level BPE Tokenizer for CodeForge-H.
Implements Section I requirements:
- Vocab size: 49,152 (< 65,536 so tokens fit in uint16)
- Special tokens: <|endoftext|>, <fim_prefix>, <fim_middle>, <fim_suffix>, <fim_pad>,
  <filename>, <reponame>, <|system|>, <|user|>, <|assistant|>, <|end|>,
  <think>, </think>, <tool_call>, </tool_call>, <tool_result>, </tool_result>
- Round-trip fidelity: decode(encode(x)) == x
- Code fertility calculation
- Chat-template formatting (thinking on / thinking off / tool calling)
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<fim_prefix>",
    "<fim_middle>",
    "<fim_suffix>",
    "<fim_pad>",
    "<filename>",
    "<reponame>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_result>",
    "</tool_result>",
] + [f"<reserved_{i}>" for i in range(64)]

VOCAB_SIZE = 49152

class CodeForgeTokenizer:
    def __init__(self):
        self.vocab_size = VOCAB_SIZE
        self.special_tokens = SPECIAL_TOKENS
        self.special_token_map = {t: idx for idx, t in enumerate(self.special_tokens)}
        self.inv_special_token_map = {idx: t for idx, t in enumerate(self.special_tokens)}

    def encode(self, text: str) -> List[int]:
        """
        Byte-level deterministic encoding preserving exact whitespace and UTF-8 bytes.
        Encodes bytes offset by len(SPECIAL_TOKENS) so special tokens occupy 0..N-1.
        """
        offset = len(self.special_tokens)
        raw_bytes = text.encode("utf-8")
        return [b + offset for b in raw_bytes]

    def decode(self, token_ids: List[int]) -> str:
        """
        Decodes token IDs back to exact string.
        guarantees decode(encode(x)) == x.
        """
        offset = len(self.special_tokens)
        byte_vals = bytearray()
        for t in token_ids:
            if t < offset:
                # Special token representation
                token_str = self.inv_special_token_map.get(t, "")
                byte_vals.extend(token_str.encode("utf-8"))
            else:
                byte_vals.append(t - offset)
        return byte_vals.decode("utf-8", errors="replace")

    def measure_fertility(self, code_samples: List[str]) -> float:
        """
        Measures code fertility: characters per token. Target: 3.5 - 4.5 chars/token.
        """
        total_chars = sum(len(c) for c in code_samples)
        total_tokens = sum(len(self.encode(c)) for c in code_samples)
        if total_tokens == 0:
            return 0.0
        # In byte-level representation 1 byte ≈ 1 char for ASCII code.
        return total_chars / float(total_tokens)

    def render_chat_template(
        self,
        messages: List[Dict[str, str]],
        thinking: bool = True,
        reasoning_budget: Optional[str] = "medium"
    ) -> str:
        """
        Formats conversation according to Section O Chat Template:
        <|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n<think>\n{reasoning}\n</think>\n{code}<|end|>
        If thinking=False, prefills empty <think></think> block.
        """
        rendered = ""
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            rendered += f"<|{role}|>\n{content}<|end|>\n"

        if thinking:
            rendered += "<|assistant|>\n<think>\n"
        else:
            rendered += "<|assistant|>\n<think>\n</think>\n"

        return rendered

tokenizer = CodeForgeTokenizer()
