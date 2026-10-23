# Token Usage & Economic Accounting (TOKEN_ACCOUNTING.md)

## 1. Accounting Ledger
Every inference request processed by CodeForge AI (`ml/inference/engine.py`) records:
- **Request ID:** Unique UUID string (`req-xxxxxxxx`)
- **Timestamp:** UTC ISO-8601
- **Model ID:** Active model tag (`codeforge-mock`, `qwen3.8:27b`, etc.)
- **Task Type:** e.g. `code_generation`, `repair_loop`, `planning`
- **Prompt Tokens:** Byte-Level BPE estimated tokens (~3.6 chars/token)
- **Completion Tokens:** Generated tokens
- **Latency:** Round-trip latency in milliseconds
- **Estimated Cost:** Hardware compute electricity estimate ($0.0001 / 1,000 tokens for local workstations)

## 2. API Telemetry Endpoint
Query real-time token metrics:
```bash
GET http://127.0.0.1:8000/api/usage/tokens
```
Returns:
```json
{
  "total_requests": 14,
  "prompt_tokens": 4200,
  "completion_tokens": 12800,
  "total_tokens": 17000,
  "cumulative_cost_usd": 0.0017,
  "average_latency_ms": 320.45
}
```
