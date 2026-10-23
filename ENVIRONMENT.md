# Environment & Runtime Specification

**Last Verified:** 2026-10-23  
**System:** Windows 11 Home Single Language x64 (Build 10.0.26300)  
**Host Architecture:** AMD Ryzen 5 7235HS (4C / 8T), 23.69 GB RAM  
**GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (6,144 MB VRAM), Driver 591.74, CUDA 13.1  

## Runtimes & Libraries

| Component | Version | Path / Reference | Notes |
| :--- | :--- | :--- | :--- |
| **Python** | 3.13.13 | `C:\Users\priya\AppData\Local\Programs\Python\Python313\python.exe` | Standard CPython 64-bit |
| **pip** | 26.2.1 | Python site-packages | Modern pip installer |
| **Node.js** | v26.7.0 | System PATH | Modern Node runtime |
| **npm** | 11.19.0 | System PATH | Package manager |
| **Git** | 2.55.0.windows.3 | System PATH | Version control |
| **PyTorch** | 2.14.0+cpu | Python site-packages | CPU build in Python 3.13 (`torch.cuda.is_available() == False`) |
| **Transformers** | 5.18.0 | Python site-packages | Hugging Face model suite |
| **Tokenizers** | 0.23.2 | Python site-packages | Fast Rust BPE/WordPiece tokenizers |
| **FastAPI** | 0.115.12 | Python site-packages | REST backend framework |
| **Uvicorn** | 0.34.2 | Python site-packages | ASGI server |
| **Pydantic** | 2.13.4 | Python site-packages | Schema validation (v2) |
| **SQLAlchemy** | 2.0.52 | Python site-packages | ORM & database layer |
| **Pytest** | 8.3.5 | Python site-packages | Automated test runner |
| **Ollama** | Active | `http://127.0.0.1:11434` | Running with `qwen3.8:27b` installed |
| **Docker** | Not on PATH | N/A | Process-level sandbox jail utilized |

## Seed & Precision Guarantees
- Random Seed: **1337** across all PyTorch, NumPy, and Python random instances.
- Precision: Default **bfloat16** or **float32** on CPU fallbacks; **int4 / NF4** for quantized inference.
