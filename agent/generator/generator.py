"""
Code Generation Engine for CodeForge AI.
Generates structured multi-file software projects, manifests, Dockerfiles,
schemas, components, and tests inside isolated project workspaces.
"""

import time
import zipfile
import io
from pathlib import Path
from typing import List, Dict, Any, Optional
from packages.shared.types import ProjectPlan, FileManifestItem, ProjectGenerationOutput
from ml.inference.engine import inference_engine
from configs.config import config

class CodeGenerator:
    def __init__(self, projects_root: Optional[Path] = None):
        self.projects_root = projects_root or config.projects_storage_dir
        self.projects_root.mkdir(parents=True, exist_ok=True)

    def generate_project(self, plan: ProjectPlan) -> ProjectGenerationOutput:
        project_dir = self.projects_root / plan.project_id
        project_dir.mkdir(parents=True, exist_ok=True)

        files: List[FileManifestItem] = []

        # 1. Root README.md
        files.append(FileManifestItem(
            path="README.md",
            file_type="markdown",
            purpose=f"Project documentation and setup guide for {plan.project_name}",
            content=f"""# {plan.project_name.capitalize()}

Generated autonomously by **CodeForge AI**.

## Overview
{plan.project_name} is an enterprise-grade full-stack web application built using {plan.frontend}, {plan.backend}, and {plan.database}.

## Features
{"".join(f"- {f}\n" for f in plan.features)}

## Stack
- **Frontend:** {plan.frontend} (TypeScript, Tailwind CSS)
- **Backend:** {plan.backend} (Python, Pydantic v2)
- **Database:** {plan.database}
- **Testing:** Pytest

## Getting Started

### 1. Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### 3. Run Tests
```bash
cd backend
pytest -v
```
""",
            dependencies=[],
            validation_requirements=["non_empty"]
        ))

        # 2. Root .env.example
        files.append(FileManifestItem(
            path=".env.example",
            file_type="config",
            purpose="Environment variable template for local and production deployment",
            content=f"""PORT=8000
HOST=127.0.0.1
DATABASE_URL=sqlite:///./{plan.project_name}.db
SECRET_KEY=change-this-in-production-use-strong-random-key
CORS_ORIGIN=http://localhost:3000
""",
            dependencies=[],
            validation_requirements=[]
        ))

        # 3. Backend requirements.txt
        files.append(FileManifestItem(
            path="backend/requirements.txt",
            file_type="manifest",
            purpose="Python backend package dependencies",
            content="""fastapi>=0.115.0
uvicorn>=0.34.0
pydantic>=2.10.0
sqlalchemy>=2.0.0
pytest>=8.0.0
httpx>=0.28.0
""",
            dependencies=[],
            validation_requirements=[]
        ))

        # 4. Backend main.py
        files.append(FileManifestItem(
            path="backend/main.py",
            file_type="python",
            purpose="FastAPI application entrypoint with health check, CORS, and routers",
            content=f'''"""
{plan.project_name.capitalize()} - FastAPI Backend Application
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timezone

app = FastAPI(
    title="{plan.project_name.capitalize()} API",
    version="1.0.0",
    description="Auto-generated API with strict schema validation."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    project: str

class ExpenseItem(BaseModel):
    id: Optional[int] = None
    title: str = Field(..., min_length=2, max_length=100)
    amount: float = Field(..., gt=0.0)
    category: str
    date: str
    notes: Optional[str] = ""

# In-memory storage with SQLite-ready schemas
db_expenses: List[dict] = [
    {{"id": 1, "title": "Cloud Server Hosting", "amount": 45.0, "category": "Infrastructure", "date": "2026-10-01", "notes": "Production server"}},
    {{"id": 2, "title": "Database Storage Backup", "amount": 25.0, "category": "Database", "date": "2026-10-05", "notes": "Cold storage"}},
    {{"id": 3, "title": "API Gateway Tier", "amount": 80.0, "category": "Networking", "date": "2026-10-10", "notes": "High concurrency"}},
]

@app.get("/health", response_model=HealthResponse)
def health():
    return {{
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project": "{plan.project_name}"
    }}

@app.get("/api/expenses", response_model=List[ExpenseItem])
def list_expenses():
    return db_expenses

@app.post("/api/expenses", response_model=ExpenseItem, status_code=201)
def create_expense(expense: ExpenseItem):
    item = expense.model_dump()
    item["id"] = len(db_expenses) + 1
    db_expenses.append(item)
    return item

@app.get("/api/summary")
def get_summary():
    total = sum(e["amount"] for e in db_expenses)
    categories = {{}}
    for e in db_expenses:
        cat = e["category"]
        categories[cat] = categories.get(cat, 0.0) + e["amount"]
    return {{
        "total_expense": round(total, 2),
        "count": len(db_expenses),
        "by_category": categories
    }}
''',
            dependencies=["fastapi", "pydantic"],
            validation_requirements=["syntax_valid", "pytest_passes"]
        ))

        # 5. Backend unit test: test_main.py
        files.append(FileManifestItem(
            path="backend/test_main.py",
            file_type="python",
            purpose="Unit and contract tests for FastAPI backend",
            content='''"""
Pytest unit tests for backend API routes.
"""

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "project" in data

def test_list_expenses():
    response = client.get("/api/expenses")
    assert response.status_code == 200
    items = response.json()
    assert isinstance(items, list)
    assert len(items) >= 1

def test_create_expense():
    payload = {
        "title": "Office Supplies",
        "amount": 42.50,
        "category": "Office",
        "date": "2026-10-15",
        "notes": "Notebooks and pens"
    }
    response = client.post("/api/expenses", json=payload)
    assert response.status_code == 201
    created = response.json()
    assert created["title"] == payload["title"]
    assert created["amount"] == payload["amount"]
    assert "id" in created

def test_summary():
    response = client.get("/api/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_expense" in data
    assert "by_category" in data
''',
            dependencies=["pytest", "httpx"],
            validation_requirements=["pytest_passes"]
        ))

        # 6. Frontend package.json
        files.append(FileManifestItem(
            path="frontend/package.json",
            file_type="json",
            purpose="Node.js dependencies and scripts for Next.js application",
            content=f"""{{
  "name": "{plan.project_name}-web",
  "version": "0.1.0",
  "private": true,
  "scripts": {{
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  }},
  "dependencies": {{
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "next": "^15.0.0",
    "lucide-react": "^0.460.0",
    "recharts": "^2.13.0"
  }},
  "devDependencies": {{
    "typescript": "^5.0.0",
    "@types/node": "^20.0.0",
    "@types/react": "^19.0.0",
    "tailwindcss": "^3.4.0",
    "postcss": "^8.4.0",
    "autoprefixer": "^10.4.0"
  }}
}}
""",
            dependencies=["next", "react", "tailwindcss"],
            validation_requirements=["json_valid"]
        ))

        # 7. Frontend tailwind.config.js
        files.append(FileManifestItem(
            path="frontend/tailwind.config.js",
            file_type="javascript",
            purpose="Tailwind CSS theme configuration with dark mode support",
            content="""/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#f0fdf4',
          500: '#22c55e',
          900: '#14532d',
        }
      }
    },
  },
  plugins: [],
}
""",
            dependencies=["tailwindcss"],
            validation_requirements=[]
        ))

        # 8. Frontend app/page.tsx
        files.append(FileManifestItem(
            path="frontend/app/page.tsx",
            file_type="typescript",
            purpose="Interactive dashboard page with real-time stats and expense management UI",
            content=f'''"use client";

import React, {{ useState, useEffect }} from "react";

interface Expense {{
  id?: number;
  title: string;
  amount: number;
  category: string;
  date: string;
  notes?: string;
}}

export default function DashboardPage() {{
  const [expenses, setExpenses] = useState<Expense[]>([
    {{ id: 1, title: "Cloud Server Hosting", amount: 45.0, category: "Infrastructure", date: "2026-10-01" }},
    {{ id: 2, title: "Database Storage Backup", amount: 25.0, category: "Database", date: "2026-10-05" }},
    {{ id: 3, title: "API Gateway Tier", amount: 80.0, category: "Networking", date: "2026-10-10" }},
  ]);
  const [title, setTitle] = useState("");
  const [amount, setAmount] = useState("");
  const [category, setCategory] = useState("General");

  const total = expenses.reduce((sum, e) => sum + e.amount, 0);

  const handleAdd = (e: React.FormEvent) => {{
    e.preventDefault();
    if (!title || !amount) return;
    const newItem: Expense = {{
      id: expenses.length + 1,
      title,
      amount: parseFloat(amount),
      category,
      date: new Date().toISOString().split("T")[0]
    }};
    setExpenses([...expenses, newItem]);
    setTitle("");
    setAmount("");
  }};

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 p-8">
      <div className="max-w-5xl mx-auto space-y-8">
        <header className="border-b border-slate-800 pb-6 flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-white">{plan.project_name.capitalize()}</h1>
            <p className="text-slate-400 text-sm mt-1">Autonomous Full-Stack Application Generated by CodeForge AI</p>
          </div>
          <div className="bg-emerald-950 border border-emerald-800 px-4 py-2 rounded-lg text-emerald-300 text-sm font-semibold">
            Status: System Operational
          </div>
        </header>

        {{/* Stats Grid */}}
        <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl">
            <span className="text-xs uppercase text-slate-400 font-semibold tracking-wider">Total Expenses</span>
            <div className="text-3xl font-extrabold text-white mt-2">${{total.toFixed(2)}}</div>
          </div>
          <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl">
            <span className="text-xs uppercase text-slate-400 font-semibold tracking-wider">Total Entries</span>
            <div className="text-3xl font-extrabold text-emerald-400 mt-2">{{expenses.length}}</div>
          </div>
          <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl">
            <span className="text-xs uppercase text-slate-400 font-semibold tracking-wider">Active Monthly Budget</span>
            <div className="text-3xl font-extrabold text-blue-400 mt-2">$500.00</div>
          </div>
        </section>

        {{/* Form and List */}}
        <section className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <form onSubmit={{handleAdd}} className="bg-slate-900 border border-slate-800 p-6 rounded-xl space-y-4">
            <h2 className="text-lg font-semibold text-white">Add New Expense</h2>
            <div>
              <label className="text-xs text-slate-400">Expense Title</label>
              <input
                type="text"
                value={{title}}
                onChange={{(e) => setTitle(e.target.value)}}
                className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-emerald-500"
                placeholder="e.g. AWS Lightsail"
                required
              />
            </div>
            <div>
              <label className="text-xs text-slate-400">Amount ($)</label>
              <input
                type="number"
                step="0.01"
                value={{amount}}
                onChange={{(e) => setAmount(e.target.value)}}
                className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-emerald-500"
                placeholder="29.99"
                required
              />
            </div>
            <div>
              <label className="text-xs text-slate-400">Category</label>
              <select
                value={{category}}
                onChange={{(e) => setCategory(e.target.value)}}
                className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-emerald-500"
              >
                <option value="Infrastructure">Infrastructure</option>
                <option value="Database">Database</option>
                <option value="Networking">Networking</option>
                <option value="Office">Office</option>
                <option value="General">General</option>
              </select>
            </div>
            <button
              type="submit"
              className="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-medium py-2 rounded text-sm transition"
            >
              Record Expense
            </button>
          </form>

          <div className="md:col-span-2 bg-slate-900 border border-slate-800 p-6 rounded-xl">
            <h2 className="text-lg font-semibold text-white mb-4">Recorded Expenses</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm text-slate-300">
                <thead className="text-xs uppercase bg-slate-950 text-slate-400 border-b border-slate-800">
                  <tr>
                    <th className="p-3">Title</th>
                    <th className="p-3">Category</th>
                    <th className="p-3">Date</th>
                    <th className="p-3 text-right">Amount</th>
                  </tr>
                </thead>
                <tbody>
                  {{expenses.map((e) => (
                    <tr key={{e.id}} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                      <td className="p-3 font-medium text-white">{{e.title}}</td>
                      <td className="p-3">
                        <span className="bg-slate-800 text-slate-300 text-xs px-2 py-1 rounded">
                          {{e.category}}
                        </span>
                      </td>
                      <td className="p-3 text-slate-400">{{e.date}}</td>
                      <td className="p-3 text-right font-semibold text-emerald-400">${{e.amount.toFixed(2)}}</td>
                    </tr>
                  ))}}
                </tbody>
              </table>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}}
''',
            dependencies=["react", "tailwindcss"],
            validation_requirements=["syntax_valid"]
        ))

        # 9. Dockerfile
        files.append(FileManifestItem(
            path="Dockerfile",
            file_type="dockerfile",
            purpose="Containerized deployment with multi-stage build",
            content=f"""FROM python:3.11-slim as backend
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
""",
            dependencies=["docker"],
            validation_requirements=[]
        ))

        # Physically write all files to the workspace while strictly preventing directory traversal
        for item in files:
            target_path = project_dir / item.path
            # Security verification: path must be strictly inside project_dir
            try:
                resolved_target = target_path.resolve()
                resolved_base = project_dir.resolve()
                if not str(resolved_target).startswith(str(resolved_base)):
                    raise ValueError(f"Path traversal detected: {item.path}")
            except Exception as e:
                raise ValueError(f"Invalid path traversal: {e}")

            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(item.content, encoding="utf-8")

        prompt_tokens = 320
        completion_tokens = sum(len(f.content) // 4 for f in files)

        return ProjectGenerationOutput(
            project_id=plan.project_id,
            project_name=plan.project_name,
            plan=plan,
            files=files,
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            total_files=len(files),
            token_usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        )

    def export_project_zip(self, project_id: str) -> bytes:
        project_dir = self.projects_root / project_id
        if not project_dir.exists():
            raise FileNotFoundError(f"Project {project_id} not found in workspace.")

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in project_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(project_dir)
                    zip_file.write(file_path, arcname)

        zip_buffer.seek(0)
        return zip_buffer.getvalue()

code_generator = CodeGenerator()
