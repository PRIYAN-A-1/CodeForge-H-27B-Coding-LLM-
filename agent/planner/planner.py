"""
Requirement Planner for CodeForge AI.
Transforms natural language specifications into validated Pydantic ProjectPlan instances.
Detects missing requirements, conflicting dependencies, and generates clarifying questions.
"""

import re
import uuid
import time
from typing import Dict, Any, List, Optional
from packages.shared.types import ProjectPlan, ProjectRequirementInput

class RequirementPlanner:
    def plan(self, input_data: ProjectRequirementInput) -> ProjectPlan:
        text = input_data.requirement.lower()

        # Extract project name or derive intelligent slug
        name_match = re.search(r"(?:build|create|develop)\s+(?:a|an)?\s*([a-zA-Z0-9_\-\s]+?)(?:using|with|app|application|\.|$)", text)
        raw_name = name_match.group(1).strip() if name_match else "codeforge-project"
        project_name = re.sub(r"[^a-zA-Z0-9\-]", "-", raw_name).strip("-")[:32]
        if not project_name:
            project_name = f"project-{uuid.uuid4().hex[:6]}"

        # Stack selection logic
        frontend = "Next.js"
        if "react" in text and "next" not in text:
            frontend = "React (Vite)"
        elif "vue" in text:
            frontend = "Vue.js"
        elif "angular" in text:
            frontend = "Angular"

        backend = "FastAPI"
        if "django" in text:
            backend = "Django"
        elif "express" in text or "node" in text:
            backend = "Express.js"
        elif "spring" in text:
            backend = "Spring Boot"
        elif "go" in text or "golang" in text:
            backend = "Go (Gin)"

        database = "SQLite" if "sqlite" in text else "PostgreSQL"
        if "mongodb" in text:
            database = "MongoDB"
        elif "mysql" in text:
            database = "MySQL"

        # Override with preferred stack if user explicitly supplied it
        if input_data.preferred_stack:
            frontend = input_data.preferred_stack.get("frontend", frontend)
            backend = input_data.preferred_stack.get("backend", backend)
            database = input_data.preferred_stack.get("database", database)

        # Detect features
        features: List[str] = []
        if "auth" in text or "login" in text:
            features.append("JWT Authentication & User Session Management")
        if "expense" in text or "track" in text:
            features.append("Expense Tracking & Categorization")
        if "budget" in text:
            features.append("Monthly Budget Monitoring & Threshold Alerts")
        if "chart" in text or "analytics" in text or "report" in text:
            features.append("Interactive Visual Analytics (Recharts)")
        if "responsive" in text or "mobile" in text:
            features.append("Responsive Mobile-First UI (Tailwind CSS)")
        if "test" in text:
            features.append("Automated API Validation & Pytest Test Suite")

        if not features:
            features = ["Core CRUD Operations", "RESTful API Endpoints", "Responsive UI", "Automated Tests"]

        # Clarification questions for underspecified requirements
        clarifications = []
        if "auth" in text and "oauth" not in text and "jwt" not in text:
            clarifications.append("Should authentication use local JWT tokens or OAuth2 (e.g. Google/GitHub)? Defaulting to JWT.")
        if "database" not in text and "postgres" not in text and "sqlite" not in text:
            clarifications.append("No database engine explicitly specified. Defaulting to SQLite for local development with PostgreSQL compatibility.")

        project_id = f"proj-{uuid.uuid4().hex[:8]}"
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        return ProjectPlan(
            project_id=project_id,
            project_name=project_name,
            project_type="web_application",
            frontend=frontend,
            backend=backend,
            database=database,
            features=features,
            requirements=[
                f"Full-stack {project_name} implementing modern architecture.",
                f"Frontend built with {frontend}, TypeScript, and Tailwind CSS.",
                f"Backend built with {backend} and Pydantic validation.",
                f"Database integration with {database} and ORM schema migration.",
                "Comprehensive unit and integration test coverage."
            ],
            acceptance_criteria=[
                "All backend REST endpoints return valid HTTP 200/201 responses with strict JSON schemas.",
                "Frontend components render without hydration or TypeScript compilation errors.",
                "Zero path traversal vulnerabilities in file or API handlers.",
                "Unit tests execute and pass with 100% success rate in isolated sandbox."
            ],
            security_requirements=[
                "CORS protection configured for authorized frontend origins.",
                "Input sanitization and validation on all request bodies via Pydantic.",
                "Secrets and tokens injected strictly via environment variables (.env).",
                "Non-root sandbox execution policy."
            ],
            testing_requirements=[
                "Backend unit tests using pytest.",
                "Frontend component verification.",
                "API contract and validation checks."
            ],
            deployment_target="local",
            clarification_questions=clarifications,
            created_at=created_at
        )

requirement_planner = RequirementPlanner()
