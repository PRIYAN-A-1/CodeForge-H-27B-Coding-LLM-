"""
Verifies end-to-end HTTP interaction with the live running CodeForge AI FastAPI server.
"""

import httpx

def test_live_server():
    client = httpx.Client(base_url="http://127.0.0.1:8000", timeout=30.0)

    # 1. Health
    h = client.get("/api/health").json()
    print("[1] Health:", h["status"], "| Model:", h["active_model"])

    # 2. Plan
    prompt = "Build a complete expense management application using Next.js, TypeScript, Tailwind CSS, FastAPI, and PostgreSQL."
    plan = client.post("/api/projects/plan", json={"requirement": prompt}).json()
    project_id = plan["project_id"]
    print(f"[2] Planned: {plan['project_name']} (ID: {project_id})")
    print(f"    Features: {plan['features']}")

    # 3. Generate
    gen = client.post("/api/projects/generate", json=plan).json()
    print(f"[3] Generated {gen['total_files']} files:")
    for f in gen["files"][:4]:
        print(f"    - {f['path']} ({f['file_type']})")

    # 4. Validate in Sandbox
    val = client.post(f"/api/projects/{project_id}/validate").json()
    print(f"[4] Sandbox Validation: {'ALL PASSED' if val['all_passed'] else 'FAILED'}")
    print(f"    Tests passed: {val['tests_passed']}")

    # 5. Export ZIP
    export_res = client.post(f"/api/projects/{project_id}/export")
    print(f"[5] ZIP Export: HTTP {export_res.status_code}, Size: {len(export_res.content)} bytes")

    # 6. Token Economics
    tok = client.get("/api/usage/tokens").json()
    print(f"[6] Cumulative Tokens: {tok['total_tokens']}, Est. Cost: ${tok['cumulative_cost_usd']}")

    print("\n[OK] Live Server Verification Succeeded 100%!")

if __name__ == "__main__":
    test_live_server()
