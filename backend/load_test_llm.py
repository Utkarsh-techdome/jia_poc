"""
Resume-JD relevance load test for llama.cpp OpenAI-compatible endpoint.

Sends (resume, job_description) pairs to the model and asks for a structured
JSON relevance score. Measures latency, throughput, and score consistency
across repeated runs of the same pair.

Usage:
    python load_test_llm.py
    python load_test_llm.py --requests 30 --concurrency 8
    python load_test_llm.py --consistency-rounds 5   # repeat each resume 5x to check score drift
"""

import asyncio
import argparse
import json
import time
import statistics
from openai import AsyncOpenAI

BASE_URL = "https://assetid-65.tail55f76c.ts.net/v1"
MODEL    = "LFM2.5-8B-A1B-Q5_K_M.gguf"

client = AsyncOpenAI(base_url=BASE_URL, api_key="none")

# ---------------------------------------------------------------------------
# Job Description
# ---------------------------------------------------------------------------
JOB_DESCRIPTION = """
Position: Senior Backend Engineer
Company: TechDome

Responsibilities:
- Design and build scalable REST and WebSocket APIs using Python (FastAPI / Django).
- Integrate and maintain LLM-based pipelines using OpenAI / Azure OpenAI APIs.
- Write and optimise SQL queries against PostgreSQL; manage schema migrations.
- Deploy and monitor microservices on AWS (ECS, Lambda, RDS, S3).
- Collaborate with frontend and ML teams; participate in code reviews.

Required Skills:
- 4+ years Python backend development.
- Strong knowledge of FastAPI or Django REST Framework.
- Experience with PostgreSQL and Redis.
- Familiarity with Docker and CI/CD pipelines (GitHub Actions).
- Working knowledge of LLMs / prompt engineering.

Nice to have:
- Celery for async task queues.
- WebRTC or real-time audio/video streaming.
- Experience with React or any frontend framework.
"""

# ---------------------------------------------------------------------------
# Sample resumes — varying relevance to the JD above
# ---------------------------------------------------------------------------
RESUMES = [
    {
        "id": "R1_perfect_match",
        "text": """
Alex Johnson | alex@example.com
Senior Backend Engineer — 6 years experience

Skills: Python, FastAPI, Django, PostgreSQL, Redis, Docker, GitHub Actions, AWS (ECS/RDS/S3),
        OpenAI API, Azure OpenAI, Celery, WebSockets, prompt engineering.

Experience:
- TechCorp (3 yrs): Built FastAPI microservices handling 50k req/day; integrated GPT-4 for
  resume screening; managed PostgreSQL schemas and Redis caching layers.
- StartupXYZ (3 yrs): Led backend for real-time WebRTC interview platform; Celery task queues
  for bulk email; CI/CD with GitHub Actions; deployed on AWS ECS.

Education: B.Tech Computer Science.
""",
    },
    {
        "id": "R2_good_match",
        "text": """
Priya Sharma | priya@example.com
Backend Developer — 5 years experience

Skills: Python, Django REST Framework, MySQL, Docker, Jenkins, AWS EC2.

Experience:
- MedApp (5 yrs): REST APIs with Django; MySQL database design; Dockerised deployments on EC2;
  basic Jenkins pipelines. No direct LLM experience but took an online course on prompt engineering.

Education: M.Sc. Information Technology.
""",
    },
    {
        "id": "R3_partial_match",
        "text": """
Carlos Rivera | carlos@example.com
Full Stack Developer — 4 years experience

Skills: Node.js, Express, React, MongoDB, PostgreSQL (basic), Docker.

Experience:
- AgencyABC (4 yrs): REST APIs in Node/Express; React frontend; MongoDB primary datastore;
  some PostgreSQL for reporting queries. No Python or LLM experience.

Education: B.Sc. Software Engineering.
""",
    },
    {
        "id": "R4_weak_match",
        "text": """
Emily Chen | emily@example.com
Data Analyst — 3 years experience

Skills: Python (pandas, numpy, matplotlib), SQL, Tableau, Excel, R.

Experience:
- DataCo (3 yrs): Built dashboards in Tableau; wrote ad-hoc SQL queries; automated reports
  with Python scripts. No backend API or cloud deployment experience.

Education: B.Sc. Statistics.
""",
    },
    {
        "id": "R5_no_match",
        "text": """
James Lee | james@example.com
Graphic Designer — 7 years experience

Skills: Adobe Photoshop, Illustrator, InDesign, Figma, After Effects, brand identity.

Experience:
- CreativeStudio (7 yrs): Designed brand identities, marketing collateral, and UI mockups
  for clients across retail and hospitality. No programming or software development experience.

Education: B.A. Visual Communication.
""",
    },
]

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert technical recruiter. "
    "Evaluate how well the candidate's resume matches the job description. "
    "Respond ONLY with a valid JSON object — no markdown, no explanation outside the JSON."
)

USER_PROMPT_TEMPLATE = """Job Description:
{jd}

Candidate Resume:
{resume}

Return this exact JSON structure:
{{
  "relevance_score": <integer 0-100>
}}"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
class ResumeResult:
    def __init__(self, resume_id, success, latency_ms, score=None, error="", tokens=0):
        self.resume_id  = resume_id
        self.success    = success
        self.latency_ms = latency_ms
        self.score      = score
        self.error      = error
        self.tokens     = tokens


# ---------------------------------------------------------------------------
# Single request
# ---------------------------------------------------------------------------
async def single_request(idx: int, resume: dict, total: int) -> ResumeResult:
    prompt = USER_PROMPT_TEMPLATE.format(jd=JOB_DESCRIPTION, resume=resume["text"])
    start  = time.perf_counter()
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            seed=0,
            max_tokens=2000,   # reasoning model needs budget to finish chain-of-thought before answering
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {
                        "relevance_score": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                        },
                    },
                    "required": ["relevance_score"],
                },
            },
        )
        latency_ms = (time.perf_counter() - start) * 1000
        tokens     = response.usage.completion_tokens if response.usage else 0
        raw        = response.choices[0].message.content

        data  = json.loads(raw)
        score = int(data.get("relevance_score", -1))

        print(
            f"  [{idx+1:>3}/{total}] {resume['id']:<22} "
            f"score={score:>3}  {latency_ms:>8.1f}ms  OK"
        )
        return ResumeResult(resume_id=resume["id"], success=True, latency_ms=latency_ms,
                            score=score, tokens=tokens)

    except json.JSONDecodeError as e:
        latency_ms = (time.perf_counter() - start) * 1000
        raw        = locals().get("raw", "<no content>")
        finish     = response.choices[0].finish_reason if "response" in locals() else "?"
        print(f"  [{idx+1:>3}/{total}] {resume['id']:<22} JSON_ERR  {latency_ms:>8.1f}ms")
        print(f"    error       : {e}")
        print(f"    finish      : {finish}")
        print(f"    raw content : {repr(raw)}")
        return ResumeResult(resume_id=resume["id"], success=False, latency_ms=latency_ms,
                            error=f"JSONDecodeError: {e}")

    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        print(f"  [{idx+1:>3}/{total}] {resume['id']:<22} FAIL      {latency_ms:>8.1f}ms")
        print(f"    error : {e}")
        return ResumeResult(resume_id=resume["id"], success=False, latency_ms=latency_ms,
                            error=str(e))


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------
async def run_load_test(total_requests: int, concurrency: int, consistency_rounds: int):
    if consistency_rounds > 1:
        tasks_resumes = [r for r in RESUMES for _ in range(consistency_rounds)]
    else:
        tasks_resumes = [RESUMES[i % len(RESUMES)] for i in range(total_requests)]

    total     = len(tasks_resumes)
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(idx, resume):
        async with semaphore:
            return await single_request(idx, resume, total)

    print(f"\nResume-JD Relevance Load Test")
    print(f"Endpoint    : {BASE_URL}")
    print(f"Model       : {MODEL}")
    print(f"Requests    : {total}  |  Concurrency: {concurrency}  |  Consistency rounds: {consistency_rounds}")
    print("-" * 75)
    print(f"  {'#':<6} {'Resume ID':<22} {'Score':<8} {'Latency':>10}  Status")
    print("-" * 75)

    wall_start = time.perf_counter()
    results    = await asyncio.gather(*[bounded(i, r) for i, r in enumerate(tasks_resumes)])
    wall_ms    = (time.perf_counter() - wall_start) * 1000

    successes = [r for r in results if r.success]
    failures  = [r for r in results if not r.success]
    latencies = [r.latency_ms for r in successes]
    total_tok = sum(r.tokens for r in successes)

    print("-" * 75)
    print(f"\n{'='*75}")
    print(f" SUMMARY")
    print(f"{'='*75}")
    print(f"  Requests   : {total}  |  Succeeded: {len(successes)}  |  Failed: {len(failures)}")
    print(f"  Wall time  : {wall_ms/1000:.2f}s  |  Throughput: {len(successes)/(wall_ms/1000):.2f} req/s")

    if latencies:
        sorted_lat = sorted(latencies)
        p95_idx    = max(0, int(len(latencies) * 0.95) - 1)
        print(f"\n  Latency (ms)")
        print(f"    Min    : {min(latencies):.1f}")
        print(f"    Median : {statistics.median(latencies):.1f}")
        print(f"    P95    : {sorted_lat[p95_idx]:.1f}")
        print(f"    Max    : {max(latencies):.1f}")
        if len(latencies) > 1:
            print(f"    StdDev : {statistics.stdev(latencies):.1f}")

    if total_tok:
        print(f"\n  Tokens")
        print(f"    Total generated : {total_tok}")
        print(f"    Tokens/sec      : {total_tok/(wall_ms/1000):.1f}")

    print(f"\n  Score breakdown (successful only)")
    print(f"  {'Resume ID':<24} {'Runs':>4}  {'Avg':>5}  {'Min':>4}  {'Max':>4}  {'StdDev':>7}  {'Scores'}")
    print(f"  {'-'*24}  {'-'*4}  {'-'*5}  {'-'*4}  {'-'*4}  {'-'*7}  {'-'*20}")

    resume_groups = {}
    for r in successes:
        resume_groups.setdefault(r.resume_id, []).append(r.score)

    for rid in [r["id"] for r in RESUMES]:
        scores = resume_groups.get(rid, [])
        if not scores:
            print(f"  {rid:<24}  {'—':>4}")
            continue
        avg    = statistics.mean(scores)
        stddev = statistics.stdev(scores) if len(scores) > 1 else 0.0
        print(
            f"  {rid:<24}  {len(scores):>4}  {avg:>5.1f}  {min(scores):>4}  "
            f"{max(scores):>4}  {stddev:>7.2f}  [{', '.join(str(s) for s in scores)}]"
        )

    if consistency_rounds > 1:
        print(f"\n  Consistency verdict (StdDev thresholds: <5 excellent, <10 good, ≥10 poor)")
        for rid in [r["id"] for r in RESUMES]:
            scores = resume_groups.get(rid, [])
            if len(scores) < 2:
                continue
            sd      = statistics.stdev(scores)
            verdict = "excellent" if sd < 5 else ("good" if sd < 10 else "POOR")
            print(f"    {rid:<24}  stdev={sd:.2f}  → {verdict}")

    if failures:
        print(f"\n  Errors ({len(failures)})")
        for r in failures[:5]:
            print(f"    {r.resume_id}: {r.error}")

    print()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume-JD relevance load tester")
    parser.add_argument("--requests",           type=int, default=25,
                        help="Total requests when not using --consistency-rounds (default: 25)")
    parser.add_argument("--concurrency",        type=int, default=5,
                        help="Max concurrent requests (default: 5)")
    parser.add_argument("--consistency-rounds", type=int, default=3,
                        help="Repeat each of the 5 resumes N times to measure score drift (default: 3)")
    args = parser.parse_args()

    asyncio.run(run_load_test(
        total_requests=args.requests,
        concurrency=args.concurrency,
        consistency_rounds=args.consistency_rounds,
    ))
