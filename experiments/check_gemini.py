"""Preflight check for Gemini before research experiments.

Never prints GEMINI_API_KEY. Verifies that the key is discoverable and that
at least one configured Gemini model can return a short response.
"""

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND = PROJECT_ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.ai import gemini_service


def main() -> int:
    env_path = PROJECT_ROOT / ".env"
    key = os.getenv("GEMINI_API_KEY")

    # gemini_service loads dotenv during import; re-read the environment after it.
    key = os.getenv("GEMINI_API_KEY")
    print("=== Gemini experiment preflight ===")
    print("Project root:", PROJECT_ROOT)
    print(".env exists:", env_path.exists())
    print("GEMINI_API_KEY available:", bool(key and key.strip() and key.strip() != "your_gemini_api_key_here"))
    print("Configured models:", ", ".join(gemini_service.DEFAULT_MODELS))

    if not key or not key.strip() or key.strip() == "your_gemini_api_key_here":
        print("\nFAIL: Gemini key is not available to the experiment process.")
        print("Add GEMINI_API_KEY=... to the project-root .env, then rerun this check.")
        return 2

    t0 = time.perf_counter()
    result = gemini_service._generate_with_fallback(
        prompt='Return exactly this JSON: {"ok": true}',
        config={"temperature": 0.0, "response_mime_type": "application/json"},
    )
    elapsed = (time.perf_counter() - t0) * 1000.0

    print("Response received:", bool(result))
    print("Latency ms:", round(elapsed, 2))
    if result:
        print("PASS: Gemini is reachable. Do not print or share your API key.")
        return 0

    print("\nFAIL: a key was found, but all configured Gemini model calls returned no response.")
    print("Check quota/model access/network. Do not run the full LLM-only benchmark yet.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
