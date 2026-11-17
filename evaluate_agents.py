"""Evaluate the full agent council (ML classifier + LLM Seismologist) on held-out test events.

    python evaluate_agents.py                      # uses the LLM from .env (or offline templates)
    python evaluate_agents.py --provider groq --n 40
"""
import argparse

from seismicsense.agents import Toolkit, evaluate_council
from seismicsense.llm import LLM, PROVIDERS, default_provider

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="", help="groq | gemini | ollama | openai | offline")
    ap.add_argument("--model", default="")
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    name = next((p for p in PROVIDERS if a.provider and a.provider.lower() in p.lower()), default_provider())
    llm = LLM(provider=name, model=a.model)
    print(f"LLM: {llm.label}")
    df = evaluate_council(Toolkit(), llm, n=a.n, progress=lambda f: print(f"\r{f * 100:5.1f}%", end=""))
    print()
    print(df.to_string(index=False))
    print(f"\nClassifier accuracy:       {(df.classifier == df.nasa).mean():.3f}")
    print(f"LLM Seismologist accuracy: {(df.seismologist == df.nasa).mean():.3f}")
    print(f"Council (final) accuracy:  {(df.council == df.nasa).mean():.3f}")
    print(f"LLM agrees with classifier:{(df.seismologist == df.classifier).mean():.3f}")
    if llm.last_error:
        print("Last LLM error:", llm.last_error)
    df.to_csv("reports/council_evaluation.csv", index=False)
    print("Saved reports/council_evaluation.csv")
