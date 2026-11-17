"""
llm.py - one small client that talks to any OpenAI-compatible LLM endpoint.

Supported out of the box (all speak the OpenAI Chat Completions + tool-calling format):
  * Groq          - free API key, very fast   (https://console.groq.com/keys)
  * Google Gemini - free API key              (https://aistudio.google.com/apikey)
  * Ollama        - 100% offline on your laptop (https://ollama.com)
  * OpenAI        - paid key
  * Custom        - any other OpenAI-compatible server (LM Studio, vLLM, OpenRouter...)
If no LLM is configured the app still runs in "Offline mode": every agent falls back to
deterministic templates so a demo never breaks, and the UI says so clearly.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

PROVIDERS = {
    "Groq (free key, fastest)": dict(base_url="https://api.groq.com/openai/v1", env="GROQ_API_KEY",
                                     model="openai/gpt-oss-120b", key_url="https://console.groq.com/keys"),
    "Google Gemini (free key)": dict(base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                                     env="GEMINI_API_KEY", model="gemini-flash-latest",
                                     key_url="https://aistudio.google.com/apikey"),
    "Ollama (offline, local)": dict(base_url="http://localhost:11434/v1", env="", model="qwen2.5:7b",
                                    key_url="https://ollama.com/download"),
    "OpenAI": dict(base_url="https://api.openai.com/v1", env="OPENAI_API_KEY", model="gpt-4.1-mini",
                   key_url="https://platform.openai.com/api-keys"),
    "Custom OpenAI-compatible": dict(base_url="http://localhost:1234/v1", env="CUSTOM_LLM_API_KEY",
                                     model="local-model", key_url=""),
    "Offline mode (no LLM)": dict(base_url="", env="", model="", key_url=""),
}
OFFLINE = "Offline mode (no LLM)"


def load_dotenv(path: Path | None = None) -> None:
    """Tiny .env reader (avoids an extra dependency)."""
    path = path or Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def default_provider() -> str:
    load_dotenv()
    p = os.getenv("SEISMIC_LLM_PROVIDER", "")
    for name in PROVIDERS:
        if p and p.lower() in name.lower():
            return name
    for name, cfg in PROVIDERS.items():
        if cfg["env"] and os.getenv(cfg["env"]):
            return name
    return OFFLINE


@dataclass
class LLM:
    provider: str = OFFLINE
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.2
    calls: int = field(default=0, init=False)
    last_error: str = field(default="", init=False)

    def __post_init__(self):
        cfg = PROVIDERS.get(self.provider, PROVIDERS[OFFLINE])
        self.base_url = self.base_url or cfg["base_url"]
        self.model = self.model or os.getenv("SEISMIC_LLM_MODEL", "") or cfg["model"]
        if not self.api_key:
            self.api_key = os.getenv(cfg["env"], "") if cfg["env"] else ""
        if self.provider.startswith("Ollama") and not self.api_key:
            self.api_key = "ollama"
        self._client = None

    # ------------------------------------------------------------------
    @property
    def online(self) -> bool:
        return self.provider != OFFLINE and bool(self.base_url) and bool(self.api_key or "localhost" in self.base_url)

    @property
    def label(self) -> str:
        return f"{self.provider.split(' (')[0]} · {self.model}" if self.online else "Offline mode"

    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key or "none", base_url=self.base_url, timeout=90, max_retries=2)
        return self._client

    def list_models(self) -> list[str]:
        return sorted(m.id for m in self.client().models.list())

    def chat(self, messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 1200):
        """Returns the assistant message object (OpenAI SDK type)."""
        kw = dict(model=self.model, messages=messages, temperature=self.temperature, max_tokens=max_tokens)
        if tools:
            kw["tools"] = tools
            kw["tool_choice"] = "auto"
        self.calls += 1
        resp = self.client().chat.completions.create(**kw)
        return resp.choices[0].message

    def text(self, system: str, user: str, max_tokens: int = 900) -> str:
        msg = self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=max_tokens)
        return (msg.content or "").strip()

    def json(self, system: str, user: str, max_tokens: int = 1200) -> dict:
        """Ask for JSON and parse it robustly (works with every provider, no special mode needed)."""
        raw = self.text(system + "\nReply with ONE valid JSON object only. No markdown fences.", user, max_tokens)
        return parse_json(raw)

    def ping(self) -> str:
        return self.text("You are a test.", "Reply with the single word: ready", max_tokens=20)


def parse_json(raw: str) -> dict:
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {"_unparsed": raw}
