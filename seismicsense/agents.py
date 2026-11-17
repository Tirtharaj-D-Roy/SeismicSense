"""
agents.py - SeismicSense's cooperating agents.

Classification "council" (runs for every event you inspect):
  🛰️ Data Scout          tool   - pulls the raw record from the NASA catalog
  📜 Mission Historian   tool   - checks Apollo mission logs for rocket/lander crashes at that time
  📈 Signal Analyst      tool   - turns numbers into evidence (strength, duration, tides, repeats)
  🤖 Pattern Classifier  ML     - gradient-boosted model trained on 7,336 labelled events
  🧑‍🔬 Seismologist       LLM    - cross-examines all evidence, gives the verdict + plain-English story
  🗺️ Cartographer        tool   - places the event on the Moon map

Mission Control (chat): an LLM agent that plans and calls tools (search the catalog, analyse
an event, draw maps/timelines, report accuracy) to answer any question in plain English.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import catalog as C
from . import model as M
from . import viz
from .llm import LLM

LABELS = ["deep", "meteoroid", "shallow", "artificial"]


# ====================================================================== shared state
class Toolkit:
    """Everything the agents can use. Built once and cached by the app."""

    def __init__(self):
        self.cat = C.load_catalog()
        self.X = C.build_features(self.cat)
        try:
            self.bundle = M.load_model()
            _ = self.bundle["model"].predict_proba(self.X[self.bundle["features"]].head(2))
        except Exception:  # missing file or scikit-learn version mismatch -> retrain (~1 min)
            M.train(verbose=False)
            self.bundle = M.load_model()
        self.model, self.features = self.bundle["model"], self.bundle["features"]
        self.classes = list(self.model.classes_)
        self.metrics = json.loads(M.METRICS_PATH.read_text()) if M.METRICS_PATH.exists() else {}
        self.locs = C.load_locations()
        self.proba = pd.DataFrame(self.model.predict_proba(self.X[self.features]), columns=self.classes, index=self.cat.index)
        self.historian = M.mission_historian(self.cat["time"])
        self.ev = self._events_table()

    def _events_table(self) -> pd.DataFrame:
        cat, X = self.cat, self.X
        ev = cat[["event_id", "time", "nasa_label", "nest", "comment"]].copy()
        ev["ai_ml"] = self.proba.idxmax(axis=1)
        ev["confidence"] = self.proba.max(axis=1)
        ev["label"] = np.where(self.historian.notna(), "artificial", ev["ai_ml"])
        ev.loc[self.historian.notna(), "confidence"] = 1.0
        amps = X[[f"amp_{s}" for s in C.STATIONS]]
        ev["strength"] = amps.max(axis=1).fillna(0.1).clip(lower=0.1)
        ev["duration_min"] = X["duration_min"]
        ev["n_stations"] = X["n_stations"]
        ev["earth_moon_km"] = X["earth_moon_km"]
        ev["days_since_perigee"] = (np.degrees(np.arctan2(X["anomalistic_sin"], X["anomalistic_cos"])) % 360) / 360 * 27.5546
        test = set(self.bundle.get("test_ids", []))
        ev["split"] = np.where(ev.event_id.isin(test), "test",
                               np.where(ev.nasa_label.isin(C.NATURAL), "train",
                                        np.where(ev.nasa_label == "artificial", "artificial", "mystery")))
        ev["year"] = ev.time.dt.year
        ev["strength_pct"] = ev["strength"].rank(pct=True)
        ev["duration_pct"] = ev["duration_min"].rank(pct=True)
        return ev

    # ------------------------------------------------------------------ tools
    def row(self, event_id: str) -> int:
        event_id = event_id.strip().upper()
        if re.fullmatch(r"\d+", event_id):
            event_id = f"EV-{int(event_id):05d}"
        hits = self.ev.index[self.ev.event_id == event_id]
        if len(hits) == 0:
            raise KeyError(f"No event called {event_id}. IDs look like EV-00042 (1 to {len(self.ev)}).")
        return hits[0]

    def classifier_probs(self, i: int) -> dict:
        return {c: float(self.proba.loc[i, c]) for c in self.classes}

    def evidence(self, i: int) -> dict:
        """Signal Analyst: plain-English evidence for one event."""
        e, x = self.ev.loc[i], self.X.loc[i]
        felt = {st: (None if pd.isna(x[f"amp_{st}"]) else float(x[f"amp_{st}"])) for st in C.STATIONS}
        operating = [st for st in C.STATIONS if x[f"operating_{st}"] == 1]
        heard = [st for st in C.STATIONS if x[f"seen_{st}"] == 1]
        dsp = float(e.days_since_perigee)
        repeat = float(x["recur_best"])
        facts = {
            "date_utc": e.time.strftime("%d %B %Y, %H:%M"),
            "stations_operating": [f"Apollo {s}" for s in operating],
            "stations_that_felt_it": [f"Apollo {s}" for s in heard],
            "shaking_strength_by_station": {f"Apollo {k}": v for k, v in felt.items() if v is not None},
            "strength_vs_all_events": f"stronger than {e.strength_pct * 100:.0f}% of all recorded events",
            "duration_minutes": None if pd.isna(e.duration_min) else int(e.duration_min),
            "duration_vs_all_events": None if pd.isna(e.duration_min) else f"longer than {e.duration_pct * 100:.0f}% of events",
            "earth_moon_distance_km": int(e.earth_moon_km),
            "days_since_moon_closest_to_earth": round(dsp, 1),
            "repeat_signature": ("very strong - an almost identical signal occurred ~1-3 lunar months before/after"
                                 if repeat < 0.12 else "moderate - a similar signal occurred 1-3 lunar months away"
                                 if repeat < 0.3 else "none - no look-alike signal 1-3 lunar months away"),
            "repeat_score (lower = closer twin)": round(repeat, 3),
            "other_events_within_24h": int(x["neighbors_24h"]),
            "signal_quality_grade": {3: "A (clear)", 2: "B", 1: "C (faint)"}.get(x["grade"], "not graded"),
        }
        return {"facts": facts, "felt": felt}

    def known_source(self, i: int) -> dict | None:
        """Published location of this exact event, if NASA scientists located it."""
        e = self.ev.loc[i]
        if pd.notna(self.historian.loc[i]):
            a = self.locs["artificial"].set_index("name").loc[self.historian.loc[i]]
            return dict(lat=a.lat, lon=a.lon, label=self.historian.loc[i])
        if e.nasa_label == "deep" and pd.notna(e.nest):
            dm = self.locs["deep_nests"]
            m = dm[dm.nest == int(e.nest)]
            if len(m):
                r = m.iloc[0]
                return dict(lat=r.lat, lon=r.lon, label=f"Nest A{int(e.nest)} · {int(r.depth_km)} km deep")
        for key, name in (("shallow", "Shallow moonquake epicentre"), ("meteoroid", "Meteoroid impact point")):
            d = self.locs[key]
            dt = ((d.time - e.time) / pd.Timedelta("1min")).abs()
            if len(dt) and dt.min() < 90:
                r = d.loc[dt.idxmin()]
                return dict(lat=r.lat, lon=r.lon, label=name)
        return None

    def nest_counts(self) -> pd.Series:
        return self.cat.loc[self.cat.nasa_label == "deep", "nest"].value_counts()


# ====================================================================== council pipeline
@dataclass
class Step:
    agent: str
    emoji: str
    kind: str          # "tool" | "ML" | "LLM"
    text: str
    seconds: float = 0.0
    data: dict = field(default_factory=dict)


SEISMOLOGIST_SYSTEM = """You are Dr. Luna, a planetary seismologist on the SeismicSense agent team.
You review evidence about ONE seismic event recorded by the Apollo seismometers (1969-1977) and give
the team's final verdict. Domain knowledge:
- DEEP MOONQUAKE (deep): 700-1,200 km underground, caused by Earth's tides squeezing the Moon. Usually weak,
  short-to-moderate duration, come from fixed 'nests' that fire again and again, so a near-identical signal
  often appears 1-3 lunar months (27.55 days) earlier/later. More common at certain days of the orbit.
- METEOROID IMPACT (meteoroid): a space rock hits the surface. Random timing, no repeating twin, often
  stronger and longer-ringing than deep moonquakes; usually strongest at the station nearest the impact.
- SHALLOW MOONQUAKE (shallow): extremely rare (28 in 8 years), the strongest natural events, long duration,
  usually felt by most operating stations.
- ARTIFICIAL IMPACT (artificial): ONLY when the Mission Historian matched a known Apollo crash time.
The Pattern Classifier (machine learning) is ~90% accurate overall: very reliable for deep moonquakes,
~70% for meteoroids, unreliable for shallow moonquakes (too few examples). Weigh it, but think for yourself.
Write for a curious 12-year-old: short sentences, no jargon, vivid but accurate. Never invent numbers."""

SEISMOLOGIST_TASK = """EVIDENCE:
{evidence}

MISSION HISTORIAN: {historian}
PATTERN CLASSIFIER PROBABILITIES: {probs}

Return JSON with exactly these keys:
{{"verdict": "deep|meteoroid|shallow|artificial",
  "agree_with_classifier": true/false,
  "confidence": "high|medium|low",
  "headline": "max 12 words, fun but accurate",
  "key_reasons": ["3 short plain-English reasons, each citing a specific piece of evidence"],
  "story": "3-4 sentences explaining what happened on the Moon, for a non-scientist",
  "caveat": "one sentence about what could make this verdict wrong"}}"""


def run_council(tk: Toolkit, event_id: str, llm: LLM, allow_override: bool = False, on_step=None) -> dict:
    """Run every agent on one event. on_step(step) is called live so the UI can stream progress."""
    steps: list[Step] = []

    def emit(s: Step):
        steps.append(s)
        if on_step:
            on_step(s)

    i = tk.row(event_id)
    e = tk.ev.loc[i]

    t = time.time()
    raw = tk.cat.loc[i, ["Y", "JD", "S", "E", "A1", "A2", "A3", "A4", "Grade"]].fillna("-").to_dict()
    emit(Step("Data Scout", "🛰️", "tool",
              f"Fetched **{e.event_id}** from the NASA Apollo catalog - recorded {e.time:%d %b %Y at %H:%M} UTC. "
              f"Raw record: year 19{raw['Y']}, day {raw['JD']}, start {raw['S']}, end {raw['E']}, "
              f"amplitudes at stations 12/14/15/16 = {raw['A1']}/{raw['A2']}/{raw['A3']}/{raw['A4']}.",
              time.time() - t, raw))

    t = time.time()
    hist = tk.historian.loc[i]
    htxt = (f"Match! At this exact time NASA deliberately crashed the **{hist}** into the Moon."
            if pd.notna(hist) else "No Apollo rocket or lander crash happened at this time, so this was natural.")
    emit(Step("Mission Historian", "📜", "tool", htxt, time.time() - t, {"match": None if pd.isna(hist) else hist}))

    t = time.time()
    ev = tk.evidence(i)
    f = ev["facts"]
    emit(Step("Signal Analyst", "📈", "tool",
              f"Felt by {len(f['stations_that_felt_it'])} of {len(f['stations_operating'])} working stations; "
              f"{f['strength_vs_all_events']}; lasted {f['duration_minutes'] or '?'} min. "
              f"Repeating signature: {f['repeat_signature'].split(' - ')[0]}. "
              f"Moon was {f['earth_moon_distance_km']:,} km from Earth ({f['days_since_moon_closest_to_earth']} days after closest approach).",
              time.time() - t, f))

    t = time.time()
    probs = tk.classifier_probs(i)
    ml = max(probs, key=probs.get)
    emit(Step("Pattern Classifier", "🤖", "ML",
              f"Machine-learning model says **{C.CLASS_INFO[ml]['name']}** with {probs[ml] * 100:.0f}% confidence "
              f"(" + ", ".join(f"{C.CLASS_INFO[k]['name']} {v * 100:.0f}%" for k, v in sorted(probs.items(), key=lambda kv: -kv[1])) + ").",
              time.time() - t, probs))

    t = time.time()
    review = seismologist(llm, f, hist, probs)
    verdict = review.get("verdict") if review.get("verdict") in LABELS else ml
    emit(Step("Seismologist", "🧑‍🔬", "LLM" if review.get("_source") == "llm" else "tool",
              f"**{review.get('headline', '')}** - verdict: {C.CLASS_INFO[verdict]['name']} "
              f"({review.get('confidence', '?')} confidence, " + ("overrides the classifier, which never learned rocket crashes" if verdict == "artificial" and pd.notna(hist)
                 else "agrees with the classifier" if verdict == ml else "DISAGREES with the classifier") + ").",
              time.time() - t, review))

    # ---- decision policy
    if pd.notna(hist):
        final, why = "artificial", "Mission records are definitive."
    elif allow_override and verdict != ml and probs[ml] < 0.60 and verdict != "artificial":
        final, why = verdict, f"Classifier was unsure ({probs[ml] * 100:.0f}%), so the Seismologist's judgement was used."
    else:
        final, why = ml, ("Classifier and Seismologist agree." if verdict == ml
                          else "Seismologist raised a doubt - shown as a second opinion; the classifier's answer is kept.")

    t = time.time()
    src = tk.known_source(i)
    fig = viz.moon_map(tk.locs, tk.nest_counts(), layers=(), highlight=src,
                       felt={k: v for k, v in ev["felt"].items()}, height=470,
                       rotation=((src or {}).get("lon", -5), 0))
    emit(Step("Cartographer", "🗺️", "tool",
              ("Pinned the published source location: " + src["label"] + ".") if src else
              "No published location for this event (the 4-station network could only locate a small fraction). "
              "Showing how strongly each station felt it instead - the biggest ring is usually closest to the source.",
              time.time() - t))

    return dict(event_id=e.event_id, time=e.time, final=final, decision=why, ml=ml, probs=probs, review=review,
                nasa=e.nasa_label, split=e.split, steps=steps, map=fig, evidence=f, source=src)


def seismologist(llm: LLM, facts: dict, hist, probs: dict) -> dict:
    if llm.online:
        try:
            out = llm.json(SEISMOLOGIST_SYSTEM, SEISMOLOGIST_TASK.format(
                evidence=json.dumps(facts, indent=1), historian=hist if pd.notna(hist) else "no match",
                probs=json.dumps({k: round(v, 3) for k, v in probs.items()})))
            if "_unparsed" not in out:
                out["verdict"] = str(out.get("verdict", "")).lower().strip()
                out["_source"] = "llm"
                return out
            llm.last_error = "LLM reply was not valid JSON"
        except Exception as ex:  # network / key / model errors -> graceful fallback
            llm.last_error = str(ex)[:300]
    return _template_review(facts, hist, probs)


def _template_review(facts: dict, hist, probs: dict) -> dict:
    """Offline fallback: deterministic explanation built from the same evidence."""
    ml = "artificial" if pd.notna(hist) else max(probs, key=probs.get)
    info = C.CLASS_INFO[ml]
    p = 1.0 if ml == "artificial" else probs[ml]
    reasons = []
    if ml == "artificial":
        reasons.append(f"Apollo records show the {hist} crashed into the Moon at this exact time.")
    rep = facts["repeat_signature"]
    if rep.startswith("very strong") or rep.startswith("moderate"):
        reasons.append("A look-alike signal appeared about a lunar month away - typical of a repeating deep source.")
    else:
        reasons.append("No look-alike signal a lunar month away - one-off events like impacts behave this way.")
    reasons.append(f"Shaking was {facts['strength_vs_all_events']} and lasted {facts['duration_minutes'] or 'an unknown number of'} minutes.")
    reasons.append(f"{len(facts['stations_that_felt_it'])} of {len(facts['stations_operating'])} working seismometers felt it.")
    return {"verdict": ml, "agree_with_classifier": True, "confidence": "high" if p > 0.85 else "medium" if p > 0.6 else "low",
            "headline": f"{info['emoji']} {info['short']}", "key_reasons": reasons[:3], "story": info["story"],
            "caveat": "Offline mode: this explanation is template-based. Connect an LLM for a full AI review.",
            "_source": "template"}


# ====================================================================== Mission Control (chat agent)
CHAT_SYSTEM = """You are Mission Control, the lead agent of SeismicSense - an AI team that studies the
13,057 seismic events recorded by Apollo seismometers on the Moon (1969-1977). You talk to people with
NO science background (e.g. a professor from another department), so explain simply and accurately.
You coordinate tools - ALWAYS use them for facts and numbers; never make data up.
Event types: deep (deep moonquake), meteoroid (meteoroid impact), shallow (shallow moonquake),
artificial (Apollo rocket/lander crash), unknown (never classified by NASA).
Event IDs look like EV-00042. When a chart/map would help, call a show_* tool - it appears under your answer.
Keep answers short (under 180 words), use **bold** for key numbers, and one fitting emoji at most."""

TOOLS = [
    {"type": "function", "function": {"name": "catalog_stats",
     "description": "Count events by type, optionally for one year and/or type. Also returns strongest event and busiest month.",
     "parameters": {"type": "object", "properties": {
         "year": {"type": "integer", "description": "1969-1977"},
         "event_type": {"type": "string", "enum": ["deep", "meteoroid", "shallow", "artificial", "unknown"]},
         "use": {"type": "string", "enum": ["nasa", "ai"], "description": "nasa = scientists' labels (default), ai = SeismicSense labels"}}}}},
    {"type": "function", "function": {"name": "find_events",
     "description": "Search the catalog and list matching events.",
     "parameters": {"type": "object", "properties": {
         "event_type": {"type": "string", "enum": ["deep", "meteoroid", "shallow", "artificial", "unknown"]},
         "year_from": {"type": "integer"}, "year_to": {"type": "integer"},
         "sort_by": {"type": "string", "enum": ["strongest", "longest", "oldest", "newest"]},
         "limit": {"type": "integer", "description": "max 15"}}}}},
    {"type": "function", "function": {"name": "analyze_event",
     "description": "Run the agent council on one event: evidence, mission check, ML probabilities, known location. Also shows its map.",
     "parameters": {"type": "object", "properties": {"event_id": {"type": "string"}}, "required": ["event_id"]}}},
    {"type": "function", "function": {"name": "show_moon_map",
     "description": "Show the interactive Moon globe with source locations.",
     "parameters": {"type": "object", "properties": {"layers": {"type": "array", "items": {"type": "string",
                    "enum": ["deep", "meteoroid", "shallow", "artificial"]}}}}}},
    {"type": "function", "function": {"name": "show_timeline",
     "description": "Show events over time (and monthly counts).",
     "parameters": {"type": "object", "properties": {"event_type": {"type": "string",
                    "enum": ["deep", "meteoroid", "shallow", "artificial", "unknown"]},
                    "year_from": {"type": "integer"}, "year_to": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "show_tidal_pattern",
     "description": "Show how deep moonquakes follow Earth's tides (27.55-day cycle) while impacts don't.",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "model_report",
     "description": "Accuracy, F1 score and per-class results of the SeismicSense classifier on held-out test data.",
     "parameters": {"type": "object", "properties": {}}}},
]


class MissionControl:
    def __init__(self, tk: Toolkit, llm: LLM):
        self.tk, self.llm = tk, llm

    # ---------------------------------------------------------- tool implementations
    def _filter(self, a: dict, col="nasa_label") -> pd.DataFrame:
        ev = self.tk.ev
        if a.get("event_type"):
            ev = ev[ev[col] == a["event_type"]]
        if a.get("year"):
            ev = ev[ev.year == int(a["year"])]
        if a.get("year_from"):
            ev = ev[ev.year >= int(a["year_from"])]
        if a.get("year_to"):
            ev = ev[ev.year <= int(a["year_to"])]
        return ev

    def call(self, name: str, a: dict, figs: list) -> dict:
        tk = self.tk
        if name == "catalog_stats":
            col = "label" if a.get("use") == "ai" else "nasa_label"
            ev = self._filter(a, col)
            if ev.empty:
                return {"count": 0}
            top = ev.loc[ev.strength.idxmax()]
            busiest = ev.groupby(ev.time.dt.to_period("M")).size()
            return {"total": int(len(ev)), "by_type": ev[col].value_counts().to_dict(),
                    "by_year": ev.year.value_counts().sort_index().to_dict(),
                    "strongest": {"id": top.event_id, "date": f"{top.time:%d %b %Y}", "type": top[col], "strength": float(top.strength)},
                    "busiest_month": f"{busiest.idxmax()} ({busiest.max()} events)", "labels_from": col}
        if name == "find_events":
            ev = self._filter(a)
            key = {"strongest": ("strength", False), "longest": ("duration_min", False),
                   "oldest": ("time", True), "newest": ("time", False)}[a.get("sort_by", "strongest")]
            ev = ev.sort_values(key[0], ascending=key[1]).head(min(int(a.get("limit", 8)), 15))
            return {"events": [{"id": r.event_id, "date": f"{r.time:%d %b %Y %H:%M}", "nasa_label": r.nasa_label,
                                "ai_label": r.label, "ai_confidence": round(float(r.confidence), 2),
                                "strength": float(r.strength), "duration_min": None if pd.isna(r.duration_min) else int(r.duration_min)}
                               for r in ev.itertuples()]}
        if name == "analyze_event":
            i = tk.row(a["event_id"])
            e = tk.ev.loc[i]
            src = tk.known_source(i)
            ev = tk.evidence(i)
            figs.append(("Where it was felt: " + e.event_id, viz.moon_map(tk.locs, tk.nest_counts(), layers=(), highlight=src,
                                                                         felt=ev["felt"], height=430)))
            return {"id": e.event_id, "evidence": ev["facts"], "mission_historian": tk.historian.loc[i] if pd.notna(tk.historian.loc[i]) else "no Apollo crash at this time",
                    "classifier_probabilities": {k: round(v, 3) for k, v in tk.classifier_probs(i).items()},
                    "seismicsense_label": e.label, "nasa_scientists_label": e.nasa_label,
                    "known_location": src["label"] if src else "not located"}
        if name == "show_moon_map":
            layers = a.get("layers") or ["deep", "meteoroid", "shallow", "artificial"]
            figs.append(("Moon map", viz.moon_map(tk.locs, tk.nest_counts(), layers=layers, height=520)))
            return {"shown": layers, "deep_nests_located": int(len(tk.locs["deep_nests"])),
                    "note": "Only events NASA could locate are on the map; most deep nests are on the near side because all stations were there."}
        if name == "show_timeline":
            ev = self._filter(a)
            figs.append(("Timeline", viz.timeline(ev, label_col="nasa_label")))
            return {"events_plotted": int(len(ev)), "per_year": ev.year.value_counts().sort_index().to_dict()}
        if name == "show_tidal_pattern":
            figs.append(("Tidal heartbeat", viz.tidal_heartbeat(tk.ev, label_col="nasa_label")))
            out = {}
            for c in ("deep", "meteoroid"):
                h, _ = np.histogram(tk.ev.loc[tk.ev.nasa_label == c, "days_since_perigee"], bins=9, range=(0, 27.5546))
                out[c] = {"lowest_bin_pct": round(100 * h.min() / h.sum(), 1), "highest_bin_pct": round(100 * h.max() / h.sum(), 1)}
            return {"percent_of_events_per_3day_window": out, "random_would_be_pct": 11.1}
        if name == "model_report":
            m = tk.metrics
            return {k: m.get(k) for k in ("test_accuracy", "test_f1_weighted", "test_f1_macro", "cv5_f1_weighted_mean",
                                          "n_train", "n_test", "per_class", "pipeline_4class")}
        raise ValueError(f"unknown tool {name}")

    # ---------------------------------------------------------- agent loop
    def ask(self, question: str, history: list[dict] | None = None) -> dict:
        figs, trace = [], []
        if not self.llm.online:
            return self._offline(question, figs, trace)
        msgs = [{"role": "system", "content": CHAT_SYSTEM}] + (history or [])[-8:] + [{"role": "user", "content": question}]
        try:
            for _ in range(6):
                msg = self.llm.chat(msgs, tools=TOOLS)
                calls = getattr(msg, "tool_calls", None) or []
                if not calls:
                    return dict(answer=(msg.content or "").strip() or "(no answer)", figs=figs, trace=trace, mode="llm")
                msgs.append({"role": "assistant", "content": msg.content,
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"}}
                                            for c in calls]})
                for c in calls:
                    try:
                        args = json.loads(c.function.arguments or "{}") or {}
                        res = self.call(c.function.name, args, figs)
                    except Exception as ex:
                        args, res = {}, {"error": str(ex)}
                    trace.append((c.function.name, args, res))
                    msgs.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(res, default=str)[:4000]})
            return dict(answer="I ran out of steps - try a more specific question.", figs=figs, trace=trace, mode="llm")
        except Exception as ex:
            self.llm.last_error = str(ex)[:300]
            out = self._offline(question, figs, trace)
            out["answer"] = f"⚠️ LLM unavailable ({self.llm.last_error[:120]}). Offline answer:\n\n" + out["answer"]
            return out

    def _offline(self, q: str, figs: list, trace: list) -> dict:
        """Keyword router used when no LLM is connected - keeps the demo working."""
        ql = q.lower()
        a = {}
        for t, words in {"deep": ["deep"], "meteoroid": ["meteor", "impact", "space rock"], "shallow": ["shallow"],
                         "artificial": ["artificial", "rocket", "crash", "lander", "s-ivb"]}.items():
            if any(w in ql for w in words):
                a["event_type"] = t
        yr = re.search(r"19(69|7[0-7])", ql)
        if yr:
            a["year"] = int(yr.group(0))
        idm = re.search(r"ev-?\s?(\d{1,5})", ql)
        if idm:
            r = self.call("analyze_event", {"event_id": idm.group(1)}, figs)
            trace.append(("analyze_event", {"event_id": idm.group(1)}, r))
            ans = (f"**{r['id']}** - SeismicSense says **{C.CLASS_INFO[r['seismicsense_label']]['name']}** "
                   f"(NASA scientists: {C.CLASS_INFO[r['nasa_scientists_label']]['name']}). "
                   f"Evidence: {r['evidence']['strength_vs_all_events']}, lasted {r['evidence']['duration_minutes'] or 'an unknown number of'} min, "
                   f"repeating signature {r['evidence']['repeat_signature'].split(' - ')[0]}. Location: {r['known_location']}.")
        elif any(w in ql for w in ["accura", "f1", "score", "good is", "performance"]):
            r = self.call("model_report", {}, figs)
            trace.append(("model_report", {}, r))
            ans = (f"On **{r['n_test']:,}** events the AI never saw, accuracy is **{r['test_accuracy'] * 100:.1f}%** "
                   f"and weighted F1 is **{r['test_f1_weighted']:.3f}**.")
        elif any(w in ql for w in ["tide", "tidal", "earth's pull", "gravity", "why"]):
            r = self.call("show_tidal_pattern", {}, figs)
            trace.append(("show_tidal_pattern", {}, r))
            ans = ("Deep moonquakes follow a **27.55-day rhythm**: Earth's gravity squeezes the Moon as its distance changes. "
                   f"Some 3-day windows get {r['percent_of_events_per_3day_window']['deep']['highest_bin_pct']}% of deep quakes, "
                   f"others only {r['percent_of_events_per_3day_window']['deep']['lowest_bin_pct']}%. Impacts stay flat - space rocks don't care about tides.")
        elif any(w in ql for w in ["map", "where", "location", "globe"]):
            r = self.call("show_moon_map", {"layers": [a["event_type"]] if a.get("event_type") else []}, figs)
            trace.append(("show_moon_map", a, r))
            ans = "Here's the Moon map 🌕 - drag to spin it. " + r["note"]
        elif any(w in ql for w in ["timeline", "over time", "when", "trend"]):
            r = self.call("show_timeline", a, figs)
            trace.append(("show_timeline", a, r))
            ans = f"Plotted **{r['events_plotted']:,}** events over time."
        elif any(w in ql for w in ["strongest", "biggest", "largest", "list", "show me", "find"]):
            r = self.call("find_events", {**a, "sort_by": "strongest", "limit": 5}, figs)
            trace.append(("find_events", a, r))
            ans = "Strongest matching events:\n" + "\n".join(
                f"- **{e['id']}** · {e['date']} · {C.CLASS_INFO[e['nasa_label']]['name']} · strength {e['strength']:g}" for e in r["events"])
        else:
            r = self.call("catalog_stats", a, figs)
            trace.append(("catalog_stats", a, r))
            if not r.get("total"):
                ans = "No events match that."
            else:
                ans = (f"**{r['total']:,}** events match. By type: " +
                       ", ".join(f"{C.CLASS_INFO[k]['name']} {v:,}" for k, v in r["by_type"].items()) +
                       f". Strongest: {r['strongest']['id']} on {r['strongest']['date']}. Busiest month: {r['busiest_month']}.")
        return dict(answer=ans + "\n\n_Offline mode - connect an LLM in the sidebar for full conversational answers._",
                    figs=figs, trace=trace, mode="offline")


# ====================================================================== LLM reports
def year_briefing(tk: Toolkit, llm: LLM, year: int) -> str:
    ev = tk.ev[tk.ev.year == year]
    stats = {"year": year, "events": int(len(ev)), "by_type_nasa": ev.nasa_label.value_counts().to_dict(),
             "mystery_events_ai_labels": ev[ev.split == "mystery"].label.value_counts().to_dict(),
             "strongest": ev.nlargest(3, "strength")[["event_id", "time", "nasa_label", "strength"]].astype(str).to_dict("records"),
             "artificial_impacts": tk.historian[ev.index].dropna().tolist(),
             "stations_working": [f"Apollo {s}" for s in C.STATIONS if pd.Timestamp(C.STATION_START[s]).year <= year]}
    if llm.online:
        try:
            return llm.text("You are the SeismicSense science communicator. Write a lively 'Moon news' briefing "
                            "(120-170 words, 3 short paragraphs, markdown, max 2 emojis) for non-scientists using ONLY the given stats.",
                            json.dumps(stats, default=str), max_tokens=700)
        except Exception as ex:
            llm.last_error = str(ex)[:300]
    bt = stats["by_type_nasa"]
    return (f"**{year} on the Moon:** the seismometers logged **{stats['events']:,} events** - "
            f"{bt.get('deep', 0):,} deep moonquakes, {bt.get('meteoroid', 0):,} meteoroid impacts and "
            f"{bt.get('shallow', 0)} shallow moonquakes. Working stations: {', '.join(stats['stations_working']) or 'none'}."
            + (f" NASA also crashed: {', '.join(stats['artificial_impacts'])}." if stats["artificial_impacts"] else "")
            + "\n\n_Offline template - connect an LLM for an AI-written briefing._")


def evaluate_council(tk: Toolkit, llm: LLM, n: int = 30, seed: int = 0, progress=None) -> pd.DataFrame:
    """Run the full council (incl. LLM Seismologist) on n random held-out test events."""
    test = tk.ev[tk.ev.split == "test"]
    # stratified-ish sample so minority classes appear
    parts = [g.sample(min(len(g), max(2, int(n * len(g) / len(test)))), random_state=seed) for _, g in test.groupby("nasa_label")]
    sample = pd.concat(parts).sample(frac=1, random_state=seed).head(n)
    rows = []
    for k, eid in enumerate(sample.event_id):
        r = run_council(tk, eid, llm, allow_override=True)
        rows.append(dict(event_id=eid, nasa=r["nasa"], classifier=r["ml"], seismologist=r["review"].get("verdict"),
                         council=r["final"], source=r["review"].get("_source")))
        if progress:
            progress((k + 1) / len(sample))
    return pd.DataFrame(rows)
