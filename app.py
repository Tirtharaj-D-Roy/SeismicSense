"""
SeismicSense - Planetary Seismic Event Mapper
CSE473 · LLM & Agentic AI · Group 26
Tirtharaj Dutta Roy (12323508) · Krishna Sharma (12313177)

Run:  streamlit run app.py
"""
import json
import time

import pandas as pd
import streamlit as st

from seismicsense import catalog as C
from seismicsense import viz
from seismicsense.agents import MissionControl, Toolkit, evaluate_council, run_council, year_briefing
from seismicsense.llm import LLM, OFFLINE, PROVIDERS, default_provider

st.set_page_config(page_title="SeismicSense · What shook the Moon?", page_icon="🌕", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1350px;}
.hero h1 {font-size: 2.6rem; margin-bottom: 0; background: linear-gradient(90deg,#C9B8FF,#8B6CFF 45%,#FF8A3D);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.hero p {font-size: 1.12rem; color: #C9C5DA; margin-top: .3rem;}
.card {background: #1A1530; border: 1px solid #2E2650; border-radius: 16px; padding: 18px 20px; height: 100%;}
.card h3 {margin: 0 0 6px 0; font-size: 1.15rem;}
.card p {color: #C9C5DA; font-size: .95rem; margin: 0;}
.kpi {background: linear-gradient(160deg,#211A3D,#151129); border: 1px solid #2E2650; border-radius: 16px;
  padding: 14px 18px; text-align: center;}
.kpi .v {font-size: 2rem; font-weight: 800; color: #fff;} .kpi .l {color: #A9A3C2; font-size: .9rem;}
.verdict {border-radius: 20px; padding: 22px 26px; border: 2px solid; background: #160F2B;}
.verdict .big {font-size: 2.3rem; font-weight: 800;}
.agent {display:inline-block; background:#221B40; border:1px solid #3A2F6B; border-radius: 999px; padding: 6px 14px;
  margin: 4px 2px; font-size: .95rem;}
.arrow {color:#8B6CFF; font-weight:800; margin: 0 2px;}
.tag {display:inline-block; padding:2px 10px; border-radius: 999px; font-size:.8rem; font-weight:700; margin-left:6px;}
.small {color:#A9A3C2; font-size:.85rem;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="🛰️ Loading 13,057 Apollo seismic events and waking up the agents…")
def get_toolkit() -> Toolkit:
    return Toolkit()


tk = get_toolkit()
PCFG = {"scrollZoom": False, "displaylogo": False}
M = tk.metrics


def info(c):
    return C.CLASS_INFO[c]


# ====================================================================== sidebar: LLM brain
with st.sidebar:
    st.markdown("## 🌕 SeismicSense")
    st.caption("Planetary Seismic Event Mapper\n\nCSE473 · LLM & Agentic AI · Group 26")
    st.markdown("**Tirtharaj Dutta Roy** · 12323508  \n**Krishna Sharma** · 12313177")
    st.divider()
    st.markdown("### 🧠 AI brain (LLM)")
    names = list(PROVIDERS)
    prov = st.selectbox("Provider", names, index=names.index(st.session_state.get("prov", default_provider())))
    cfg = PROVIDERS[prov]
    if prov != OFFLINE:
        tmp = LLM(provider=prov)
        key = st.text_input("API key" + (" (not needed)" if prov.startswith("Ollama") else ""), type="password",
                            value=st.session_state.get(f"key_{prov}", tmp.api_key if not prov.startswith("Ollama") else ""),
                            help="Stays in this browser session only. You can also put it in a .env file (see README).")
        st.session_state[f"key_{prov}"] = key
        models = st.session_state.get(f"models_{prov}", [])
        model = st.selectbox("Model", models, index=models.index(tmp.model) if tmp.model in models else 0) if models \
            else st.text_input("Model", value=st.session_state.get(f"model_{prov}", tmp.model))
        st.session_state[f"model_{prov}"] = model
        base = st.text_input("Endpoint", value=cfg["base_url"]) if prov.startswith("Custom") else cfg["base_url"]
        if cfg["key_url"]:
            st.caption(f"Get a key / install: {cfg['key_url']}")
        llm = LLM(provider=prov, model=model, api_key=key, base_url=base)
        c1, c2 = st.columns(2)
        if c1.button("🔌 Test", width="stretch"):
            try:
                t = time.time()
                llm.ping()
                st.success(f"Connected in {time.time() - t:.1f}s")
            except Exception as ex:
                st.error(f"Failed: {str(ex)[:220]}")
        if c2.button("📋 Models", width="stretch", help="Fetch the models this provider offers right now"):
            try:
                st.session_state[f"models_{prov}"] = llm.list_models()
                st.rerun()
            except Exception as ex:
                st.error(f"Failed: {str(ex)[:220]}")
    else:
        llm = LLM()
        st.info("Running without an LLM: agents use templates. Pick **Groq** or **Gemini** (free keys) "
                "or **Ollama** (offline) for the full AI experience.")
    st.session_state["prov"] = prov
    allow_override = st.toggle("Let the LLM overrule an unsure classifier", value=False,
                               help="If the ML model is <60% sure and the LLM Seismologist disagrees, use the LLM's verdict.")
    status = "🟢 " + llm.label if llm.online else "⚪ Offline mode (templates)"
    st.markdown(f"**Status:** {status}")
    if llm.last_error:
        st.warning(llm.last_error)
    st.divider()
    st.caption("Data: NASA PDS Geosciences Node - Apollo Seismic Event Catalog (Nakamura et al.; Weber).")

# ====================================================================== header
st.markdown("""<div class="hero"><h1>What shook the Moon?</h1>
<p>Between 1969 and 1977, seismometers left by Apollo astronauts recorded <b>13,057</b> vibrations on the Moon.
SeismicSense is a team of AI agents that works out <b>what caused each one</b>, <b>where</b> it came from,
and explains it in plain English.</p></div>""", unsafe_allow_html=True)

tabs = st.tabs(["🏠 Start here", "🔍 Investigate an event", "🌕 Moon map", "📅 Timeline & tides",
                "💬 Ask Mission Control", "📊 Report card"])

# ====================================================================== HOME
with tabs[0]:
    k = st.columns(4)
    for col, v, l in zip(k, ["13,057", "8 years", f"{M.get('test_accuracy', 0) * 100:.1f}%", f"{M.get('test_f1_weighted', 0):.2f}"],
                         ["events recorded on the Moon", "of listening (1969-1977)", "accuracy on events the AI never saw",
                          "F1 score (1.00 = perfect)"]):
        col.markdown(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>', unsafe_allow_html=True)
    st.markdown("#### The four things that can shake the Moon")
    cols = st.columns(4)
    counts = tk.ev.nasa_label.value_counts()
    for col, c in zip(cols, ["deep", "meteoroid", "shallow", "artificial"]):
        col.markdown(f"""<div class="card" style="border-top: 4px solid {info(c)['color']}">
        <h3>{info(c)['emoji']} {info(c)['name']}</h3><p><b>{info(c)['short']}.</b><br><br>{info(c)['story']}</p>
        <p class="small" style="margin-top:10px">Recorded: <b>{counts.get(c, 0):,}</b></p></div>""", unsafe_allow_html=True)
    st.markdown("#### How the AI team works")
    st.markdown("""<div>
    <span class="agent">🛰️ Data Scout</span><span class="arrow">→</span>
    <span class="agent">📜 Mission Historian</span><span class="arrow">→</span>
    <span class="agent">📈 Signal Analyst</span><span class="arrow">→</span>
    <span class="agent">🤖 Pattern Classifier <span class="small">(ML)</span></span><span class="arrow">→</span>
    <span class="agent">🧑‍🔬 Seismologist <span class="small">(LLM)</span></span><span class="arrow">→</span>
    <span class="agent">🗺️ Cartographer</span>
    <br><span class="agent" style="margin-top:8px">💬 Mission Control <span class="small">(LLM + tools)</span></span>
    <span class="small"> - the chat agent that plans, calls these tools and answers your questions</span></div>""",
                unsafe_allow_html=True)
    st.markdown("""
| Agent | What it does | Powered by |
|---|---|---|
| 🛰️ Data Scout | Fetches the event's raw record from NASA's catalog | Tool |
| 📜 Mission Historian | Checks Apollo mission logs: did NASA crash a rocket at that moment? | Tool (mission records) |
| 📈 Signal Analyst | Turns numbers into evidence: strength, duration, tides, "has this signal happened before?" | Tool (physics + statistics) |
| 🤖 Pattern Classifier | Machine-learning model trained on 7,336 events labelled by NASA scientists | Gradient-boosted trees |
| 🧑‍🔬 Seismologist | Cross-examines all evidence, gives the verdict, writes the plain-English story | **LLM** |
| 🗺️ Cartographer | Places the event on the Moon globe | Tool |
| 💬 Mission Control | Understands your question, decides which tools to call, combines the results | **LLM with tool-calling** |
""")
    st.info("**Try it in 30 seconds:** open **🔍 Investigate an event** → press **🚀 Run the agent team**. "
            "Then ask **💬 Mission Control** something like *“Why do deep moonquakes follow the tides?”*")

# ====================================================================== INVESTIGATE
with tabs[1]:
    left, right = st.columns([1, 2.1])
    with left:
        mode = st.radio("Which event?", ["🎲 Surprise me - an event the AI never trained on",
                                         "🕵️ Mystery - an event NASA never classified",
                                         "🚀 One of the 9 Apollo crashes", "🔎 Type an event ID"])
        if mode.startswith("🔎"):
            eid = st.text_input("Event ID", value=st.session_state.get("eid", "EV-03342"), help="EV-00001 … EV-13057")
        else:
            pool = "test" if "Surprise" in mode else "mystery" if "Mystery" in mode else "artificial"
            if st.session_state.get("pool") != pool or "eid" not in st.session_state:
                st.session_state["eid"] = tk.ev[tk.ev.split == pool].sample(1).event_id.iloc[0]
                st.session_state["pool"] = pool
            if st.button("🔀 Pick another"):
                st.session_state["eid"] = tk.ev[tk.ev.split == pool].sample(1).event_id.iloc[0]
            eid = st.session_state["eid"]
            st.markdown(f"Selected: **{eid}**")
        go_btn = st.button("🚀 Run the agent team", type="primary", width="stretch")
        st.caption("Each agent's work appears live on the right.")

    with right:
        if go_btn:
            box = st.status(f"Agents investigating {eid}…", expanded=True)

            def show(s):
                tag = {"LLM": "#8B6CFF", "ML": "#FF8A3D", "tool": "#2EC4B6"}[s.kind]
                box.markdown(f"{s.emoji} **{s.agent}** <span class='tag' style='background:{tag}33;color:{tag}'>{s.kind}</span> "
                             f"<span class='small'>{s.seconds:.1f}s</span><br>{s.text}", unsafe_allow_html=True)

            try:
                res = run_council(tk, eid, llm, allow_override, on_step=show)
                box.update(label=f"Done - {len(res['steps'])} agents finished", state="complete", expanded=False)
                st.session_state["res"] = res
            except KeyError as ex:
                box.update(label="Event not found", state="error")
                st.error(str(ex))
        res = st.session_state.get("res")
        if res:
            fi = info(res["final"])
            conf = 1.0 if res["final"] == "artificial" else res["probs"].get(res["final"], 0)
            st.markdown(f"""<div class="verdict" style="border-color:{fi['color']}">
            <div class="small">{res['event_id']} · {res['time']:%d %B %Y, %H:%M} UTC</div>
            <div class="big">{fi['emoji']} {fi['name']}</div>
            <div style="font-size:1.1rem;color:#D8D4EA">{fi['short']} · <b>{conf * 100:.0f}% confidence</b></div>
            <div class="small" style="margin-top:6px">Decision: {res['decision']}</div></div>""", unsafe_allow_html=True)
            if res["nasa"] != "unknown":
                ok = res["nasa"] == res["final"]
                (st.success if ok else st.warning)(
                    f"{'✅ Matches' if ok else '❌ Differs from'} NASA scientists' label: **{info(res['nasa'])['name']}**"
                    + (" · this event was held out from training, so this is a fair test." if res["split"] == "test" else ""))
            else:
                st.info("🕵️ NASA never classified this event - SeismicSense's answer is a new prediction.")
            rv = res["review"]
            a, b = st.columns([1.15, 1])
            with a:
                st.markdown(f"#### 🧑‍🔬 The Seismologist's explanation {'' if rv.get('_source') == 'llm' else '(offline template)'}")
                st.markdown(f"**{rv.get('headline', '')}**")
                st.write(rv.get("story", ""))
                for r in rv.get("key_reasons", [])[:4]:
                    st.markdown(f"- {r}")
                if rv.get("caveat"):
                    st.caption("⚠️ " + rv["caveat"])
            with b:
                st.markdown("#### 🤖 Classifier's probabilities")
                st.plotly_chart(viz.probability_bars(res["probs"]), width="stretch", config={"displayModeBar": False})
                st.caption("The ML model only learned natural events; rocket crashes come from the Mission Historian.")
            st.markdown("#### 🗺️ Where on the Moon")
            st.plotly_chart(res["map"], width="stretch", config=PCFG)
            st.caption("Drag to spin the Moon. Yellow diamonds = Apollo landing sites; orange glow = how strongly each seismometer felt this event (bigger = stronger).")
            with st.expander("🔬 All evidence the agents shared (JSON)"):
                st.json(res["evidence"])

# ====================================================================== MAP
with tabs[2]:
    c1, c2 = st.columns([3, 1])
    with c2:
        st.markdown("#### Layers")
        layers = [c for c in ["deep", "meteoroid", "shallow", "artificial"]
                  if st.checkbox(f"{info(c)['emoji']} {info(c)['name']}", value=True, key=f"lay_{c}")]
        rot = st.slider("Spin the Moon (longitude facing you)", -180, 180, 0, step=15,
                        help="0 = the side that always faces Earth; ±180 = the far side")
        xray = st.toggle("🩻 X-ray view: see deep quakes at their real depth", value=False)
        st.markdown(f"""<div class="card"><p>
        • <b>{len(tk.locs['deep_nests'])}</b> deep-moonquake “nests”, sized by how many quakes each produced.<br>
        • <b>{len(tk.locs['shallow'])}</b> located shallow moonquakes<br>
        • <b>{len(tk.locs['meteoroid'])}</b> located meteoroid impacts<br>
        • <b>{len(tk.locs['artificial'])}</b> Apollo crash sites<br><br>
        Most events can't be pinpointed - with only 4 seismometers close together, scientists could locate only the
        clearest ones. That's why most dots sit on the near side.</p></div>""", unsafe_allow_html=True)
    with c1:
        st.plotly_chart(viz.moon_map(tk.locs, tk.nest_counts(), layers=layers, height=680, rotation=(rot, 0), xray=xray),
                        width="stretch", config=PCFG)
        st.caption("🖱️ Drag to spin the Moon, hover any dot for details (zoom with the toolbar buttons).")
        st.caption("Source locations: Nakamura (2005) deep nests, Nakamura (1979) shallow quakes, Nakamura (1983) & "
                   "Gagnepain-Beyneix (2006) impacts - all from NASA PDS. Dark patches are approximate outlines of the lunar 'seas' (maria).")

# ====================================================================== TIMELINE
with tabs[3]:
    f1, f2, f3 = st.columns([2, 2, 1.3])
    yr = f1.slider("Years", 1969, 1977, (1969, 1977))
    types = f2.multiselect("Event types", ["deep", "meteoroid", "shallow", "artificial", "unknown"],
                           default=["deep", "meteoroid", "shallow", "artificial"], format_func=lambda c: f"{info(c)['emoji']} {info(c)['name']}")
    src = f3.radio("Labels from", ["NASA scientists", "SeismicSense AI"])
    col = "nasa_label" if src.startswith("NASA") else "label"
    ev = tk.ev[(tk.ev.year >= yr[0]) & (tk.ev.year <= yr[1]) & tk.ev[col].isin(types)]
    st.plotly_chart(viz.timeline(ev, label_col=col), width="stretch", config=PCFG)
    st.plotly_chart(viz.monthly_counts(ev, label_col=col), width="stretch", config=PCFG)
    st.caption("When SeismicSense AI labels are selected, the 3,878 events NASA never classified get a label too.")
    st.markdown("### 💓 The Moon's tidal heartbeat")
    a, b = st.columns([2, 1])
    a.plotly_chart(viz.tidal_heartbeat(tk.ev, label_col="nasa_label"), width="stretch", config=PCFG)
    b.markdown("""<div class="card"><h3>Why do deep moonquakes have a rhythm?</h3><p>
    The Moon's orbit is an oval. Every <b>27.55 days</b> it swings close to Earth and then far away.
    Earth's gravity squeezes and relaxes the Moon like a stress ball, and deep moonquakes happen
    <b>about twice as often</b> at some points of that cycle as at others.<br><br>
    Meteoroid impacts stay flat: space rocks don't care where the Moon is in its orbit.
    The AI uses this rhythm (and the fact that deep "nests" repeat) to tell the two apart.</p></div>""",
               unsafe_allow_html=True)
    st.markdown("### 📰 AI Moon-news briefing")
    y1, y2 = st.columns([1, 4])
    by = y1.selectbox("Year", list(range(1969, 1978)), index=3)
    if y1.button("🧠 Write briefing"):
        with st.spinner("The Storyteller agent is writing…"):
            st.session_state["brief"] = (by, year_briefing(tk, llm, by))
    if st.session_state.get("brief"):
        y2.markdown(f"**{st.session_state['brief'][0]}**  \n" + st.session_state["brief"][1])

# ====================================================================== CHAT
with tabs[4]:
    st.markdown("Ask anything about the Moon's seismic record. **Mission Control** (an LLM agent) decides which tools "
                "to use - searching 13,057 events, analysing one, drawing maps and charts - then answers in plain English.")
    if not llm.online:
        st.warning("Offline mode: answers come from a simple keyword router. Connect an LLM in the sidebar for real conversation.")
    hist = st.session_state.setdefault("chat", [])
    sugg = ["Why do deep moonquakes follow the tides?", "What were the 5 strongest meteoroid impacts?",
            "Tell me about 1972 on the Moon", "Explain event EV-03342", "Show me where Apollo rockets crashed",
            "How accurate is SeismicSense?"]
    sc = st.columns(3)
    clicked = None
    for i, s in enumerate(sugg):
        if sc[i % 3].button(s, key=f"sg{i}", width="stretch"):
            clicked = s
    for mi, m in enumerate(hist):
        with st.chat_message(m["role"], avatar="🧑‍🚀" if m["role"] == "user" else "🌕"):
            st.markdown(m["content"])
            for j, (t, f) in enumerate(m.get("figs", [])):
                st.plotly_chart(f, width="stretch", key=f"hist_{mi}_{j}", config=PCFG)
            if m.get("trace"):
                with st.expander(f"🔧 Tools the agent used ({len(m['trace'])})"):
                    for name, args, res in m["trace"]:
                        st.markdown(f"**{name}**(`{json.dumps(args)}`)")
                        st.code(json.dumps(res, default=str)[:700], language="json")
    q = st.chat_input("e.g. How many shallow moonquakes were there?") or clicked
    if q:
        with st.chat_message("user", avatar="🧑‍🚀"):
            st.markdown(q)
        with st.chat_message("assistant", avatar="🌕"):
            with st.spinner("Mission Control is planning and calling tools…"):
                out = MissionControl(tk, llm).ask(q, [{"role": m["role"], "content": m["content"]} for m in hist])
            st.markdown(out["answer"])
            for j, (t, f) in enumerate(out["figs"]):
                st.plotly_chart(f, width="stretch", key=f"hist_{len(hist) + 1}_{j}", config=PCFG)
            if out["trace"]:
                with st.expander(f"🔧 Tools the agent used ({len(out['trace'])})"):
                    for name, args, res in out["trace"]:
                        st.markdown(f"**{name}**(`{json.dumps(args)}`)")
                        st.code(json.dumps(res, default=str)[:700], language="json")
        hist += [{"role": "user", "content": q},
                 {"role": "assistant", "content": out["answer"], "figs": out["figs"], "trace": out["trace"]}]
    if hist and st.button("🧹 Clear chat"):
        st.session_state["chat"] = []
        st.rerun()

# ====================================================================== REPORT CARD
with tabs[5]:
    st.markdown("Tested on **{:,} events the AI never saw during training** (a random 20% held out from NASA's "
                "labelled events), plus 5-fold cross-validation on the training set.".format(M.get("n_test", 0)))
    k = st.columns(5)
    vals = [(f"{M['test_accuracy'] * 100:.1f}%", "Accuracy"), (f"{M['test_f1_weighted']:.3f}", "F1 (weighted)"),
            (f"{M['test_precision_weighted']:.3f}", "Precision (weighted)"), (f"{M['test_recall_weighted']:.3f}", "Recall (weighted)"),
            (f"{M['cv5_f1_weighted_mean']:.3f} ± {M['cv5_f1_weighted_std']:.3f}", "5-fold CV F1")]
    for col, (v, l) in zip(k, vals):
        col.markdown(f'<div class="kpi"><div class="v" style="font-size:1.6rem">{v}</div><div class="l">{l}</div></div>',
                     unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.markdown("#### Where the AI is right and wrong")
        st.plotly_chart(viz.confusion(M["confusion_matrix"], M["classes"]), width="stretch", config=PCFG)
        rows = [{"Type": f"{info(c)['emoji']} {info(c)['name']}", "Test events": int(v["support"]),
                 "Precision": round(v["precision"], 3), "Recall": round(v["recall"], 3), "F1": round(v["f1-score"], 3)}
                for c, v in M["per_class"].items()]
        p4 = M["pipeline_4class"]
        rows.append({"Type": "🚀 Artificial Impact (Mission Historian)", "Test events": p4["artificial_total"],
                     "Precision": 1.0, "Recall": p4["artificial_found"] / p4["artificial_total"], "F1": 1.0})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with b:
        st.markdown("#### What the AI pays attention to")
        st.plotly_chart(viz.importance_bars(M["feature_group_importance"]), width="stretch", config=PCFG)
        st.caption("Permutation importance on the test set. Tidal timing is mostly captured inside the 'repeating fingerprint' "
                   "features, which compare events at the same point of the 27.55-day tidal cycle.")
    st.markdown("""<div class="card"><h3>📌 Notes</h3><p>
    • <b>Deep moonquakes</b> (81% of labelled events) are recognised very reliably.<br>
    • <b>Meteoroid impacts</b> are harder: with only the catalog numbers (not the full waveforms) some look like deep quakes.<br>
    • <b>Shallow moonquakes</b>: only 28 exist in 8 years (6 in the test set) and the catalog alone barely separates them
      from impacts - NASA scientists used high-frequency waveform details to spot them. The macro-F1 (unweighted average over
      the 3 types) is therefore lower ({:.2f}); weighted F1 reflects performance on the events as they actually occur.<br>
    • <b>Artificial impacts</b> are identified by the Mission Historian from Apollo records: 9 of 9 found, no false alarms.</p></div>"""
                .format(M["test_f1_macro"]), unsafe_allow_html=True)

    st.markdown("### 🧑‍🔬 Live test: does the LLM Seismologist agree?")
    st.caption(f"Runs the full agent council (with {llm.label}) on held-out test events and compares classifier, LLM and NASA.")
    n = st.slider("Events to test", 5, 60, 15, step=5)
    if st.button("▶️ Run live evaluation"):
        bar = st.progress(0.0)
        df = evaluate_council(tk, llm, n=n, progress=bar.progress)
        bar.empty()
        st.session_state["eval"] = df
    if "eval" in st.session_state:
        df = st.session_state["eval"]
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Classifier accuracy", f"{(df.classifier == df.nasa).mean() * 100:.0f}%")
        e2.metric("LLM Seismologist accuracy", f"{(df.seismologist == df.nasa).mean() * 100:.0f}%")
        e3.metric("Council (final) accuracy", f"{(df.council == df.nasa).mean() * 100:.0f}%")
        e4.metric("LLM agrees with classifier", f"{(df.seismologist == df.classifier).mean() * 100:.0f}%")
        st.dataframe(df, hide_index=True, width="stretch")
