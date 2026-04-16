import streamlit as st
import pandas as pd
from pipeline import extract_entities, get_triples, ask_llm
from drug_interaction import get_drug_profile, find_shared_risks, assess_safety
from patient_safety import analyze_patient
from threehop import get_3hop_triples, ask_llm_3hop
from mlp_retriever import retrieve_with_mlp_dde
from groq import Groq
from neo4j import GraphDatabase
import networkx as nx
from collections import Counter

GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)
driver = GraphDatabase.driver("neo4j://127.0.0.1:7687", auth=("neo4j", "password123"))

st.set_page_config(page_title="Medical SubgraphRAG", page_icon="🏥", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f4f6f9; }
    section[data-testid="stSidebar"] {
        background: #1c2a3a;
        border-right: 1px solid #2e3f52;
    }
    .hero {
        background: linear-gradient(135deg, #1c2a3a, #2e3f52);
        border-radius: 14px;
        padding: 36px;
        text-align: center;
        margin-bottom: 24px;
    }
    .hero h1 { color: #ffffff; font-size: 2.2em; margin: 0; }
    .hero p { color: #a0b0c0; margin: 8px 0 0 0; }
    .stat-card {
        background: #ffffff;
        border: 1px solid #dde3ea;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .stat-number { color: #2c5f8a; font-size: 1.8em; font-weight: bold; }
    .stat-label { color: #6b7c8d; font-size: 0.85em; margin-top: 4px; }
    .step-label {
        color: #2c5f8a;
        font-size: 0.8em;
        font-weight: bold;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 16px 0 6px 0;
    }
    .entity-tag {
        background: #e8f0fe;
        border: 1px solid #b0c4de;
        border-radius: 20px;
        padding: 3px 12px;
        color: #2c5f8a;
        font-size: 0.85em;
        margin-right: 6px;
    }
    .answer-box {
        background: #ffffff;
        border: 1px solid #dde3ea;
        border-left: 4px solid #2c5f8a;
        border-radius: 10px;
        padding: 20px;
        color: #2d3748;
        line-height: 1.7;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .triple-item {
        font-family: monospace;
        font-size: 0.82em;
        color: #2c5f8a;
        background: #f0f4f8;
        border: 1px solid #dde3ea;
        border-radius: 6px;
        padding: 5px 10px;
        margin: 3px 0;
    }
    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        background: #4caf84;
        border-radius: 50%;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<h3 style="color:#ffffff">🏥 Medical SubgraphRAG</h3>', unsafe_allow_html=True)
    st.markdown('<span class="status-dot"></span><span style="color:#4caf84;font-size:0.85em">Connected to PrimeKG</span>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<p style="color:#a0b0c0;font-size:0.8em;font-weight:bold">DATASET INFO</p>', unsafe_allow_html=True)
    for k, v in {"Dataset": "PrimeKG", "Diseases": "17,029", "Drugs": "2,232", "Relationships": "284,089", "LLM": "Llama 3.3 70B"}.items():
        st.markdown(f'<div style="display:flex;justify-content:space-between;color:#8892b0;font-size:0.85em;padding:3px 0"><span>{k}</span><span style="color:#ffffff">{v}</span></div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<p style="color:#a0b0c0;font-size:0.8em;font-weight:bold">SAMPLE QUESTIONS</p>', unsafe_allow_html=True)
    for s in ["What drugs treat Alzheimer disease?", "What are the side effects of Galantamine?",
              "What drugs treat Parkinson disease?", "Contraindications of Metformin?", "What drugs treat epilepsy?"]:
        if st.button(s, use_container_width=True, key=s):
            st.session_state.query = s
    st.divider()
    st.markdown('<div style="color:#6b7c8d;font-size:0.75em;text-align:center">Powered by Llama 3.3 70B via Groq</div>', unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🏥 Medical SubgraphRAG Assistant</h1>
    <p>Graph-based retrieval for medical question answering using PrimeKG</p>
</div>
""", unsafe_allow_html=True)

# ── STATS ─────────────────────────────────────────────────────────────────────
for col, num, label in zip(st.columns(4), ["17,029","2,232","284K","10/10"], ["Diseases","Drugs","Relationships","Eval Score"]):
    with col:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{num}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔍 Ask a Question",
    "📊 Evaluation Results",
    "💊 Drug Interaction Checker",
    "🧑‍⚕️ Patient Safety Checker",
    "🔬 Disease Explorer",
    "📈 KG Statistics"
])

# ── TAB 1: QA ─────────────────────────────────────────────────────────────────
with tab1:
    query = st.text_input("Your Question",
        value=st.session_state.get("query", ""),
        placeholder="e.g. What drugs treat Alzheimer disease?",
        label_visibility="collapsed")

    method = st.radio("🧠 Retrieval Method",
        ["Cosine Similarity (Baseline)", "DDE + MLP (SubgraphRAG)", "2-hop", "3-hop"],
        horizontal=True)

    search = st.button("🔍 Search Knowledge Graph", type="primary", use_container_width=True)

    if search and query:
        left, right = st.columns([1, 1])
        with left:
            st.markdown('<div class="step-label">⚡ Step 1 — Entity Extraction</div>', unsafe_allow_html=True)
            with st.spinner("Extracting entities..."):
                entities = extract_entities(query, client)
            entity_html = " ".join([f'<span class="entity-tag">{e}</span>' for e in entities])
            st.markdown(f"Extracted: {entity_html}", unsafe_allow_html=True)

            st.markdown('<div class="step-label">🔗 Step 2 — Subgraph Retrieval</div>', unsafe_allow_html=True)
            all_triples = []

            if "DDE + MLP" in method:
                st.info("🧮 Running DDE encoding + MLP training... (~30 seconds)")
                with st.spinner("Computing DDE + training MLP..."):
                    for entity in entities:
                        result_mlp = retrieve_with_mlp_dde(query, entity, top_k=50)
                        all_triples.extend(result_mlp["mlp_dde"])
                mlp_ind = result_mlp.get("mlp_indication_count", 0)
                cos_ind = result_mlp.get("cosine_indication_count", 0)
                st.success(f"✅ MLP+DDE: **{mlp_ind}** indication triples in top-10 vs Cosine: **{cos_ind}**")
                st.markdown(f"Retrieved **{len(all_triples)}** triples via **DDE+MLP** from Neo4j")

            elif "3-hop" in method:
                with st.spinner("Querying Neo4j (3-hop)..."):
                    for entity in entities:
                        all_triples.extend(get_3hop_triples(entity))
                st.markdown(f"Retrieved **{len(all_triples)}** triples via **3-hop** from Neo4j")

            elif "2-hop" in method:
                with st.spinner("Querying Neo4j (2-hop)..."):
                    from multihop import get_2hop_triples
                    for entity in entities:
                        all_triples.extend(get_2hop_triples(entity))
                st.markdown(f"Retrieved **{len(all_triples)}** triples via **2-hop** from Neo4j")

            else:
                with st.spinner("Querying Neo4j..."):
                    for entity in entities:
                        all_triples.extend(get_triples(entity))
                st.markdown(f"Retrieved **{len(all_triples)}** triples via **cosine similarity** from Neo4j")

            with st.expander("View Retrieved Triples"):
                for t in all_triples[:15]:
                    st.markdown(f'<div class="triple-item">{t["text"]}</div>', unsafe_allow_html=True)
                if len(all_triples) > 15:
                    st.caption(f"+ {len(all_triples)-15} more triples")

            st.markdown('<div class="step-label">🤖 Step 3 — LLM Reasoning</div>', unsafe_allow_html=True)
            with st.spinner("Reasoning over subgraph..."):
                if "3-hop" in method:
                    answer = ask_llm_3hop(query, all_triples)
                else:
                    answer = ask_llm(query, all_triples, client)
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="step-label">🕸️ Knowledge Subgraph (from Neo4j)</div>', unsafe_allow_html=True)
            priority = ["indication","side_effect","mechanism","contraindication"]
            sorted_triples = sorted(all_triples[:50],
                key=lambda t: next((i for i,p in enumerate(priority) if p in t["text"]), 99))[:12]
            G = nx.DiGraph()
            for t in sorted_triples:
                text = t["text"]
                if " AND " in text:
                    for part in text.split(" AND "):
                        p = part.strip().strip("()").split(", ")
                        if len(p) == 3:
                            h    = p[0].strip()[:22] + "..." if len(p[0].strip()) > 22 else p[0].strip()
                            tail = p[2].strip()[:22] + "..." if len(p[2].strip()) > 22 else p[2].strip()
                            G.add_edge(h, tail, label=p[1].strip())
                else:
                    parts = text.strip("()").split(", ")
                    if len(parts) == 3:
                        h    = parts[0].strip()[:22] + "..." if len(parts[0].strip()) > 22 else parts[0].strip()
                        tail = parts[2].strip()[:22] + "..." if len(parts[2].strip()) > 22 else parts[2].strip()
                        G.add_edge(h, tail, label=parts[1].strip())

            if G.number_of_nodes() > 0:
                from pyvis.network import Network
                net = Network(height="500px", width="100%", bgcolor="#f8fafc", font_color="#2d3748")
                topic = entities[0] if entities else ""
                for node in G.nodes():
                    if topic.lower()[:8] in node.lower():
                        net.add_node(node, color="#2c5f8a", size=35, font={"color":"#ffffff","size":13,"bold":True}, shape="ellipse")
                    else:
                        net.add_node(node, color="#7fa8c9", size=20, font={"color":"#1a202c","size":10}, shape="ellipse")
                for u, v, data in G.edges(data=True):
                    net.add_edge(u, v, label=data.get("label",""), color="#5a8ab0", width=2,
                                 font={"color":"#4a6080","size":9}, arrows="to")
                net.set_options('{"physics":{"stabilization":{"iterations":300},"repulsion":{"nodeDistance":250,"springLength":250,"springConstant":0.01}},"layout":{"improvedLayout":true}}')
                st.components.v1.html(net.generate_html(), height=520)
            else:
                st.info("No graph data available for this query.")

# ── TAB 2: EVALUATION ─────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📊 SubgraphRAG vs Baseline RAG — Full Evaluation")
    st.markdown("Evaluated on 10 medical questions using PrimeKG as the knowledge source.")
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### 🎯 Key Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown('<div class="stat-card"><div class="stat-number" style="color:#27ae60">10/10</div><div class="stat-label">SubgraphRAG Hit@1</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="stat-card"><div class="stat-number" style="color:#c0392b">2/10</div><div class="stat-label">Baseline Hit@1</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="stat-card"><div class="stat-number" style="color:#27ae60">100</div><div class="stat-label">SubgraphRAG Hallucination Score</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="stat-card"><div class="stat-number" style="color:#c0392b">20</div><div class="stat-label">Baseline Hallucination Score</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="stat-card"><div class="stat-number" style="color:#2c5f8a">5×</div><div class="stat-label">Overall Improvement</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # DDE+MLP vs Cosine comparison
    st.markdown("#### 🧠 DDE+MLP vs Cosine Similarity — Retrieval Quality")
    st.dataframe(pd.DataFrame({
        "Query": [
            "What drugs treat Alzheimer disease?",
            "What drugs treat Parkinson disease?",
            "What drugs treat epilepsy?",
        ],
        "Cosine — Indication in Top-10": ["0/10 ❌", "0/10 ❌", "0/10 ❌"],
        "DDE+MLP — Indication in Top-10": ["10/10 ✅", "10/10 ✅", "10/10 ✅"],
    }), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Single-hop Evaluation (Hit@1)")
    st.dataframe(pd.DataFrame({
        "Question": [
            "What drugs treat Alzheimer disease?",
            "What are the contraindications of Codeine?",
            "What drugs treat Parkinson disease?",
            "What are the side effects of Galantamine?",
            "What drugs treat hypertension?",
            "What are the contraindications of Metformin?",
            "What drugs treat epilepsy?",
            "What are the side effects of Donepezil?",
            "What drugs treat asthma?",
            "What are the contraindications of Aspirin?",
        ],
        "SubgraphRAG Hit@1": ["✅ Hit"] * 10,
        "Baseline Hit@1": ["❌ Miss","❌ Miss","❌ Miss","❌ Miss","✅ Hit",
                           "❌ Miss","❌ Miss","❌ Miss","✅ Hit","❌ Miss"],
        "SubgraphRAG Grounded": ["✅ Grounded"] * 10,
        "Baseline Grounded": ["❌ No","❌ No","❌ No","❌ No","✅ Yes",
                              "❌ No","❌ No","❌ No","✅ Yes","❌ No"],
    }), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Multi-hop Evaluation (2-hop & 3-hop)")
    st.dataframe(pd.DataFrame({
        "Question": [
            "Side effects of drugs treating Alzheimer disease?",
            "Side effects of drugs treating Parkinson disease?",
            "Contraindications of drugs treating Epilepsy?",
            "Diseases related to side effects of Alzheimer drugs?",
            "Diseases sharing contraindications with Parkinson drugs?",
        ],
        "Hops": [2, 2, 2, 3, 3],
        "SubgraphRAG": ["⚠️ Partial", "✅ Rich results", "✅ Extensive", "✅ Found", "⚠️ Partial"],
        "Baseline RAG": ["❌ Not available"] * 5,
    }), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 SubgraphRAG with DDE+MLP achieves 10/10 indication triples in top results vs 0/10 for cosine similarity — proving structural encoding dramatically improves retrieval quality.")

# ── TAB 3: DRUG INTERACTION ───────────────────────────────────────────────────
with tab3:
    st.markdown("### 💊 Drug-Drug Interaction Checker")
    st.markdown('<p style="color:#2d3748">Enter two drugs to check if they are safe to take together, based on your medical KG.</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        drug1 = st.text_input("💊 Drug 1", placeholder="e.g. Metformin", key="drug1")
    with c2:
        drug2 = st.text_input("💊 Drug 2", placeholder="e.g. Codeine", key="drug2")

    check = st.button("🔍 Check Interaction", type="primary", use_container_width=True)

    if check and drug1 and drug2:
        with st.spinner(f"Fetching profiles for {drug1} and {drug2} from Neo4j..."):
            profile1 = get_drug_profile(drug1)
            profile2 = get_drug_profile(drug2)
            risks    = find_shared_risks(profile1, profile2)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📋 Drug Profiles")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div style="background:#ffffff;border:1px solid #dde3ea;border-top:4px solid #2c5f8a;
            border-radius:10px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,0.05)">
                <h4 style="color:#2c5f8a;margin:0 0 12px 0">💊 {drug1}</h4>
                <p style="margin:6px 0;color:#2d3748">🎯 Treats <b>{len(profile1['treats'])}</b> conditions</p>
                <p style="margin:6px 0;color:#2d3748">⚠️ <b>{len(profile1['contraindications'])}</b> contraindications</p>
                <p style="margin:6px 0;color:#2d3748">🔬 <b>{len(profile1['side_effects'])}</b> side effects</p>
                <hr style="border-color:#eee;margin:12px 0">
                <p style="color:#6b7c8d;font-size:0.85em">Treats: {', '.join(profile1['treats'][:3]) or 'N/A'}</p>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div style="background:#ffffff;border:1px solid #dde3ea;border-top:4px solid #7fa8c9;
            border-radius:10px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,0.05)">
                <h4 style="color:#2c5f8a;margin:0 0 12px 0">💊 {drug2}</h4>
                <p style="margin:6px 0;color:#2d3748">🎯 Treats <b>{len(profile2['treats'])}</b> conditions</p>
                <p style="margin:6px 0;color:#2d3748">⚠️ <b>{len(profile2['contraindications'])}</b> contraindications</p>
                <p style="margin:6px 0;color:#2d3748">🔬 <b>{len(profile2['side_effects'])}</b> side effects</p>
                <hr style="border-color:#eee;margin:12px 0">
                <p style="color:#6b7c8d;font-size:0.85em">Treats: {', '.join(profile2['treats'][:3]) or 'N/A'}</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### ⚠️ Shared Risks Found in KG")
        r1, r2, r3 = st.columns(3)
        with r1:
            color = "#c0392b" if len(risks["shared_contraindications"]) > 5 else "#e67e22" if len(risks["shared_contraindications"]) > 0 else "#27ae60"
            st.markdown(f'<div class="stat-card"><div class="stat-number" style="color:{color}">{len(risks["shared_contraindications"])}</div><div class="stat-label">Shared Contraindications</div></div>', unsafe_allow_html=True)
        with r2:
            color = "#c0392b" if len(risks["shared_side_effects"]) > 15 else "#e67e22" if len(risks["shared_side_effects"]) > 0 else "#27ae60"
            st.markdown(f'<div class="stat-card"><div class="stat-number" style="color:{color}">{len(risks["shared_side_effects"])}</div><div class="stat-label">Shared Side Effects</div></div>', unsafe_allow_html=True)
        with r3:
            color = "#c0392b" if len(risks["therapeutic_conflicts"]) > 0 else "#27ae60"
            st.markdown(f'<div class="stat-card"><div class="stat-number" style="color:{color}">{len(risks["therapeutic_conflicts"])}</div><div class="stat-label">Therapeutic Conflicts</div></div>', unsafe_allow_html=True)

        if risks["shared_contraindications"]:
            with st.expander(f"View {len(risks['shared_contraindications'])} Shared Contraindications"):
                cols = st.columns(3)
                for i, c in enumerate(risks["shared_contraindications"][:15]):
                    with cols[i % 3]:
                        st.markdown(f"🔴 {c}")

        if risks["therapeutic_conflicts"]:
            with st.expander(f"View {len(risks['therapeutic_conflicts'])} Therapeutic Conflicts"):
                for c in risks["therapeutic_conflicts"]:
                    st.markdown(f"⚡ {c}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🕸️ Interaction Subgraph (from Neo4j)")
        G2 = nx.DiGraph()
        for c in risks["shared_contraindications"][:6]:
            G2.add_edge(drug1, c, label="contraindication")
            G2.add_edge(drug2, c, label="contraindication")
        for c in risks["therapeutic_conflicts"][:3]:
            G2.add_edge(drug1, c, label="treats")
            G2.add_edge(drug2, c, label="conflict")

        if G2.number_of_nodes() > 0:
            from pyvis.network import Network
            net2 = Network(height="420px", width="100%", bgcolor="#f8fafc", font_color="#2d3748")
            for node in G2.nodes():
                if node == drug1:
                    net2.add_node(node, color="#2c5f8a", size=40, font={"color":"#ffffff","size":16,"bold":True}, shape="ellipse")
                elif node == drug2:
                    net2.add_node(node, color="#5a8ab0", size=40, font={"color":"#ffffff","size":16,"bold":True}, shape="ellipse")
                else:
                    net2.add_node(node, color="#f0f4f8", size=22, font={"color":"#2d3748","size":11}, shape="ellipse")
            for u, v, data in G2.edges(data=True):
                lbl   = data.get("label","")
                color = "#c0392b" if lbl == "contraindication" else "#e67e22"
                net2.add_edge(u, v, label=lbl, color=color, width=2,
                              font={"color":color,"size":10}, arrows="to")
            net2.set_options('{"physics":{"stabilization":{"iterations":300},"repulsion":{"nodeDistance":220,"springLength":220,"springConstant":0.02}}}')
            st.components.v1.html(net2.generate_html(), height=440)
            st.markdown(f'<div style="display:flex;gap:20px;padding:8px 0;font-size:0.85em;color:#6b7c8d"><span>🔵 <b>{drug1}</b></span><span>🔹 <b>{drug2}</b></span><span>⚪ Shared condition</span><span style="color:#c0392b">─── Contraindication</span></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🤖 AI Safety Assessment")
        with st.spinner("Running LLM safety assessment..."):
            assessment = assess_safety(drug1, drug2, profile1, profile2, risks, client)

        is_avoid   = "AVOID" in assessment
        is_caution = "CAUTION" in assessment
        verdict_color = "#27ae60" if not is_avoid and not is_caution else "#c0392b" if is_avoid else "#e67e22"
        verdict_text  = "✅ SAFE" if not is_avoid and not is_caution else "🚫 AVOID" if is_avoid else "⚠️ CAUTION"
        st.markdown(f'<div style="background:#ffffff;border:1px solid {verdict_color}33;border-left:5px solid {verdict_color};border-radius:10px;padding:24px"><h3 style="color:{verdict_color};margin:0 0 12px 0">{verdict_text}: {drug1} + {drug2}</h3><p style="color:#2d3748;line-height:1.8;margin:0">{assessment}</p></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("⚠️ For educational purposes only. Always consult a licensed healthcare professional.")

# ── TAB 4: PATIENT SAFETY ─────────────────────────────────────────────────────
with tab4:
    st.markdown("### 🧑‍⚕️ Patient Safety Checker")
    st.markdown('<p style="color:#2d3748">Enter your medical conditions and drugs to check — we\'ll tell you which are safe based on the knowledge graph.</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    conditions_input = st.text_input("🏥 Patient Conditions (comma separated)",
        placeholder="e.g. diabetes, kidney disease, hypertension")
    drugs_input = st.text_input("💊 Drugs to Check (comma separated)",
        placeholder="e.g. Metformin, Codeine, Lisinopril, Aspirin")

    check_safety = st.button("🔍 Check Patient Safety", type="primary", use_container_width=True)

    if check_safety and conditions_input and drugs_input:
        conditions = [c.strip() for c in conditions_input.split(",") if c.strip()]
        drugs      = [d.strip() for d in drugs_input.split(",") if d.strip()]

        with st.spinner("Analyzing patient safety from Neo4j..."):
            result = analyze_patient(conditions, drugs, client)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"#### Results for patient with: `{', '.join(conditions)}`")

        cols = st.columns(min(len(drugs), 4))
        for i, r in enumerate(result["drug_results"]):
            with cols[i % 4]:
                color  = "#27ae60" if r["safe"] else "#c0392b"
                bg     = "#f0fff4" if r["safe"] else "#fff5f5"
                icon   = "✅" if r["safe"] else "❌"
                status = "SAFE" if r["safe"] else "UNSAFE"
                reason = "" if r["safe"] else f"<p style='color:#c0392b;font-size:0.8em;margin:4px 0'>Unsafe for: {', '.join(r['unsafe_for'])}</p>"
                st.markdown(f"""
                <div style="background:{bg};border:1px solid {color}33;border-top:4px solid {color};
                border-radius:10px;padding:16px;text-align:center;margin-bottom:10px">
                    <div style="font-size:2em">{icon}</div>
                    <h4 style="color:#2d3748;margin:8px 0">{r['drug']}</h4>
                    <span style="color:{color};font-weight:bold">{status}</span>
                    {reason}
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🤖 Clinical Summary")
        border_color = "#27ae60" if all(r["safe"] for r in result["drug_results"]) else "#c0392b"
        st.markdown(f'<div style="background:#ffffff;border:1px solid {border_color}33;border-left:5px solid {border_color};border-radius:10px;padding:20px"><p style="color:#2d3748;line-height:1.8;margin:0">{result["summary"]}</p></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("⚠️ For educational purposes only. Always consult a licensed healthcare professional.")

# ── TAB 5: DISEASE EXPLORER ───────────────────────────────────────────────────
with tab5:
    st.markdown("### 🔬 Disease Explorer")
    st.markdown('<p style="color:#2d3748">Search any disease to explore all connected drugs, side effects and relationships in the knowledge graph.</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    disease_input = st.text_input("🦠 Enter a Disease",
        placeholder="e.g. Alzheimer disease, Parkinson disease, epilepsy")
    explore = st.button("🔍 Explore Disease", type="primary", use_container_width=True)

    if explore and disease_input:
        with st.spinner(f"Fetching all connections for {disease_input}..."):
            with driver.session(database="neo4j") as session:
                result = session.run("""
                    MATCH (a)-[r]->(b)
                    WHERE toLower(a.name) CONTAINS toLower($disease)
                       OR toLower(b.name) CONTAINS toLower($disease)
                    RETURN a.name AS head, r.type AS relation, b.name AS tail,
                           a.type AS head_type, b.type AS tail_type
                    LIMIT 200
                """, disease=disease_input)
                rows = [dict(r) for r in result]

        if not rows:
            st.warning(f"No data found for '{disease_input}'. Try a different name.")
        else:
            drugs   = list(set(r["head"] for r in rows if r["relation"] == "indication"))
            contra  = list(set(r["head"] for r in rows if r["relation"] == "contraindication"))
            effects = list(set(r["tail"] for r in rows if r["relation"] == "drug_effect"))

            st.markdown(f"#### Results for: `{disease_input}`")
            st.markdown("<br>", unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="stat-card"><div class="stat-number" style="color:#27ae60">{len(drugs)}</div><div class="stat-label">Drugs that Treat</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="stat-card"><div class="stat-number" style="color:#c0392b">{len(contra)}</div><div class="stat-label">Contraindicated Drugs</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="stat-card"><div class="stat-number" style="color:#e67e22">{len(effects)}</div><div class="stat-label">Related Effects</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="stat-card"><div class="stat-number" style="color:#2c5f8a">{len(rows)}</div><div class="stat-label">Total Connections</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if drugs:
                with st.expander(f"✅ {len(drugs)} Drugs that treat {disease_input}"):
                    cols = st.columns(3)
                    for i, d in enumerate(drugs[:30]):
                        with cols[i % 3]:
                            st.markdown(f"💊 {d}")

            if contra:
                with st.expander(f"⚠️ {len(contra)} Contraindicated Drugs"):
                    cols = st.columns(3)
                    for i, d in enumerate(contra[:30]):
                        with cols[i % 3]:
                            st.markdown(f"🔴 {d}")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🕸️ Knowledge Graph")

            G3 = nx.DiGraph()
            shown  = {"indication": 6, "contraindication": 4, "drug_effect": 4, "disease_disease": 4}
            counts = {k: 0 for k in shown}
            for r in rows:
                rel = r["relation"]
                if rel in shown and counts.get(rel, 0) < shown[rel]:
                    h = r["head"][:20] + "..." if len(r["head"]) > 20 else r["head"]
                    t = r["tail"][:20] + "..." if len(r["tail"]) > 20 else r["tail"]
                    G3.add_edge(h, t, label=rel)
                    counts[rel] = counts.get(rel, 0) + 1

            if G3.number_of_nodes() > 0:
                from pyvis.network import Network
                net3 = Network(height="500px", width="100%", bgcolor="#f8fafc", font_color="#2d3748")
                disease_lower = disease_input.lower()[:10]
                for node in G3.nodes():
                    if disease_lower in node.lower():
                        net3.add_node(node, color="#2c5f8a", size=45, font={"color":"#ffffff","size":15,"bold":True}, shape="ellipse")
                    elif any(node == d[:20] or node == d[:20]+"..." for d in drugs):
                        net3.add_node(node, color="#27ae60", size=28, font={"color":"#ffffff","size":11}, shape="ellipse")
                    elif any(node == d[:20] or node == d[:20]+"..." for d in contra):
                        net3.add_node(node, color="#c0392b", size=25, font={"color":"#ffffff","size":11}, shape="ellipse")
                    else:
                        net3.add_node(node, color="#7fa8c9", size=22, font={"color":"#1a202c","size":10}, shape="ellipse")

                edge_colors = {"indication":"#27ae60","contraindication":"#c0392b","drug_effect":"#e67e22","disease_disease":"#8e44ad"}
                for u, v, data in G3.edges(data=True):
                    lbl = data.get("label","")
                    net3.add_edge(u, v, label=lbl, color=edge_colors.get(lbl,"#5a8ab0"),
                                  width=2, font={"color":edge_colors.get(lbl,"#5a8ab0"),"size":9}, arrows="to")

                net3.set_options('{"physics":{"stabilization":{"iterations":300},"repulsion":{"nodeDistance":230,"springLength":230,"springConstant":0.02}},"layout":{"improvedLayout":true}}')
                st.components.v1.html(net3.generate_html(), height=520)
                st.markdown('<div style="display:flex;gap:16px;padding:8px 0;font-size:0.85em"><span style="color:#2c5f8a">🔵 Disease</span><span style="color:#27ae60">🟢 Treats</span><span style="color:#c0392b">🔴 Contraindicated</span><span style="color:#7fa8c9">🔹 Other</span></div>', unsafe_allow_html=True)

# ── TAB 6: KG STATISTICS ──────────────────────────────────────────────────────
with tab6:
    st.markdown("### 📈 Knowledge Graph Statistics")
    st.markdown('<p style="color:#2d3748">Visual overview of your PrimeKG medical knowledge graph stored in Neo4j.</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.spinner("Loading KG statistics from Neo4j..."):
        with driver.session(database="neo4j") as session:
            node_result = session.run("""
                MATCH (n) WHERE n.type IS NOT NULL
                RETURN n.type AS type, count(n) AS count
                ORDER BY count DESC LIMIT 10
            """)
            node_data = [(r["type"], r["count"]) for r in node_result]

            rel_result = session.run("""
                MATCH ()-[r]->() WHERE r.type IS NOT NULL
                RETURN r.type AS type, count(r) AS count
                ORDER BY count DESC LIMIT 10
            """)
            rel_data = [(r["type"], r["count"]) for r in rel_result]

            top_diseases = session.run("""
                MATCH (n {type: 'disease'})-[r]-()
                RETURN n.name AS name, count(r) AS connections
                ORDER BY connections DESC LIMIT 10
            """)
            disease_data = [(r["name"], r["connections"]) for r in top_diseases]

            top_drugs = session.run("""
                MATCH (n {type: 'drug'})-[r]-()
                RETURN n.name AS name, count(r) AS connections
                ORDER BY connections DESC LIMIT 10
            """)
            drug_data = [(r["name"], r["connections"]) for r in top_drugs]

    st.markdown("#### 🗃️ Overall Graph Size")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="stat-card"><div class="stat-number" style="color:#2c5f8a">31,388</div><div class="stat-label">Total Nodes</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="stat-card"><div class="stat-number" style="color:#2c5f8a">284,089</div><div class="stat-label">Total Relationships</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="stat-card"><div class="stat-number" style="color:#2c5f8a">5</div><div class="stat-label">Relationship Types</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown("#### 🔵 Nodes by Type")
        if node_data:
            st.bar_chart(pd.DataFrame(node_data, columns=["Type","Count"]).set_index("Type"))
    with right:
        st.markdown("#### 🔗 Relationships by Type")
        if rel_data:
            st.bar_chart(pd.DataFrame(rel_data, columns=["Type","Count"]).set_index("Type"))

    st.markdown("<br>", unsafe_allow_html=True)
    left2, right2 = st.columns(2)
    with left2:
        st.markdown("#### 🦠 Most Connected Diseases")
        if disease_data:
            st.dataframe(pd.DataFrame(disease_data, columns=["Disease","Connections"]),
                         use_container_width=True, hide_index=True)
    with right2:
        st.markdown("#### 💊 Most Connected Drugs")
        if drug_data:
            st.dataframe(pd.DataFrame(drug_data, columns=["Drug","Connections"]),
                         use_container_width=True, hide_index=True)