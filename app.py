"""
🦁 SmartZoo AI - Bilingual Animal Classification with AI Assistant
English & Urdu Support | Voice Input | Powered by Groq LLM + Helsinki-NLP
RAG Knowledge Base System Integrated
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import os
import json
from groq import Groq
from PIL import Image
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import tempfile

# ============================================================
# PAGE CONFIGURATION - DARK THEME
# ============================================================
st.set_page_config(
    page_title="SmartZoo AI - Bilingual Animal Classifier",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# REDESIGNED CSS — Emerald Forest × Midnight Gold Theme
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

    :root {
        --bg-base:       #060d0a;
        --bg-surface:    #0c1a14;
        --bg-elevated:   #112219;
        --bg-card:       #0f1f17;
        --accent-primary:#00e87a;
        --accent-gold:   #ffc84a;
        --accent-teal:   #00c4b4;
        --accent-rose:   #ff5e7d;
        --text-primary:  #e8f5ee;
        --text-secondary:#7eb894;
        --text-muted:    #3d6651;
        --border-glow:   rgba(0,232,122,0.25);
        --border-subtle: rgba(0,232,122,0.1);
        --shadow-green:  0 0 40px rgba(0,232,122,0.08);
        --shadow-gold:   0 0 40px rgba(255,200,74,0.12);
        --radius-sm:     8px;
        --radius-md:     16px;
        --radius-lg:     24px;
        --radius-xl:     32px;
    }

    * { box-sizing: border-box; }

    html, body, .stApp {
        background-color: var(--bg-base) !important;
        font-family: 'DM Sans', sans-serif !important;
        color: var(--text-primary) !important;
    }

    [data-testid="stSidebar"] {
        background: var(--bg-surface) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }

    [data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }

    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: var(--bg-base); }
    ::-webkit-scrollbar-thumb { background: var(--accent-primary); border-radius: 4px; }

    .main-header {
        position: relative;
        overflow: hidden;
        background: var(--bg-surface);
        padding: 3rem 2.5rem;
        border-radius: var(--radius-xl);
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid var(--border-glow);
        box-shadow: var(--shadow-green), inset 0 1px 0 rgba(0,232,122,0.1);
    }

    .main-header::before {
        content: '';
        position: absolute;
        inset: 0;
        background-image:
            linear-gradient(rgba(0,232,122,0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,232,122,0.04) 1px, transparent 1px);
        background-size: 32px 32px;
        border-radius: var(--radius-xl);
        pointer-events: none;
    }

    .main-header::after {
        content: '';
        position: absolute;
        top: -60px; left: 50%;
        transform: translateX(-50%);
        width: 320px; height: 200px;
        background: radial-gradient(ellipse, rgba(0,232,122,0.18) 0%, transparent 70%);
        pointer-events: none;
    }

    .main-header h1 {
        position: relative;
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: clamp(2rem, 5vw, 3.5rem);
        color: var(--text-primary);
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
    }

    .main-header h1 .heading-text {
        background: linear-gradient(135deg, #ffffff 30%, var(--accent-primary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .main-header p {
        position: relative;
        color: var(--text-secondary);
        font-size: clamp(0.9rem, 2vw, 1.15rem);
        font-weight: 300;
        margin: 0;
    }

    .bilingual-badge {
        position: relative;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-teal));
        color: #060d0a;
        padding: 0.4rem 1.2rem;
        border-radius: 100px;
        margin: 0.8rem 0 0.5rem;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        box-shadow: 0 4px 20px rgba(0,232,122,0.35);
    }

    /* ── RAG Section ─────────────────────────────────── */
    .rag-section {
        background: var(--bg-surface);
        border: 1px solid rgba(0,196,180,0.3);
        border-radius: var(--radius-xl);
        padding: 2rem 2rem 1.5rem;
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
    }

    .rag-section::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-teal), var(--accent-primary), var(--accent-gold));
    }

    .rag-section::after {
        content: '';
        position: absolute;
        top: -80px; right: -80px;
        width: 260px; height: 260px;
        background: radial-gradient(ellipse, rgba(0,196,180,0.08) 0%, transparent 70%);
        pointer-events: none;
    }

    .rag-header {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin-bottom: 0.4rem;
    }

    .rag-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: rgba(0,196,180,0.15);
        color: var(--accent-teal);
        border: 1px solid rgba(0,196,180,0.35);
        padding: 0.25rem 0.75rem;
        border-radius: 100px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .rag-result-box {
        background: var(--bg-elevated);
        border: 1px solid rgba(0,232,122,0.25);
        border-left: 4px solid var(--accent-primary);
        border-radius: var(--radius-md);
        padding: 1.4rem 1.2rem;
        margin: 1rem 0;
        position: relative;
    }

    .rag-result-label {
        font-family: 'Syne', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--accent-primary);
        margin-bottom: 0.4rem;
    }

    .rag-result-field {
        font-size: 0.78rem;
        color: var(--text-secondary);
        margin-bottom: 0.3rem;
    }

    .rag-result-answer {
        font-size: 1rem;
        color: var(--text-primary);
        line-height: 1.6;
        margin: 0;
    }

    .rag-result-footer {
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-top: 0.8rem;
        padding-top: 0.6rem;
        border-top: 1px dashed rgba(0,232,122,0.15);
    }

    .rag-sample-chip {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 100px;
        padding: 0.35rem 0.85rem;
        font-size: 0.82rem;
        color: var(--text-secondary);
        cursor: pointer;
        transition: all 0.2s;
        display: inline-block;
    }

    .rag-sample-chip:hover {
        border-color: var(--accent-teal);
        color: var(--accent-teal);
    }

    .rag-kb-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin: 0.6rem 0 1rem;
    }

    .rag-kb-pill {
        background: rgba(0,196,180,0.08);
        border: 1px solid rgba(0,196,180,0.2);
        border-radius: 100px;
        padding: 0.2rem 0.7rem;
        font-size: 0.75rem;
        color: var(--accent-teal);
        font-weight: 500;
    }

    .rag-upload-hint {
        background: rgba(255,200,74,0.07);
        border: 1px solid rgba(255,200,74,0.2);
        border-radius: var(--radius-md);
        padding: 0.8rem 1rem;
        margin-top: 1rem;
        font-size: 0.85rem;
        color: var(--accent-gold);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Feature Cards ──────────────────────────────── */
    .feature-card {
        background: var(--bg-card);
        padding: 1.6rem 1.2rem;
        border-radius: var(--radius-md);
        text-align: center;
        border: 1px solid var(--border-subtle);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        height: 100%;
        position: relative;
        overflow: hidden;
    }

    .feature-card::before {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent-primary), var(--accent-teal));
        transform: scaleX(0);
        transform-origin: left;
        transition: transform 0.3s ease;
    }

    .feature-card:hover {
        transform: translateY(-6px);
        border-color: var(--border-glow);
        box-shadow: 0 12px 40px rgba(0,232,122,0.15);
    }

    .feature-card:hover::before { transform: scaleX(1); }

    .feature-card h3 {
        font-size: 2rem;
        margin-bottom: 0.4rem;
    }

    .feature-card h3, .feature-card h4, .feature-card p, .feature-card small {
        color: var(--text-primary) !important;
    }

    .feature-card p {
        font-family: 'Syne', sans-serif;
        font-weight: 600;
        font-size: 0.9rem;
        color: var(--accent-primary) !important;
        margin-bottom: 0.3rem;
    }

    .feature-card small {
        color: var(--text-secondary) !important;
        font-size: 0.8rem;
        line-height: 1.4;
    }

    /* ── Prediction Box ─────────────────────────────── */
    .prediction-box {
        background: var(--bg-elevated);
        padding: 2rem 1.5rem;
        border-radius: var(--radius-lg);
        text-align: center;
        margin: 1rem 0;
        border: 1px solid var(--border-glow);
        box-shadow: var(--shadow-green);
        position: relative;
        overflow: hidden;
    }

    .prediction-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-primary), var(--accent-gold), var(--accent-teal));
    }

    .prediction-box h2 {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: clamp(1.6rem, 3vw, 2.2rem);
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-gold));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.4rem;
    }

    .prediction-box p {
        color: var(--text-secondary) !important;
        font-size: 1rem;
    }

    /* ── Chat Messages ──────────────────────────────── */
    .user-message {
        background: linear-gradient(135deg, #0a2518, #0f2e1f);
        color: var(--text-primary);
        padding: 0.9rem 1.1rem;
        border-radius: var(--radius-md) var(--radius-sm) var(--radius-sm) var(--radius-md);
        margin: 0.6rem 0;
        text-align: right;
        border: 1px solid rgba(0,232,122,0.2);
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .assistant-message {
        background: var(--bg-card);
        color: var(--text-primary);
        padding: 0.9rem 1.1rem;
        border-radius: var(--radius-sm) var(--radius-md) var(--radius-md) var(--radius-sm);
        margin: 0.6rem 0;
        text-align: left;
        border: 1px solid var(--border-subtle);
        border-left: 3px solid var(--accent-primary);
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .urdu-text {
        font-family: 'Noto Nastaliq Urdu', 'Urdu Typesetting', 'Jameel Noori Nastaleeq', serif;
        font-size: 1.05rem;
        direction: rtl;
        text-align: right;
        color: var(--accent-gold);
        margin-top: 0.6rem;
        padding-top: 0.6rem;
        border-top: 1px dashed rgba(255,200,74,0.3);
    }

    /* ── Info Boxes ─────────────────────────────────── */
    .info-box {
        background: var(--bg-card);
        padding: 1.2rem 1.3rem;
        border-radius: var(--radius-md);
        margin: 0.6rem 0;
        border: 1px solid var(--border-subtle);
        border-left: 3px solid var(--accent-primary);
        transition: border-color 0.2s;
    }

    .info-box:hover {
        border-left-color: var(--accent-gold);
        border-color: rgba(255,200,74,0.2);
    }

    .info-box h4 {
        font-family: 'Syne', sans-serif;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--accent-primary) !important;
        margin-bottom: 0.4rem;
    }

    .info-box p {
        color: var(--text-primary) !important;
        font-size: 0.92rem;
        line-height: 1.5;
        margin: 0;
    }

    /* ── Stat Cards ─────────────────────────────────── */
    .stat-card {
        background: var(--bg-card);
        color: var(--text-primary);
        padding: 1.4rem 1rem;
        border-radius: var(--radius-md);
        text-align: center;
        border: 1px solid var(--border-subtle);
        transition: transform 0.2s, box-shadow 0.2s;
        position: relative;
        overflow: hidden;
    }

    .stat-card::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent-primary), var(--accent-teal));
    }

    .stat-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0,232,122,0.12);
    }

    .stat-card h3 {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: clamp(1.6rem, 3vw, 2rem);
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-gold));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
    }

    .stat-card p {
        color: var(--text-secondary) !important;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 500;
        margin: 0;
    }

    /* ── Voice Section ──────────────────────────────── */
    .voice-section {
        background: var(--bg-elevated);
        padding: 1.2rem;
        border-radius: var(--radius-lg);
        margin: 1rem 0;
        border: 1px solid rgba(0,196,180,0.25);
        box-shadow: 0 0 30px rgba(0,196,180,0.06);
    }

    /* ── Footer ─────────────────────────────────────── */
    .footer {
        text-align: center;
        padding: 2.5rem 2rem;
        background: var(--bg-surface);
        border-radius: var(--radius-xl);
        margin-top: 3rem;
        border: 1px solid var(--border-subtle);
        position: relative;
        overflow: hidden;
    }

    .footer::before {
        content: '';
        position: absolute;
        top: 0; left: 20%; right: 20%;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--accent-primary), transparent);
    }

    .footer p {
        color: var(--text-secondary) !important;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }

    /* ── Buttons ─────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-primary) 0%, #00b85e 100%);
        color: #060d0a !important;
        border: none !important;
        padding: 0.55rem 1.8rem !important;
        border-radius: 100px !important;
        font-weight: 700 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.01em !important;
        width: 100% !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
        box-shadow: 0 4px 16px rgba(0,232,122,0.25) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(0,232,122,0.4) !important;
        background: linear-gradient(135deg, #1fffa0 0%, #00d870 100%) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── Text Input ──────────────────────────────────── */
    .stTextInput > div > div > input {
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-glow) !important;
        border-radius: var(--radius-md) !important;
        font-family: 'DM Sans', sans-serif !important;
        padding: 0.6rem 1rem !important;
        font-size: 0.95rem !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 3px rgba(0,232,122,0.12) !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: var(--text-muted) !important;
    }

    /* ── Streamlit Overrides ─────────────────────────── */
    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: var(--text-primary) !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Syne', sans-serif !important;
        color: var(--text-primary) !important;
    }

    .stMarkdown h3 {
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
        color: var(--text-primary) !important;
        margin-top: 1.5rem !important;
    }

    .stRadio label, .stRadio div {
        color: var(--text-primary) !important;
    }

    .stRadio [data-testid="stMarkdownContainer"] p {
        color: var(--text-secondary) !important;
        font-size: 0.85rem !important;
    }

    .stAlert {
        background: var(--bg-elevated) !important;
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--border-subtle) !important;
    }

    [data-testid="stFileUploader"] {
        background: var(--bg-card) !important;
        border: 1px dashed var(--border-glow) !important;
        border-radius: var(--radius-md) !important;
        transition: border-color 0.2s !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent-primary) !important;
    }

    [data-testid="stAudioInput"] {
        background: var(--bg-elevated) !important;
        border-radius: var(--radius-md) !important;
        border: 1px solid rgba(0,196,180,0.2) !important;
        padding: 0.5rem !important;
    }

    .stSpinner > div { border-top-color: var(--accent-primary) !important; }

    .js-plotly-plot .plotly {
        border-radius: var(--radius-md) !important;
    }

    hr {
        border: none !important;
        border-top: 1px solid var(--border-subtle) !important;
        margin: 1.5rem 0 !important;
    }

    @media (max-width: 1024px) {
        .main-header { padding: 2rem 1.5rem; }
        .feature-card { padding: 1.2rem 1rem; }
        .stat-card { padding: 1rem 0.8rem; }
        .stat-card h3 { font-size: 1.5rem; }
        .rag-section { padding: 1.5rem 1.2rem; }
    }

    @media (max-width: 768px) {
        .main-header { padding: 1.6rem 1rem; border-radius: var(--radius-lg); }
        .main-header h1 { font-size: 2rem; }
        .main-header p { font-size: 0.9rem; }
        .bilingual-badge { font-size: 0.78rem; padding: 0.3rem 0.9rem; }
        .feature-card { padding: 1rem 0.8rem; border-radius: var(--radius-sm); }
        .feature-card h3 { font-size: 1.6rem; }
        .prediction-box { padding: 1.4rem 1rem; }
        .prediction-box h2 { font-size: 1.5rem; }
        .info-box { padding: 1rem; }
        .info-box h4 { font-size: 0.8rem; }
        .stat-card h3 { font-size: 1.4rem; }
        .stat-card p { font-size: 0.75rem; }
        .user-message, .assistant-message { font-size: 0.88rem; padding: 0.75rem 0.9rem; }
        .urdu-text { font-size: 0.95rem; }
        .footer { padding: 1.5rem 1rem; border-radius: var(--radius-lg); }
        .footer p { font-size: 0.82rem; }
        .stButton > button { font-size: 0.85rem !important; padding: 0.5rem 1rem !important; }
        .rag-section { padding: 1.2rem 1rem; border-radius: var(--radius-lg); }
    }

    @media (max-width: 480px) {
        .main-header h1 { font-size: 1.6rem; }
        .prediction-box h2 { font-size: 1.2rem; }
        .feature-card p { font-size: 0.8rem; }
        .rag-result-answer { font-size: 0.9rem; }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD ANIMAL DATABASE FROM JSON
# ============================================================
@st.cache_data
def load_animal_database():
    """
    Load animal database from animals_data.json
    
    Required JSON structure (array of objects):
    [
      {
        "name": "Lion",                          ← required, used for matching
        "scientific_name": "Panthera leo",       ← required
        "habitat": "Savannas of Africa",         ← required
        "habitat_ur": "افریقہ کے سوانا",         ← optional Urdu field
        "diet": "Carnivore - large mammals",     ← required
        "diet_ur": "گوشت خور",                   ← optional Urdu field
        "conservation": "Vulnerable",            ← required
        "conservation_ur": "خطرے سے دوچار",      ← optional Urdu field
        "lifespan": "10-14 years",               ← required
        "lifespan_ur": "10-14 سال",              ← optional Urdu field
        "fun_fact": "Lions live in prides",      ← required
        "fun_fact_ur": "شیر گروپوں میں رہتے ہیں", ← optional Urdu field
        "speed": "80 km/h",                      ← optional
        "speed_ur": "80 کلومیٹر",                ← optional Urdu
        "weight": "120-250 kg",                  ← optional
        "weight_ur": "120-250 کلوگرام",          ← optional Urdu
        "region": "Sub-Saharan Africa",          ← optional
        "region_ur": "جنوبی افریقہ",             ← optional Urdu
        "tags": ["big cat","predator","africa"]  ← required, used for RAG search
      },
      ...
    ]
    """
    try:
        with open("animals_data.json", "r", encoding="utf-8") as f:
            animals = json.load(f)
        return animals
    except Exception as e:
        st.warning(f"Could not load animals_data.json: {e}")
        return []


def find_animal_in_json(animal_name, animals):
    """Find animal in JSON database by name matching"""
    if not animals:
        return None

    animal_name_lower = animal_name.strip().lower()

    for animal in animals:
        if animal["name"].lower() == animal_name_lower:
            return animal

    best_match = None
    best_score = 0

    for animal in animals:
        name = animal["name"].lower()
        if animal_name_lower in name or name in animal_name_lower:
            score = len(set(animal_name_lower.split()) & set(name.split()))
            if score > best_score:
                best_score = score
                best_match = animal

        if "tags" in animal:
            for tag in animal["tags"]:
                if tag.lower() in animal_name_lower:
                    if best_match is None:
                        best_match = animal
                        break

    return best_match


# ============================================================
# RAG RETRIEVAL FUNCTION
# ============================================================
def rag_retrieve(query, animals_json):
    """
    Retrieve the best matching animal and relevant field from JSON knowledge base.
    Returns (matched_animal, field_label, answer_en, answer_ur) or None.
    
    This is the core RAG 'Retrieval' step:
    - animals_data.json acts as the Vector Store / Knowledge Base
    - Keyword matching acts as the retriever
    - Groq LLM then 'Augments' and 'Generates' the final answer
    """
    if not animals_json:
        return None, None, None, None

    query_lower = query.lower()
    matched_animal = None

    # Step 1 — find animal by name or tag in query
    for animal in animals_json:
        name_match = animal["name"].lower() in query_lower
        tag_match = any(tag.lower() in query_lower for tag in animal.get("tags", []))
        if name_match or tag_match:
            matched_animal = animal
            break

    if not matched_animal:
        return None, None, None, None

    # Step 2 — determine which field the user is asking about
    diet_kws    = ["eat", "food", "diet", "feed", "prey", "کھاتا", "خوراک", "غذا"]
    habitat_kws = ["live", "habitat", "where", "home", "found", "رہتا", "رہائش", "کہاں", "مسکن"]
    lifespan_kws= ["long", "lifespan", "age", "old", "live for", "زندہ", "عمر", "کتنا جیتا", "زندگی"]
    conserv_kws = ["endanger", "conservation", "extinct", "threat", "status",
                   "معدوم", "تحفظ", "خطرہ", "محفوظ"]
    speed_kws   = ["fast", "speed", "run", "رفتار", "تیز"]
    weight_kws  = ["weight", "heavy", "weigh", "وزن", "بھاری"]

    if any(k in query_lower for k in diet_kws):
        field_label = "Diet / خوراک"
        answer_en = matched_animal.get("diet", "N/A")
        answer_ur = matched_animal.get("diet_ur", answer_en)
    elif any(k in query_lower for k in habitat_kws):
        field_label = "Habitat / رہائش گاہ"
        answer_en = matched_animal.get("habitat", "N/A")
        answer_ur = matched_animal.get("habitat_ur", answer_en)
    elif any(k in query_lower for k in lifespan_kws):
        field_label = "Lifespan / زندگی کی مدت"
        answer_en = matched_animal.get("lifespan", "N/A")
        answer_ur = matched_animal.get("lifespan_ur", answer_en)
    elif any(k in query_lower for k in conserv_kws):
        field_label = "Conservation Status / تحفظ کی حیثیت"
        answer_en = matched_animal.get("conservation", "N/A")
        answer_ur = matched_animal.get("conservation_ur", answer_en)
    elif any(k in query_lower for k in speed_kws):
        field_label = "Speed / رفتار"
        answer_en = matched_animal.get("speed", "N/A")
        answer_ur = matched_animal.get("speed_ur", answer_en)
    elif any(k in query_lower for k in weight_kws):
        field_label = "Weight / وزن"
        answer_en = matched_animal.get("weight", "N/A")
        answer_ur = matched_animal.get("weight_ur", answer_en)
    else:
        field_label = "Fun Fact / دلچسپ حقیقت"
        answer_en = matched_animal.get("fun_fact", "N/A")
        answer_ur = matched_animal.get("fun_fact_ur", answer_en)

    return matched_animal, field_label, answer_en, answer_ur


# ============================================================
# INITIALIZE TRANSLATION MODEL
# ============================================================
@st.cache_resource
def load_translation_model():
    try:
        model_name = "Helsinki-NLP/opus-mt-en-ur"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        return None, None


def translate_to_urdu(text, tokenizer, model):
    if tokenizer is None or model is None:
        return None
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=200)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except:
        return None


# ============================================================
# INITIALIZE GROQ CLIENT
# ============================================================
@st.cache_resource
def init_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key is None:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except:
            api_key = None
    if api_key:
        return Groq(api_key=api_key)
    return None


# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model_and_labels():
    try:
        model = tf.keras.models.load_model("animal_mobilenet_model.keras")
        with open("labels.pkl", "rb") as f:
            labels = pickle.load(f)
        class_names = list(labels.keys())
        return model, labels, class_names
    except Exception as e:
        return None, None, None


# ============================================================
# FALLBACK ANIMAL INFORMATION DATABASE
# ============================================================
animal_info_fallback = {
    "Lion": {
        "habitat": "Savannas and grasslands of Africa",
        "habitat_ur": "افریقہ کے سوانا اور گھاس کے میدان",
        "diet": "Carnivore - primarily large mammals",
        "diet_ur": "گوشت خور - بنیادی طور پر بڑے جانور",
        "conservation": "Vulnerable - population decreasing",
        "conservation_ur": "خطرے سے دوچار - آبادی کم ہو رہی ہے",
        "lifespan": "10-14 years in wild",
        "lifespan_ur": "جنگل میں 10-14 سال",
        "fun_fact": "Lions live in social groups called prides",
        "fun_fact_ur": "شیر سماجی گروپوں میں رہتے ہیں",
        "scientific_name": "Panthera leo"
    },
    "Tiger": {
        "habitat": "Rainforests and grasslands of Asia",
        "habitat_ur": "ایشیا کے برساتی جنگلات",
        "diet": "Carnivore - deer, wild boar",
        "diet_ur": "گوشت خور - ہرن، جنگلی سور",
        "conservation": "Endangered - only 3,900 remain",
        "conservation_ur": "خطرے سے دوچار - صرف 3,900 باقی",
        "lifespan": "8-10 years in wild",
        "lifespan_ur": "جنگل میں 8-10 سال",
        "fun_fact": "Each tiger has unique stripe patterns",
        "fun_fact_ur": "ہر شیر کے پٹیوں کے منفرد نمونے",
        "scientific_name": "Panthera tigris"
    },
    "Elephant": {
        "habitat": "Savannas and forests of Africa and Asia",
        "habitat_ur": "افریقہ اور ایشیا کے جنگلات",
        "diet": "Herbivore - grasses, fruits, bark",
        "diet_ur": "سبزی خور - گھاس، پھل، چھال",
        "conservation": "Endangered",
        "conservation_ur": "خطرے سے دوچار",
        "lifespan": "60-70 years",
        "lifespan_ur": "60-70 سال",
        "fun_fact": "Elephants can recognize themselves in mirrors",
        "fun_fact_ur": "ہاتھی آئینے میں خود کو پہچان سکتے ہیں",
        "scientific_name": "Loxodonta africana"
    },
    "Giraffe": {
        "habitat": "Savannas of Africa",
        "habitat_ur": "افریقہ کے سوانا",
        "diet": "Herbivore - acacia leaves",
        "diet_ur": "سبزی خور - ببول کے پتے",
        "conservation": "Vulnerable",
        "conservation_ur": "خطرے سے دوچار",
        "lifespan": "20-25 years",
        "lifespan_ur": "20-25 سال",
        "fun_fact": "Giraffes have 7 neck vertebrae like humans",
        "fun_fact_ur": "زرافوں کی گردن میں 7 vertebrae",
        "scientific_name": "Giraffa camelopardalis"
    }
}


def get_fallback_animal_info(animal_name, language="english"):
    for key in animal_info_fallback:
        if key.lower() in animal_name.lower():
            info = animal_info_fallback[key]
            if language == "urdu":
                return {
                    "habitat": info.get("habitat_ur", info["habitat"]),
                    "diet": info.get("diet_ur", info["diet"]),
                    "conservation": info.get("conservation_ur", info["conservation"]),
                    "lifespan": info.get("lifespan_ur", info["lifespan"]),
                    "fun_fact": info.get("fun_fact_ur", info["fun_fact"]),
                    "scientific_name": info["scientific_name"]
                }
            return info
    return {
        "habitat": "Information not available",
        "diet": "Information not available",
        "conservation": "Information not available",
        "lifespan": "Information not available",
        "fun_fact": "This amazing animal needs our protection",
        "scientific_name": "Information not available"
    }


# ============================================================
# LLM CHAT FUNCTION
# ============================================================
def get_llm_response(client, animal_name, user_question, is_urdu=False,
                     tokenizer=None, model=None, retrieved_context=None):
    """
    RAG-enhanced LLM response.
    retrieved_context: if provided, injected into the system prompt so the
    LLM augments a retrieved fact rather than hallucinating from scratch.
    """
    if client is None:
        return "⚠️ Groq API key not configured. Please add your API key to continue.", None

    context_block = ""
    if retrieved_context:
        context_block = f"\n\nRetrieved fact from knowledge base: {retrieved_context}\nUse this fact as your primary source. Expand on it naturally."

    if is_urdu:
        system_prompt = f"""آپ SmartZoo AI اسسٹنٹ ہیں، ایک سنجیدہ ماہرِ حیاتِ وحش۔
آپ {animal_name} کے بارے میں بات کر رہے ہیں۔{context_block}
صرف درست، سائنسی، اور تعلیمی معلومات دیں۔
مذاق، طنز، یا غیر ضروری تبصرہ بالکل نہ کریں۔
جواب مختصر رکھیں — زیادہ سے زیادہ 2 جملے۔
صرف ان موضوعات پر بات کریں: رہائش گاہ، خوراک، رویہ، یا تحفظ کی حیثیت۔
جواب صرف اردو میں دیں۔"""
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_question}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=300,
            )
            return chat_completion.choices[0].message.content.strip(), None
        except Exception as e:
            return f"⚠️ خرابی: {str(e)}", None
    else:
        system_prompt = f"""You are SmartZoo AI Assistant, a serious wildlife expert.
You are discussing {animal_name}. Provide ONLY factual, accurate, educational information.{context_block}
DO NOT include jokes, sarcasm, or humorous content.
Keep responses concise (1-2 sentences max).
Focus ONLY on: habitat, diet, behavior, or conservation status."""
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_question}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=200,
            )
            return chat_completion.choices[0].message.content.strip(), None
        except Exception as e:
            return f"⚠️ Error: {str(e)}", None


# ============================================================
# SPEECH-TO-TEXT
# ============================================================
def speech_to_text_groq(client, audio_bytes):
    try:
        if hasattr(audio_bytes, 'read'):
            audio_data = audio_bytes.read()
        else:
            audio_data = audio_bytes

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_data)
            tmp_file_path = tmp_file.name

        with open(tmp_file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(tmp_file_path, file.read()),
                model="whisper-large-v3",
                response_format="json",
                language="en",
            )

        os.unlink(tmp_file_path)
        return transcription.text
    except Exception as e:
        return None


# ============================================================
# PRE-PROCESSING FUNCTIONS
# ============================================================
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array


def create_confidence_chart(probabilities, class_names, top_n=5):
    top_indices = np.argsort(probabilities)[-top_n:][::-1]
    top_names = [class_names[i] for i in top_indices]
    top_probs = [probabilities[i] * 100 for i in top_indices]

    fig = go.Figure(data=[
        go.Bar(
            x=top_probs,
            y=top_names,
            orientation='h',
            marker=dict(
                color=top_probs,
                colorscale=[[0, '#0f2e1f'], [0.5, '#00b85e'], [1, '#00e87a']],
                showscale=True,
                colorbar=dict(tickfont=dict(color='#7eb894'))
            ),
            text=[f"{p:.1f}%" for p in top_probs],
            textposition='outside',
            textfont=dict(color='#e8f5ee')
        )
    ])

    fig.update_layout(
        title=dict(text="Top Predictions Confidence Score",
                   font=dict(color='#e8f5ee', family='Syne', size=14)),
        xaxis_title=dict(text="Confidence (%)", font=dict(color='#7eb894')),
        yaxis_title=dict(text="Animal Species", font=dict(color='#7eb894')),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8f5ee', family='DM Sans'),
        xaxis=dict(gridcolor='rgba(0,232,122,0.07)', color='#7eb894'),
        yaxis=dict(gridcolor='rgba(0,232,122,0.07)', color='#7eb894'),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig, top_names, top_probs


def get_suggested_questions(animal_name, language="english"):
    if language == "urdu":
        return [
            f"{animal_name} کیا کھاتا ہے؟",
            f"{animal_name} کہاں رہتا ہے؟",
            f"{animal_name} کے بارے میں ایک دلچسپ حقیقت بتائیں",
            f"کیا {animal_name} معدوم ہونے کے خطرے میں ہے؟",
            f"{animal_name} کتنی دیر زندہ رہتا ہے؟"
        ]
    return [
        f"What does {animal_name} eat?",
        f"Where does {animal_name} live?",
        f"Tell me an interesting fact about {animal_name}",
        f"Is {animal_name} endangered?",
        f"How long does {animal_name} live?"
    ]


# ============================================================
# MAIN APP
# ============================================================
def main():
    # ── Header ──────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>🦁 <span class="heading-text">SmartZoo AI</span></h1>
        <p>Bilingual Animal Classification with AI-Powered Chat Assistant</p>
        <div class="bilingual-badge">🎤 Voice Input | 🇵🇰 English | اردو 🇵🇰</div>
        <p style="font-size: 0.9rem;">Powered by MobileNetV2 + Groq LLM + Helsinki-NLP Translation</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🌐 Language / زبان")
        language = st.radio(
            "Select Language / زبان منتخب کریں",
            ["English", "اردو (Urdu)"],
            index=0
        )

        st.markdown("---")
        st.markdown("### 🏆 SmartZoo AI Features")

        if language == "English":
            st.info("""
            ✅ Real-time Animal Recognition
            ✅ AI Chat Assistant (Groq LLM)
            ✅ Voice Input (Whisper)
            ✅ Bilingual Support (English/Urdu)
            ✅ RAG Knowledge Base System
            ✅ Conservation Information
            ✅ Interactive Confidence Charts
            """)
        else:
            st.info("""
            ✅ حقیقی وقت میں جانوروں کی شناخت
            ✅ AI چیٹ اسسٹنٹ (Groq LLM)
            ✅ آواز سے سوال (Whisper)
            ✅ دو لسانی سپورٹ (انگریزی/اردو)
            ✅ RAG علمی ذخیرہ سسٹم
            ✅ تحفظ کی معلومات
            ✅ انٹرایکٹو اعتماد کے چارٹ
            """)

    is_urdu = (language == "اردو (Urdu)")

    # ── Load resources ───────────────────────────────────────
    model, labels, class_names = load_model_and_labels()
    groq_client = init_groq_client()
    tokenizer, translation_model = load_translation_model()
    animals_json = load_animal_database()

    if model is None:
        st.error("⚠️ Model not loaded. Please check your files.")
        return

    # ── Features Section ─────────────────────────────────────
    if is_urdu:
        st.markdown("### 🌟 خصوصیات")
    else:
        st.markdown("### 🌟 Features")

    col1, col2, col3, col4 = st.columns(4)

    if is_urdu:
        features = [
            ("📸", "فوری شناخت", "جانوروں کی فوری شناخت"),
            ("🤖", "AI چیٹ اسسٹنٹ", "جانوروں کے بارے میں سوالات کریں"),
            ("🎤", "آواز سے سوال", "مائیکروفون سے سوال کریں"),
            ("📚", "RAG سسٹم", "علمی ذخیرے سے جواب")
        ]
    else:
        features = [
            ("📸", "Real-time Recognition", "Instant animal identification"),
            ("🤖", "AI Chat Assistant", "Ask questions about animals"),
            ("🎤", "Voice Input", "Ask by speaking"),
            ("📚", "RAG System", "Knowledge base retrieval")
        ]

    for col, (emoji, title, desc) in zip([col1, col2, col3, col4], features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <h3>{emoji}</h3>
                <p>{title}</p>
                <small>{desc}</small>
            </div>
            """, unsafe_allow_html=True)

    # ============================================================
    # RAG KNOWLEDGE BASE SECTION
    # Shown ONLY when no image is uploaded — auto-hides on upload
    # Connects to: animals_data.json (see load_animal_database docstring)
    # ============================================================

    # We need to know if the file uploader has a file BEFORE rendering RAG.
    # Streamlit evaluates widgets top-to-bottom, so we use session_state to
    # track whether an image has been uploaded in any previous run.
    if "has_uploaded_file" not in st.session_state:
        st.session_state.has_uploaded_file = False

    # Render RAG section only when no image is uploaded
    if not st.session_state.has_uploaded_file:

        st.markdown("---")

        # Build knowledge base pills from JSON
        kb_animals = [a["name"] for a in animals_json[:16]] if animals_json else [
            "Lion", "Tiger", "Elephant", "Giraffe"
        ]
        kb_pills_html = "".join(
            f'<span class="rag-kb-pill">🐾 {name}</span>' for name in kb_animals
        )

        if is_urdu:
            st.markdown("""
            <div class="rag-section">
                <div class="rag-header">
                    <span style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.3rem;color:#e8f5ee;">
                        📚 RAG علمی ذخیرہ سسٹم
                    </span>
                    <span class="rag-badge">🔍 Retrieval Augmented Generation</span>
                </div>
                <p style="color:#7eb894;font-size:0.9rem;margin:0.2rem 0 0.8rem;">
                    تصویر اپ لوڈ کرنے سے پہلے، ہمارے علمی ذخیرے سے براہ راست جانوروں کے بارے میں سوال کریں۔
                    تصویر اپ لوڈ ہوتے ہی یہ سیکشن خود بخود چھپ جائے گا۔
                </p>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="rag-section">
                <div class="rag-header">
                    <span style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.3rem;color:#e8f5ee;">
                        📚 RAG Knowledge Base System
                    </span>
                    <span class="rag-badge">🔍 Retrieval Augmented Generation</span>
                </div>
                <p style="color:#7eb894;font-size:0.9rem;margin:0.2rem 0 0.8rem;">
                    Before uploading an image, query our animal knowledge base directly using RAG.
                    This section <strong style="color:#00e87a;">automatically hides</strong> once you upload an image above.
                </p>
                <p style="color:#3d6651;font-size:0.8rem;margin:0 0 0.5rem;">
                    📂 Knowledge base source: <code style="color:#00c4b4;background:rgba(0,196,180,0.1);
                    padding:0.1rem 0.4rem;border-radius:4px;">animals_data.json</code>
                    &nbsp;·&nbsp; {len(animals_json)} animals loaded
                </p>
                <div class="rag-kb-pills">{kb_pills_html}</div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ── RAG Query Input ──────────────────────────────────
        if is_urdu:
            st.markdown("#### 🔍 علمی ذخیرے سے سوال کریں:")
            rag_placeholder = "مثال: شیر کہاں رہتا ہے؟ یا ہاتھی کیا کھاتا ہے؟"
            rag_btn_label = "🔍 علمی ذخیرہ تلاش کریں"
            rag_samples_label = "**⚡ فوری نمونے:**"
            rag_samples = [
                "شیر کہاں رہتا ہے؟",
                "ہاتھی کیا کھاتا ہے؟",
                "زرافہ کتنا جیتا ہے؟",
                "ببر شیر معدوم ہے کیا؟",
                "شیر کی رفتار کتنی ہے؟",
                "ہاتھی کا وزن کتنا ہے؟"
            ]
        else:
            st.markdown("#### 🔍 Query the Knowledge Base:")
            rag_placeholder = "e.g. Where does a lion live?  |  What does an elephant eat?  |  Is tiger endangered?"
            rag_btn_label = "🔍 Search Knowledge Base"
            rag_samples_label = "**⚡ Quick samples — click to try:**"
            rag_samples = [
                "Where does a lion live?",
                "What does an elephant eat?",
                "How long does a giraffe live?",
                "Is the tiger endangered?",
                "How fast can a lion run?",
                "How heavy is an elephant?"
            ]

        rag_col1, rag_col2 = st.columns([3, 1])
        with rag_col1:
            rag_query_input = st.text_input(
                "", placeholder=rag_placeholder, key="rag_query_input",
                label_visibility="collapsed"
            )
        with rag_col2:
            rag_search_clicked = st.button(
                rag_btn_label, key="rag_search_btn", use_container_width=True
            )

        # ── Sample query chips ───────────────────────────────
        st.markdown(rag_samples_label)
        sample_cols = st.columns(3)
        for idx, sq in enumerate(rag_samples):
            with sample_cols[idx % 3]:
                if st.button(sq, key=f"rag_sample_{idx}", use_container_width=True):
                    st.session_state["rag_active_query"] = sq

        # ── Resolve the active query ─────────────────────────
        active_rag_query = st.session_state.get("rag_active_query", None)
        final_rag_query = None

        if rag_search_clicked and rag_query_input.strip():
            final_rag_query = rag_query_input.strip()
            st.session_state["rag_active_query"] = None   # clear chip state
        elif active_rag_query:
            final_rag_query = active_rag_query

        # ── RAG Pipeline: Retrieve → Augment → Generate ──────
        if final_rag_query:
            with st.spinner(
                "🔍 Searching knowledge base..." if not is_urdu
                else "🔍 علمی ذخیرے میں تلاش ہو رہی ہے..."
            ):
                matched_animal, field_label, answer_en, answer_ur = rag_retrieve(
                    final_rag_query, animals_json
                )

            if matched_animal:
                display_answer  = answer_ur if is_urdu else answer_en
                animal_name_disp = matched_animal["name"]
                sci_name        = matched_animal.get("scientific_name", "")

                # ── Retrieved fact box ───────────────────────
                st.markdown(f"""
                <div class="rag-result-box">
                    <div class="rag-result-label">
                        📚 RAG RESULT &nbsp;·&nbsp; {animal_name_disp}
                        &nbsp;<i style="font-weight:400;color:#7eb894;font-size:0.85rem;">
                        ({sci_name})</i>
                    </div>
                    <div class="rag-result-field">{field_label}</div>
                    <p class="rag-result-answer">{display_answer}</p>
                    <div class="rag-result-footer">
                        ✅ Retrieved from <strong>animals_data.json</strong>
                        &nbsp;·&nbsp; Step 1 of RAG: Retrieval complete
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Groq LLM augments and generates ─────────
                if groq_client:
                    with st.spinner(
                        "🤖 AI augmenting answer..." if not is_urdu
                        else "🤖 AI جواب کو وسیع کر رہا ہے..."
                    ):
                        ai_response, _ = get_llm_response(
                            groq_client,
                            animal_name_disp,
                            final_rag_query,
                            is_urdu,
                            tokenizer,
                            translation_model,
                            retrieved_context=display_answer  # inject retrieved fact
                        )

                    if is_urdu:
                        st.markdown(
                            f'<div class="assistant-message">'
                            f'🤖 AI جواب (RAG): {ai_response}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="assistant-message">'
                            f'🤖 AI (RAG-augmented): {ai_response}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.warning(
                        "⚠️ Groq API key not configured — showing retrieved fact only."
                        if not is_urdu
                        else "⚠️ Groq API key نہیں ملی — صرف بنیادی نتیجہ دکھایا جا رہا ہے۔"
                    )

                # Hint to upload image
                if is_urdu:
                    st.markdown("""
                    <div class="rag-upload-hint">
                        ⬆️ اوپر تصویر اپ لوڈ کریں تاکہ AI خودبخود جانور پہچانے اور یہ سیکشن بند ہو جائے۔
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="rag-upload-hint">
                        ⬆️ Upload an image above to let AI automatically identify the animal
                        — this RAG section will hide automatically.
                    </div>
                    """, unsafe_allow_html=True)

            else:
                if is_urdu:
                    st.warning(
                        "⚠️ علمی ذخیرے میں یہ جانور نہیں ملا۔ "
                        "براہ کرم Lion، Tiger، Elephant جیسے معروف نام استعمال کریں۔"
                    )
                else:
                    st.warning(
                        "⚠️ Animal not found in knowledge base. "
                        "Try names like: Lion, Tiger, Elephant, Giraffe."
                    )

        st.markdown("---")

    # ============================================================
    # IMAGE UPLOAD SECTION
    # ============================================================
    col1, col2 = st.columns([1, 1])

    with col1:
        if is_urdu:
            st.markdown("### 📤 جانور کی تصویر اپ لوڈ کریں")
            upload_text = "تصویر منتخب کریں..."
            help_text   = "جانور کی واضح تصویر اپ لوڈ کریں"
        else:
            st.markdown("### 📤 Upload Animal Image")
            upload_text = "Choose an image..."
            help_text   = "Upload a clear image of an animal"

        uploaded_file = st.file_uploader(
            upload_text,
            type=['jpg', 'jpeg', 'png', 'webp'],
            help=help_text
        )

        # Track upload state so RAG section knows to hide
        if uploaded_file is not None:
            st.session_state.has_uploaded_file = True
        else:
            st.session_state.has_uploaded_file = False

        if uploaded_file:
            img, img_array = preprocess_image(uploaded_file)
            caption = "اپ لوڈ کردہ تصویر" if is_urdu else "Uploaded Image"
            st.image(img, caption=caption, use_container_width=True)

    with col2:
        if uploaded_file:
            if is_urdu:
                st.markdown("### 🔍 درجہ بندی کے نتائج")
            else:
                st.markdown("### 🔍 Classification Results")

            with st.spinner(
                "🦁 Analyzing image..." if not is_urdu
                else "🦁 تصویر کا تجزیہ ہو رہا ہے..."
            ):
                predictions = model.predict(img_array, verbose=0)[0]
                fig, top_names, top_probs = create_confidence_chart(predictions, class_names)

                top_animal     = top_names[0]
                top_confidence = top_probs[0]

                st.markdown(f"""
                <div class="prediction-box">
                    <h2>🐾 {top_animal}</h2>
                    <p>{'Confidence' if not is_urdu else 'اعتماد'}: {top_confidence:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

                st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # ANIMAL INFORMATION SECTION (after upload)
    # ============================================================
    if uploaded_file:
        top_animal  = top_names[0]
        animal_data = find_animal_in_json(top_animal, animals_json)

        st.markdown("---")
        if is_urdu:
            st.markdown(f"### ℹ️ {top_animal} کے بارے میں")
        else:
            st.markdown(f"### ℹ️ About {top_animal}")

        if animal_data:
            if is_urdu:
                habitat       = animal_data.get("habitat_ur",      animal_data.get("habitat",      "N/A"))
                diet          = animal_data.get("diet_ur",          animal_data.get("diet",          "N/A"))
                conservation  = animal_data.get("conservation_ur",  animal_data.get("conservation",  "N/A"))
                lifespan      = animal_data.get("lifespan_ur",      animal_data.get("lifespan",      "N/A"))
                fun_fact      = animal_data.get("fun_fact_ur",      animal_data.get("fun_fact",      "N/A"))
                scientific_name = animal_data.get("scientific_name", "N/A")
                speed  = animal_data.get("speed_ur",  animal_data.get("speed",  None)) if "speed"  in animal_data else None
                weight = animal_data.get("weight_ur", animal_data.get("weight", None)) if "weight" in animal_data else None
                region = animal_data.get("region_ur", animal_data.get("region", None)) if "region" in animal_data else None
            else:
                habitat       = animal_data.get("habitat",       "N/A")
                diet          = animal_data.get("diet",          "N/A")
                conservation  = animal_data.get("conservation",  "N/A")
                lifespan      = animal_data.get("lifespan",      "N/A")
                fun_fact      = animal_data.get("fun_fact",      "N/A")
                scientific_name = animal_data.get("scientific_name", "N/A")
                speed  = animal_data.get("speed",  None)
                weight = animal_data.get("weight", None)
                region = animal_data.get("region", None)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="info-box"><h4>{"🏠 Habitat" if not is_urdu else "🏠 رہائش گاہ"}</h4><p>{habitat}</p></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-box"><h4>{"🍽️ Diet" if not is_urdu else "🍽️ خوراک"}</h4><p>{diet}</p></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="info-box"><h4>{"⚠️ Conservation" if not is_urdu else "⚠️ تحفظ کی حیثیت"}</h4><p>{conservation}</p></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-box"><h4>{"📅 Lifespan" if not is_urdu else "📅 زندگی کی مدت"}</h4><p>{lifespan}</p></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="info-box"><h4>{"🔬 Scientific Name" if not is_urdu else "🔬 سائنسی نام"}</h4><p><i>{scientific_name}</i></p></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-box"><h4>{"✨ Fun Fact" if not is_urdu else "✨ دلچسپ حقیقت"}</h4><p>{fun_fact}</p></div>', unsafe_allow_html=True)

            if speed or weight or region:
                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                if speed:
                    with c1:
                        st.markdown(f'<div class="info-box"><h4>⚡ {"Speed" if not is_urdu else "رفتار"}</h4><p>{speed}</p></div>', unsafe_allow_html=True)
                if weight:
                    with c2:
                        st.markdown(f'<div class="info-box"><h4>⚖️ {"Weight" if not is_urdu else "وزن"}</h4><p>{weight}</p></div>', unsafe_allow_html=True)
                if region:
                    with c3:
                        st.markdown(f'<div class="info-box"><h4>🌍 {"Region" if not is_urdu else "علاقہ"}</h4><p>{region}</p></div>', unsafe_allow_html=True)

        else:
            lang_code = "urdu" if is_urdu else "english"
            adf = get_fallback_animal_info(top_animal, lang_code)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="info-box"><h4>{"🏠 Habitat" if not is_urdu else "🏠 رہائش گاہ"}</h4><p>{adf["habitat"]}</p></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-box"><h4>{"🍽️ Diet" if not is_urdu else "🍽️ خوراک"}</h4><p>{adf["diet"]}</p></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="info-box"><h4>{"⚠️ Conservation" if not is_urdu else "⚠️ تحفظ کی حیثیت"}</h4><p>{adf["conservation"]}</p></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-box"><h4>{"📅 Lifespan" if not is_urdu else "📅 زندگی کی مدت"}</h4><p>{adf["lifespan"]}</p></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="info-box"><h4>{"🔬 Scientific Name" if not is_urdu else "🔬 سائنسی نام"}</h4><p><i>{adf["scientific_name"]}</i></p></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-box"><h4>{"✨ Fun Fact" if not is_urdu else "✨ دلچسپ حقیقت"}</h4><p>{adf["fun_fact"]}</p></div>', unsafe_allow_html=True)

        # ============================================================
        # CHAT SECTION
        # ============================================================
        st.markdown("---")
        if is_urdu:
            st.markdown(f"### 🤖 {top_animal} کے بارے میں AI اسسٹنٹ سے پوچھیں")
        else:
            st.markdown(f"### 🤖 Ask AI Assistant About {top_animal}")

        if "messages" not in st.session_state:
            if is_urdu:
                welcome_msg = f"👋 السلام علیکم! میں {top_animal} کے بارے میں آپ کا AI اسسٹنٹ ہوں۔ اس حیرت انگیز جانور کے بارے میں کچھ بھی پوچھیں!"
            else:
                welcome_msg = f"👋 Hi! I'm your AI assistant for {top_animal}. Ask me anything about this amazing animal!"
            st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]

        for message in st.session_state.messages:
            if message["role"] == "user":
                label = "🧑‍💻 آپ:" if is_urdu else "🧑‍💻 You:"
                st.markdown(f'<div class="user-message">{label} {message["content"]}</div>', unsafe_allow_html=True)
            else:
                label = "🤖 اسسٹنٹ:" if is_urdu else "🤖 Assistant:"
                st.markdown(f'<div class="assistant-message">{label} {message["content"]}</div>', unsafe_allow_html=True)

        if is_urdu:
            st.markdown("#### 💡 پوچھنے کے لیے تجویز کردہ سوالات:")
        else:
            st.markdown("#### 💡 Try asking:")

        suggested_questions = get_suggested_questions(top_animal, "urdu" if is_urdu else "english")
        cols = st.columns(3)
        for idx, question in enumerate(suggested_questions[:5]):
            with cols[idx % 3]:
                if st.button(question, key=f"suggested_{idx}", use_container_width=True):
                    with st.spinner("🤔 Thinking..." if not is_urdu else "🤔 سوچ رہا ہوں..."):
                        st.session_state.messages.append({"role": "user", "content": question})
                        response, _ = get_llm_response(
                            groq_client, top_animal, question,
                            is_urdu, tokenizer, translation_model
                        )
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.rerun()

        if is_urdu:
            st.markdown("#### ✍️ یا اپنا سوال لکھیں:")
            placeholder = f"{top_animal} کے بارے میں کچھ پوچھیں..."
        else:
            st.markdown("#### ✍️ Or type your question:")
            placeholder = f"Ask something about {top_animal}..."

        text_question = st.text_input("", placeholder=placeholder, key="text_input")

        if st.button("Send Question" if not is_urdu else "سوال بھیجیں",
                     key="send_text_btn", use_container_width=True):
            if text_question:
                with st.spinner("🤔 Thinking..." if not is_urdu else "🤔 سوچ رہا ہوں..."):
                    st.session_state.messages.append({"role": "user", "content": text_question})
                    response, _ = get_llm_response(
                        groq_client, top_animal, text_question,
                        is_urdu, tokenizer, translation_model
                    )
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()

        # ============================================================
        # VOICE INPUT SECTION
        # ============================================================
        st.markdown("---")
        if is_urdu:
            st.markdown("### 🎤 آواز سے سوال پوچھیں")
            st.markdown("نیچے دیے گئے مائیکروفون پر کلک کریں، ریکارڈ کریں، پھر 'Process Voice' بٹن دبائیں")
        else:
            st.markdown("### 🎤 Ask Question by Voice")
            st.markdown("Click the microphone below, record, then click 'Process Voice' button")

        audio_value = st.audio_input(
            "Record your voice" if not is_urdu else "اپنی آواز ریکارڈ کریں"
        )

        if "stored_audio" not in st.session_state:
            st.session_state.stored_audio = None

        if audio_value:
            st.session_state.stored_audio = audio_value
            st.success("✅ Audio recorded! Click 'Process Voice' to send.")

        vc1, vc2 = st.columns(2)
        with vc1:
            if st.button(
                "🎤 Process Voice" if not is_urdu else "🎤 آواز پروسیس کریں",
                key="process_voice_btn", use_container_width=True
            ):
                if st.session_state.stored_audio:
                    with st.spinner(
                        "🎤 Processing your voice..." if not is_urdu
                        else "🎤 آواز پر کارروائی ہو رہی ہے..."
                    ):
                        if groq_client:
                            transcribed_text = speech_to_text_groq(
                                groq_client, st.session_state.stored_audio
                            )
                            if transcribed_text:
                                st.success(f"📝 Recognized: {transcribed_text}")
                                st.session_state.messages.append(
                                    {"role": "user", "content": transcribed_text}
                                )
                                response, _ = get_llm_response(
                                    groq_client, top_animal, transcribed_text,
                                    is_urdu, tokenizer, translation_model
                                )
                                st.session_state.messages.append(
                                    {"role": "assistant", "content": response}
                                )
                                st.session_state.stored_audio = None
                                st.rerun()
                            else:
                                st.error("Failed to transcribe. Please try again.")
                        else:
                            st.error("Groq client not configured. Please add GROQ_API_KEY")
                else:
                    st.warning(
                        "No audio recorded. Please record your voice first."
                        if not is_urdu
                        else "کوئی آواز ریکارڈ نہیں کی گئی۔ براہ کرم پہلے آواز ریکارڈ کریں۔"
                    )

        with vc2:
            if st.button(
                "🗑️ Clear Audio" if not is_urdu else "🗑️ آواز صاف کریں",
                key="clear_audio_btn", use_container_width=True
            ):
                st.session_state.stored_audio = None
                st.rerun()

        st.markdown("---")
        if st.button(
            "🗑️ Clear Chat History" if not is_urdu else "🗑️ چیٹ ہسٹری صاف کریں",
            key="clear_chat_btn", use_container_width=True
        ):
            if is_urdu:
                welcome_msg = f"👋 السلام علیکم! میں {top_animal} کے بارے میں آپ کا AI اسسٹنٹ ہوں۔ اس حیرت انگیز جانور کے بارے میں کچھ بھی پوچھیں!"
            else:
                welcome_msg = f"👋 Hi! I'm your AI assistant for {top_animal}. Ask me anything about this amazing animal!"
            st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]
            st.rerun()

    # ============================================================
    # STATISTICS SECTION
    # ============================================================
    st.markdown("---")
    if is_urdu:
        st.markdown("### 📊 ماڈل کے اعدادوشمار")
    else:
        st.markdown("### 📊 Model Statistics")

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.markdown(f'<div class="stat-card"><h3>🎯 85%</h3><p>{"Model Accuracy" if not is_urdu else "ماڈل کی درستگی"}</p></div>', unsafe_allow_html=True)
    with sc2:
        st.markdown(f'<div class="stat-card"><h3>{len(class_names)}+</h3><p>{"Animal Classes" if not is_urdu else "جانوروں کی کلاسیں"}</p></div>', unsafe_allow_html=True)
    with sc3:
        st.markdown(f'<div class="stat-card"><h3>10k+</h3><p>{"Training Images" if not is_urdu else "تربیتی تصاویر"}</p></div>', unsafe_allow_html=True)
    with sc4:
        st.markdown(f'<div class="stat-card"><h3>&lt;0.1s</h3><p>{"Inference Time" if not is_urdu else "تشخیص کا وقت"}</p></div>', unsafe_allow_html=True)

    # ============================================================
    # FOOTER
    # ============================================================
    if is_urdu:
        st.markdown("""
        <div class="footer">
            <p>🐾 SmartZoo AI - ٹیکنالوجی کے ذریعے جنگلی حیات کا تحفظ 🐾</p>
            <p>TensorFlow، MobileNetV2، Groq LLM، Helsinki-NLP اور RAG سسٹم کے ذریعے تقویت یافتہ</p>
            <p style="font-size: 0.8rem;">⚠️ نوٹ: ماڈل کی درستگی تصویر کے معیار کے لحاظ سے مختلف ہو سکتی ہے۔</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="footer">
            <p>🐾 SmartZoo AI - Protecting Wildlife Through Technology 🐾</p>
            <p>Powered by TensorFlow · MobileNetV2 · Groq LLM · Helsinki-NLP · RAG Knowledge Base</p>
            <p style="font-size: 0.8rem;">⚠️ Note: Model accuracy may vary based on image quality.</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()