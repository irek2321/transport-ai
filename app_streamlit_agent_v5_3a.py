
# Agent transportu v5.3a – poprawka filtru arkuszy i komunikatów
import os, re, json, time
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import streamlit as st
from openai import OpenAI

APP_TITLE = "Agent transportu v5.3a (auto arkusze)"

ADDRESS_BOOK = {
    "Tholen": "Marconiweg 3, 4691 SV Tholen",
    "DSV Tholen": "Marconiweg 3, 4691 SV Tholen",
    "DSV Moerdijk": "Tradeboulevard 4, 4761 RL Zevenbergen",
    "Moerdijk – DSV": "Tradeboulevard 4, 4761 RL Zevenbergen",
    "Tradeboulevard 4": "Tradeboulevard 4, 4761 RL Zevenbergen",
}

ADDRESS_COORDS = {
    "Tholen": (51.5468, 4.2208),
    "Marconiweg 3, 4691 SV Tholen": (51.5459, 4.2085),
    "Tradeboulevard 4, 4761 RL Zevenbergen": (51.6574745, 4.5676789),
    "DSV Moerdijk": (51.6574745, 4.5676789),
    "Moerdijk – DSV": (51.6574745, 4.5676789),
}

def norm_time(val): m=re.search(r"(\\d{1,2})",str(val));return f"{int(m.group(1)):02d}:00" if m else None

def split_van_tot(df): return df  # stub; pełne funkcje z v5.3 jeśli potrzebne

def get_api_key():
    try: return st.secrets["OPENAI_API_KEY"]
    except Exception: return os.getenv("OPENAI_API_KEY")

def run_agent(question, plan, drivers, dest_override):
    return f"Demo OK – wczytano {len(plan)} wierszy planu."

# ---------------- UI ----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.subheader("Domyślny cel")
    dest_choice = st.selectbox("Cel", ["Auto", "Tholen", "DSV Moerdijk"], 0)
    dest_override = None if dest_choice=="Auto" else ADDRESS_BOOK[dest_choice]

uploaded = st.file_uploader("Excel z planem", type=[".xlsx",".xlsm"])

if uploaded:
    xls = pd.ExcelFile(uploaded)
    wanted = ("Planing", "Plan", "Tholen", "Moerdijk")
    sheets = [s for s in xls.sheet_names if any(w.lower() in s.lower() for w in wanted)]
    if not sheets:
        st.error(f"⚠️ Nie znaleziono arkusza z planem. Nazwy arkuszy: {xls.sheet_names}")
        st.stop()
    plan = pd.concat([pd.read_excel(xls,s) for s in sheets], ignore_index=True)
    drivers = pd.read_excel(xls,"Kierowcy") if "Kierowcy" in xls.sheet_names else pd.DataFrame()
    q = st.text_input("Cel", "Ułóż trasy 06:00")
    if st.button("Start"):
        st.success(run_agent(q, plan, drivers, dest_override))
else:
    st.info("Wgraj plik Excel z planem.")
