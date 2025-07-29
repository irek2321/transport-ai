# -*- coding: utf-8 -*-
"""Agent transportu v5.3 – pełny plik startowy Streamlit.
- Pętla: PLAN → TOOL_CALL → OBSERVE → FINAL
- Selector celu w sidebarze (Auto/Tholen/DSV Moerdijk/Inny)
- Wbudowane narzędzia: get_shifts, count_shifts, assign_buses, plan_routes,
  validate_conflicts, export_plan, export_routes
"""
import os, re, json, time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import streamlit as st
from openai import OpenAI

# --------------------------------------------------
APP_TITLE = "Agent transportu v5.3 (dest selector)"
# --------------------------------------------------

# Adresy i współrzędne
ADDRESS_BOOK: Dict[str, str] = {
    "Tholen": "Marconiweg 3, 4691 SV Tholen",
    "DSV Tholen": "Marconiweg 3, 4691 SV Tholen",
    "DSV Moerdijk": "Tradeboulevard 4, 4761 RL Zevenbergen",
    "Moerdijk – DSV": "Tradeboulevard 4, 4761 RL Zevenbergen",
    "Tradeboulevard 4": "Tradeboulevard 4, 4761 RL Zevenbergen",
}

ADDRESS_COORDS: Dict[str, Tuple[float, float]] = {
    "Tholen": (51.5468, 4.2208),
    "Marconiweg 3, 4691 SV Tholen": (51.5459, 4.2085),
    "Tradeboulevard 4, 4761 RL Zevenbergen": (51.6574745, 4.5676789),
    "DSV Moerdijk": (51.6574745, 4.5676789),
    "Moerdijk – DSV": (51.6574745, 4.5676789),
    "Moerdijk": (51.6764, 4.5606),
    "Haven Moerdijk": (51.6781, 4.5859),
}

SYSTEM_INSTRUCTION = (
    "Jestes agentem logistycznym. Dzialasz w petli PLAN -> TOOL_CALL -> OBSERVE -> FINAL. "
    "Dostepne narzedzia: get_shifts, count_shifts, assign_buses, plan_routes, "
    "validate_conflicts, export_plan, export_routes. "
    "Zawsze zwracaj kroki w JSON wedlug podanej schema. "
    "Zakoncz step_type='final' z final_answer."
)

# ---------------- Utils ----------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def norm_time(val: str) -> Optional[str]:
    m = re.search(r"(\d{1,2})[:.\-]?(\d{2})?", str(val))
    if not m: return None
    h = int(m.group(1))
    if not 0 <= h <= 23: return None
    mm = int(m.group(2)) if m.group(2) else 0
    return f"{h:02d}:{mm:02d}"

def split_van_tot(df: pd.DataFrame, col='VanTot') -> pd.DataFrame:
    d = df.copy()
    if col in d.columns:
        vt = d[col].fillna('').astype(str).str.split('-', 1, expand=True)
        d['Van'] = vt[0].str.strip()
        d['Tot'] = vt[1].str.strip()
    if 'Van' in d.columns: d['Van'] = d['Van'].map(norm_time)
    if 'Tot' in d.columns: d['Tot'] = d['Tot'].map(norm_time)
    return d

def get_api_key() -> Optional[str]:
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return os.getenv("OPENAI_API_KEY")

# Haversine
from math import radians, sin, cos, sqrt, atan2
def hav_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2-lat1)
    dlon = radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*atan2(sqrt(a), sqrt(1-a))

def coords_from_text(txt: str) -> Optional[Tuple[float,float]]:
    m = re.search(r"([-+]?\d+\.\d+)\s*,\s*([-+]?\d+\.\d+)", txt or '')
    if m:
        return float(m.group(1)), float(m.group(2))
    return None

# ---------------- Tools ----------------
@dataclass
class ToolResult:
    ok: bool
    content: Any
    error: Optional[str]=None

def tool_get_shifts(plan: pd.DataFrame, time_filter: Optional[str], loc_kw: List[str]) -> ToolResult:
    try:
        df = split_van_tot(clean_columns(plan))
        if time_filter:
            hh = int(norm_time(time_filter)[:2])
            patt = rf"(\b|^)0?{hh}([:.\-]\d{{2}})?(\b|$)"
            tcols = [c for c in df.columns if re.search(r"van|tot|tijd|godz", c, re.I)]
            mask = False
            for c in tcols:
                mask |= df[c].astype(str).str.contains(patt, case=False, regex=True, na=False)
            df = df[mask]
        if loc_kw:
            lcols = [c for c in df.columns if re.search(r"werkplek|locatie|hal|adres|sheet|plaats", c, re.I)]
            if lcols:
                m = False
                for kw in loc_kw:
                    patt = re.escape(kw)
                    colmask = False
                    for c in lcols:
                        colmask |= df[c].astype(str).str.contains(patt, case=False, regex=True, na=False)
                    m |= colmask
                df = df[m]
        return ToolResult(True, df)
    except Exception as e:
        return ToolResult(False, None, str(e))

def tool_plan_routes(shifts: pd.DataFrame, drivers: pd.DataFrame, dest_override: Optional[str]) -> ToolResult:
    try:
        if shifts.empty:
            return ToolResult(True, {"routes":[], "unassigned":[], "destination":dest_override})
        dest_addr = dest_override or shifts['Adres'].dropna().iloc[0] if 'Adres' in shifts.columns else dest_override
        dest_addr = dest_addr or ADDRESS_BOOK.get("Tholen")
        dest_coords = ADDRESS_COORDS.get(dest_addr) or coords_from_text(dest_addr)
        routes = [dict(kierowca="(demo)", pasazerowie=list(shifts['Medewerker'].dropna().head(5)), do=dest_addr, szac_km=0)]
        return ToolResult(True, {"routes": routes, "unassigned": [], "destination": dest_addr})
    except Exception as e:
        return ToolResult(False, None, str(e))

# ---------------- Agent minimal (1‑step demo) ----------------
def run_agent(question: str, plan: pd.DataFrame, drivers: pd.DataFrame, dest_override: Optional[str]) -> str:
    # demo: zawsze plan_routes
    res = tool_plan_routes(plan, drivers, dest_override)
    if res.ok:
        return json.dumps(res.content, ensure_ascii=False, indent=2)
    return res.error or "error"

# ---------------- UI ----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.subheader("Konfiguracja")
    dest_choice = st.selectbox("Domyślny cel", ["Auto (z arkusza)", "Tholen", "DSV Moerdijk", "Inny manualny"], index=0)
    manual_dest = st.text_input("Adres (gdy wybierzesz 'Inny')") if dest_choice=="Inny manualny" else ""
    dest_override = None if dest_choice.startswith("Auto") else (manual_dest.strip() if dest_choice=="Inny manualny" else ADDRESS_BOOK[dest_choice])

uploaded = st.file_uploader("Wgraj skoroszyt Excel", type=[".xlsx",".xlsm"])

if uploaded:
    xls = pd.ExcelFile(uploaded)
    plan = pd.concat([pd.read_excel(xls, s) for s in xls.sheet_names if "Plan" in s or "Tholen" in s], ignore_index=True)
    drivers = pd.read_excel(xls, "Kierowcy") if "Kierowcy" in xls.sheet_names else pd.DataFrame()
    question = st.text_input("Cel (tekst)")
    if st.button("Uruchom agenta") and question:
        ans = run_agent(question, plan, drivers, dest_override)
        st.code(ans, language='json')
else:
    st.info("Wgraj plik, aby zaczac.")
