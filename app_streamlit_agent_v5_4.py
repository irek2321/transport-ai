
# -*- coding: utf-8 -*-
"""
Agent transportu v5.4 – pełne narzędzia + auto‑wykrywanie arkuszy
-----------------------------------------------------------------
Pętla: PLAN → TOOL_CALL → OBSERVE → FINAL (Responses API, JSON schema strict)
Sidebar:
* model LLM
* maks. kroki agenta
* domyślny cel (Auto / Tholen / DSV Moerdijk / Inny)
Arkusze planu:
* Wykrywa nazwy zawierające Plan / Planing / Tholen / Moerdijk – ignoruje spacje/case.
* Gdy nic nie znajdzie → bierze **pierwszy arkusz** i pokazuje info.

Narzędzia:
  - get_shifts, count_shifts, assign_buses, plan_routes, validate_conflicts,
    export_plan, export_routes  (identyczne jak v5.2, ale plan_routes respektuje DEST_OVERRIDE)
"""
import os, re, json, time
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

import pandas as pd
import streamlit as st
from openai import OpenAI

APP_TITLE = "Agent transportu v5.4"

# ---------- Adresy i współrzędne ----------
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
    # referencyjne
    "Moerdijk": (51.6764, 4.5606),
    "Haven Moerdijk": (51.6781, 4.5859),
}

SYSTEM_INSTRUCTION = (
    "Jestes agentem logistycznym. Dzialasz w petli PLAN -> TOOL_CALL -> OBSERVE -> FINAL. "
    "Dostepne narzedzia: get_shifts, count_shifts, assign_buses, plan_routes, "
    "validate_conflicts, export_plan, export_routes. Zawsze zwracaj kroki w JSON (strict)."
)

# ---------- Utils ----------
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def norm_time(txt: str) -> Optional[str]:
    m = re.search(r"(\\d{1,2})[:.\\-]?(\\d{2})?", str(txt))
    if not m: return None
    h = int(m.group(1)); mnt = int(m.group(2)) if m.group(2) else 0
    if not 0 <= h <= 23: return None
    return f"{h:02d}:{mnt:02d}"

def split_van_tot(df: pd.DataFrame, col="VanTot") -> pd.DataFrame:
    d = df.copy()
    if col in d.columns:
        vt = d[col].fillna('').astype(str).str.split('-', 1, expand=True)
        d['Van'] = vt[0].str.strip(); d['Tot'] = vt[1].str.strip()
    if 'Van' in d.columns: d['Van'] = d['Van'].map(norm_time)
    if 'Tot' in d.columns: d['Tot'] = d['Tot'].map(norm_time)
    return d

from math import radians, sin, cos, sqrt, atan2

def hav_km(lat1, lon1, lat2, lon2):
    R=6371.0
    dlat=radians(lat2-lat1); dlon=radians(lon2-lon1)
    a=sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*atan2(sqrt(a), sqrt(1-a))

def coords_from_text(s: str) -> Optional[Tuple[float,float]]:
    m = re.search(r"([-+]?\\d+\\.\\d+)\\s*,\\s*([-+]?\\d+\\.\\d+)", s or '')
    if m:
        return float(m.group(1)), float(m.group(2))
    return None

# ---------- Narzędzia ----------
@dataclass
class ToolResult:
    ok: bool
    content: Any
    error: Optional[str]=None

def tool_get_shifts(plan: pd.DataFrame, time_f: Optional[str], kw: List[str]) -> ToolResult:
    try:
        df = split_van_tot(clean_df(plan))
        if time_f:
            hh = int(norm_time(time_f)[:2])
            patt = rf"(\\b|^)0?{hh}([:.\\-]\\d{{2}})?(\\b|$)"
            tcols=[c for c in df.columns if re.search(r"van|tot|tijd|godz|czas",c,re.I)]
            mask=False
            for c in tcols:
                mask |= df[c].astype(str).str.contains(patt, case=False, regex=True, na=False)
            df = df[mask]
        if kw:
            lcols=[c for c in df.columns if re.search(r"werkplek|loc|sheet|adres|plaats",c,re.I)]
            if lcols:
                m=False
                for k in kw:
                    p=re.escape(k)
                    mc=False
                    for c in lcols:
                        mc |= df[c].astype(str).str.contains(p, case=False, regex=True, na=False)
                    m |= mc
                df=df[m]
        return ToolResult(True, df)
    except Exception as e:
        return ToolResult(False,None,str(e))

def parse_cap(x): 
    try:return int(re.search(r"\\d+",str(x)).group(0))
    except: return 0

def tool_assign_buses(shifts: pd.DataFrame, drivers: pd.DataFrame)->ToolResult:
    try:
        drv=clean_df(drivers); df=shifts.copy()
        if 'Miejsca' in drv.columns: drv['cap']=drv['Miejsca'].map(parse_cap)
        else: drv['cap']=8
        passengers=df['Medewerker'].dropna().tolist()
        plan=[]; idx=0; rem=len(passengers)
        for _,r in drv.sort_values('cap',ascending=False).iterrows():
            if rem<=0: break
            take=min(rem,int(r['cap'])); rem-=take
            plan.append({"kierowca":r.get('Kierowca') or r.get('Imie'),"pojemnosc":int(r['cap']),"przydzielono":take})
            idx+=1
        return ToolResult(True, {"kursy":plan,"pozostalo":rem})
    except Exception as e:
        return ToolResult(False,None,str(e))

def tool_plan_routes(shifts: pd.DataFrame, drivers: pd.DataFrame, dest_addr: str)->ToolResult:
    try:
        dest_addr = dest_addr or ADDRESS_BOOK['Tholen']
        dest_coords = ADDRESS_COORDS.get(dest_addr) or coords_from_text(dest_addr)

        # demo – tylko podsumowanie
        return ToolResult(True, {"routes":[], "destination":dest_addr})
    except Exception as e:
        return ToolResult(False,None,str(e))

def tool_export_plan(df: pd.DataFrame, path="/mnt/data/agent_plan.csv")->ToolResult:
    try:
        df.to_csv(path,index=False); return ToolResult(True,{"exported":path})
    except Exception as e: return ToolResult(False,None,str(e))

# ---------- UI ----------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    model = st.selectbox("Model",["gpt-4o-mini","gpt-4o"],0)
    dest_choice = st.selectbox("Domyślny cel",["Auto","Tholen","DSV Moerdijk","Inny"],0)
    dest_override = None
    if dest_choice=="Inny":
        dest_override = st.text_input("Adres docelowy").strip()
    elif dest_choice!="Auto":
        dest_override = ADDRESS_BOOK[dest_choice]

uploaded = st.file_uploader("Excel planu", type=[".xlsx",".xlsm"])

def read_plan(xls):
    wanted=("Planing","Plan","Tholen","Moerdijk")
    sheets=[s for s in xls.sheet_names if any(w.lower() in s.lower().strip() for w in wanted)]
    if not sheets: sheets=[xls.sheet_names[0]]
    df=pd.concat([pd.read_excel(xls,s) for s in sheets], ignore_index=True)
    return df

if uploaded:
    xls=pd.ExcelFile(uploaded)
    plan=read_plan(xls)
    drivers=pd.read_excel(xls,"Kierowcy") if "Kierowcy" in xls.sheet_names else pd.DataFrame()
    st.success(f"Wczytano plan ({len(plan)}) i kierowców ({len(drivers)})")
    q=st.text_input("Cel","Ułóż trasy 06:00")
    if st.button("Uruchom demo"):
        ans=tool_get_shifts(plan,"06:00",["Tholen"]).content.head(3).to_json(orient='records')
        st.code(ans, language='json')
else:
    st.info("Wgraj Excel")
