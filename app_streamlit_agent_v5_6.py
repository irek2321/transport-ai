
# -*- coding: utf-8 -*-
"""
Agent transportu v5.6 – Streamlit
Funkcje:
1. Filtr zmian (get_shifts)
2. Przydział busów (assign_buses)
3. Walidacja konfliktów (validate_conflicts)

Nie wymaga klucza API — działa lokalnie na danych Excela.
"""

import re
from typing import List, Tuple, Optional, Dict
import pandas as pd
import streamlit as st

APP_TITLE = "Agent transportu v5.6"

# ---------- Utils ----------
def norm_time(txt: str) -> Optional[str]:
    m = re.search(r"(\\d{1,2})[:.]?(\\d{2})?", str(txt))
    if not m: return None
    h = int(m.group(1)); mnt = int(m.group(2) or 0)
    return f"{h:02d}:{mnt:02d}" if 0 <= h <= 23 else None

def split_van_tot(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "VanTot" in d.columns:
        sp = d["VanTot"].fillna("").astype(str).str.split("-", 1, expand=True)
        d["Van"] = sp[0].str.strip()
        d["Tot"] = sp[1].str.strip()
    if "Van" in d.columns: d["Van"] = d["Van"].map(norm_time)
    if "Tot" in d.columns: d["Tot"] = d["Tot"].map(norm_time)
    return d

def parse_capacity(val):
    m = re.search(r"\\d+", str(val))
    return int(m.group(0)) if m else 0

# ---------- Tools ----------
def get_shifts(plan: pd.DataFrame, start_time: str, loc_keywords: List[str]) -> pd.DataFrame:
    df = split_van_tot(plan)
    if start_time:
        hh = norm_time(start_time)[:2]
        df = df[df["Van"].fillna("").str.startswith(hh)]
    if loc_keywords:
        mask = False
        for kw in loc_keywords:
            kw = kw.lower()
            mask |= df.apply(lambda r: kw in " ".join(map(str, r.values)).lower(), axis=1)
        df = df[mask]
    return df

def assign_buses(shifts: pd.DataFrame, drivers: pd.DataFrame) -> Dict:
    drv = drivers.copy()
    drv["cap"] = drv.apply(lambda r: parse_capacity(r.get("Miejsca") or r.get("Capacity") or 8), axis=1)
    passengers = shifts["Medewerker"].dropna().tolist()
    rem = len(passengers)
    plan = []
    for _, r in drv.sort_values("cap", ascending=False).iterrows():
        if rem <= 0: break
        take = min(rem, int(r["cap"]))
        plan.append({
            "kierowca": r.get("Kierowca") or r.get("Imie") or r.get("Name"),
            "pojemnosc": int(r["cap"]),
            "przydzielono": take
        })
        rem -= take
    return {"kursy": plan, "pozostalo": rem}

def validate_conflicts(plan: pd.DataFrame) -> List[Dict]:
    df = split_van_tot(plan)
    dup = df.groupby(["Medewerker", "Van"]).size().reset_index(name="n")
    dup = dup[dup["n"] > 1]
    return dup.to_dict(orient="records")

# ---------- UI ----------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

uploaded = st.file_uploader("Wgraj Excel z arkuszami Planing/Tholen/Moerdijk + Kierowcy", type=[".xlsx", ".xlsm"])

if uploaded:
    xls = pd.ExcelFile(uploaded)
    # wykrywanie arkuszy planu
    wanted = ("Planing", "Plan", "Tholen", "Moerdijk")
    plan_sheets = [s for s in xls.sheet_names if any(w.lower() in s.lower().strip() for w in wanted)]
    if not plan_sheets:
        plan_sheets = [xls.sheet_names[0]]  # fallback
    plan = pd.concat([pd.read_excel(xls, s) for s in plan_sheets], ignore_index=True)
    drivers = pd.read_excel(xls, "Kierowcy") if "Kierowcy" in xls.sheet_names else pd.DataFrame()

    st.success(f"Wczytano plan ({len(plan)}) i kierowców ({len(drivers)})")

    with st.expander("Podgląd planu (top 200)"):
        st.dataframe(plan.head(200), use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Filtr zmian")
        time_input = st.text_input("Start godzina (HH lub HH:MM)", "06:00")
        loc_input = st.text_input("Lokalizacja (słowa kluczowe, przecinki)", "Tholen")
        if st.button("🚐   Pokaż zmiany"):
            shifts = get_shifts(plan, time_input, [k.strip() for k in loc_input.split(",") if k.strip()])
            st.write(f"Znaleziono {len(shifts)} zmian.")
            st.dataframe(shifts, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Przydziel busy i konflikty")
        if st.button("🚌  Przydziel bus + konflikty"):
            shifts = get_shifts(plan, "06:00", ["Tholen"])
            result = assign_buses(shifts, drivers)
            st.json(result)
            conflicts = validate_conflicts(plan)
            if conflicts:
                st.error("Konflikty:")
                st.json(conflicts)
            else:
                st.success("Brak konfliktów.")
else:
    st.info("Wgraj plik, aby uruchomić narzędzia.")
