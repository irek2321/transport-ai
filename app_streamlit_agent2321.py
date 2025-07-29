# -*- coding: utf-8 -*-
"""
Smart Transport Agent  – sam wykrywa arkusze planu i kierowców.
Funkcje:
  • Filtr zmian po godzinie i lokalizacji
  • Przydział busów wg pojemności
  • Wykrywanie konfliktów (ten sam pracownik + ta sama godzina)
  • Eksport shifts.csv
Autor: ChatGPT demo
"""
import re, tempfile, pandas as pd, streamlit as st
from typing import List, Dict, Tuple, Optional

# ---------------- Utils ----------------
def norm_time(txt: str) -> Optional[str]:
    m = re.search(r"(\\d{1,2})[:.]?(\\d{2})?", str(txt))
    if not m: return None
    h, mnt = int(m.group(1)), int(m.group(2) or 0)
    return f"{h:02d}:{mnt:02d}" if 0 <= h <= 23 else None

def split_van_tot(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "VanTot" in d.columns:
        sp = d["VanTot"].fillna("").astype(str).str.split("-", 1, expand=True)
        d["Van"], d["Tot"] = sp[0].str.strip(), sp[1].str.strip()
    if "Van" in d.columns: d["Van"] = d["Van"].map(norm_time)
    if "Tot" in d.columns: d["Tot"] = d["Tot"].map(norm_time)
    return d

def parse_cap(val) -> int:
    m = re.search(r"\\d+", str(val))
    return int(m.group(0)) if m else 0

# ---------------- Heurystyczne wykrywanie arkuszy ----------------
def detect_sheets(xls: pd.ExcelFile) -> Tuple[str, str]:
    plan_sheet = driver_sheet = None
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet, nrows=5)
        cols = [c.lower() for c in df.columns.astype(str)]
        if any("medewerker" in c for c in cols) and any("van" in c or "vantot" in c for c in cols):
            plan_sheet = sheet
        if any("kierowc" in c or "driver" in c for c in cols) and any(parse_cap(x) > 0 for x in df.iloc[0]):
            driver_sheet = sheet
    if not plan_sheet:
        plan_sheet = xls.sheet_names[0]
    if not driver_sheet:
        driver_sheet = plan_sheet  # fallback
    return plan_sheet, driver_sheet

# ---------------- Narzędzia ----------------
def get_shifts(plan: pd.DataFrame, start: str, kw: List[str]) -> pd.DataFrame:
    df = split_van_tot(plan.copy())
    canon = norm_time(start)
    if canon and "Van" in df.columns:
        df = df[df["Van"].fillna("").str.startswith(canon[:2])]
    if kw:
        mask = False
        for k in kw:
            k = k.lower()
            mask |= df.apply(lambda r: k in " ".join(map(str, r.values)).lower(), axis=1)
        df = df[mask]
    return df

def assign_buses(shifts: pd.DataFrame, drivers: pd.DataFrame) -> Dict:
    drv = drivers.copy()
    drv["cap"] = drv.apply(lambda r: parse_cap(r.get("Miejsca") or r.get("Capacity") or 8), axis=1)
    passengers = shifts["Medewerker"].dropna().tolist()
    rem, plan = len(passengers), []
    for _, r in drv.sort_values("cap", ascending=False).iterrows():
        if rem <= 0: break
        take = min(rem, int(r["cap"]))
        plan.append({"kierowca": r.get("Kierowca") or r.get("Imie") or r.get("Name"),
                     "pojemnosc": int(r["cap"]),
                     "przydzielono": take})
        rem -= take
    return {"kursy": plan, "pozostalo": rem}

def validate_conflicts(plan: pd.DataFrame) -> List[Dict]:
    df = split_van_tot(plan)
    dup = df.groupby(["Medewerker", "Van"]).size().reset_index(name="n")
    return dup[dup["n"] > 1].to_dict(orient="records")

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Smart Transport Agent", layout="wide")
st.title("🚐 Smart Transport Agent")

uploaded = st.file_uploader("⬆️ Wgraj plik Excel", type=[".xlsx", ".xlsm"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name

    xls = pd.ExcelFile(tmp_path)
    plan_sheet, driver_sheet = detect_sheets(xls)

    st.success(f"Plan ➜ **{plan_sheet}** • Kierowcy ➜ **{driver_sheet}**")

    plan_df = pd.read_excel(xls, plan_sheet)
    drivers_df = pd.read_excel(xls, driver_sheet)

    st_time = st.text_input("Start godzina (HH lub HH:MM)", "06:00")
    st_loc = st.text_input("Lokalizacja (słowa kluczowe, przecinki)", "Tholen, Moerdijk")

    if st.button("🔍 Analizuj"):
        shifts = get_shifts(plan_df, st_time, [k.strip() for k in st_loc.split(",") if k.strip()])
        st.write(f"🔎 Znaleziono **{len(shifts)}** zmian.")
        st.dataframe(shifts, use_container_width=True, hide_index=True)

        buses = assign_buses(shifts, drivers_df)
        st.subheader("🚌 Przydział busów")
        st.json(buses)

        conflicts = validate_conflicts(plan_df)
        if conflicts:
            st.error("❗ Konflikty:")
            st.json(conflicts)
        else:
            st.success("✅ Brak konfliktów.")

        csv = shifts.to_csv(index=False).encode()
        st.download_button("💾 Pobierz shifts.csv", csv, file_name="shifts.csv", mime="text/csv")
else:
    st.info("Wgraj plik, a agent zrobi resztę.")
