
import os, re, json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import streamlit as st
from openai import OpenAI

APP_TITLE = "Agent transportu v5 (plan-act-observe)"

SYSTEM_INSTRUCTION = (
    "Jestes agentem logistycznym do planowania transportu i obslugi grafikow. "
    "Pracujesz w petli PLAN -> ACT (narzedzie) -> OBSERVE (wynik/blad) az do FINAL. "
    "Zawsze dbasz o przejrzysty wynik i minimalna liczbe krokow. "
    "Priorytet: godziny (Van/Tot) i miejsce. Wymagaj adresu dla Tholen: Marconiweg 3, 4691 SV Tholen. "
    "Zanim wywolasz narzedzie, krociutko zaplanuj. Po bledzie popraw parametry i sprobuj ponownie."
)

ADDRESS_BOOK: Dict[str, str] = {
    "Tholen": "Marconiweg 3, 4691 SV Tholen",
    "DSV Tholen": "Marconiweg 3, 4691 SV Tholen",
}

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def norm_time(s: str) -> Optional[str]:
    if not isinstance(s, str):
        s = str(s)
    m = re.search(r"(\\d{1,2})[:.\\-]?(\\d{2})?", s)
    if not m:
        return None
    h = int(m.group(1))
    if h < 0 or h > 23:
        return None
    mm = int(m.group(2)) if m.group(2) else 0
    return f"{h:02d}:{mm:02d}"

def split_van_tot(df: pd.DataFrame, col="VanTot") -> pd.DataFrame:
    d = df.copy()
    if col in d.columns:
        vt = d[col].fillna("").astype(str).str.split("-", n=1, expand=True)
        d["Van"] = vt[0].str.strip()
        d["Tot"] = vt[1].str.strip()
    if "Van" in d.columns:
        d["Van"] = d["Van"].map(norm_time)
    if "Tot" in d.columns:
        d["Tot"] = d["Tot"].map(norm_time)
    return d

def ensure_address(row: pd.Series) -> str:
    for c in row.index:
        if re.search(r"werkplek|locatie|hal|facility|miast|adres|sheet|plaats|stad|to|from", c, re.I):
            val = str(row[c])
            for key, adr in ADDRESS_BOOK.items():
                if re.search(key, val, re.I):
                    return adr
    return ""

@dataclass
class ToolResult:
    ok: bool
    content: Any
    error: Optional[str] = None

def tool_get_shifts(plan: pd.DataFrame, time_filter: Optional[str], location_keywords: List[str], limit: int = 200) -> ToolResult:
    try:
        p = clean_columns(plan)
        p = split_van_tot(p)
        df = p
        if time_filter:
            t = norm_time(time_filter)
            if t:
                time_cols = [c for c in df.columns if re.search(r"van|tot|czas|godz|begin|einde|tijd", c, re.I)]
                mask_any = False
                for col in time_cols:
                    mask_any = mask_any | df[col].astype(str).str.contains(re.escape(t[:2]), regex=False, case=False, na=False)
                df = df[mask_any]
        if location_keywords:
            loc_cols = [c for c in p.columns if re.search(r"werkplek|locatie|hal|facility|miast|adres|sheet|plaats|stad", c, re.I)]
            if loc_cols:
                m = False
                for kw in location_keywords:
                    patt = re.escape(str(kw))
                    mcol = False
                    for col in loc_cols:
                        mcol = mcol | p[col].astype(str).str.contains(patt, case=False, regex=True, na=False)
                    m = m | mcol
                df = df[m]
        if "Adres" not in df.columns:
            df["Adres"] = df.apply(ensure_address, axis=1)
            df.loc[df["Adres"] == "", "Adres"] = df.apply(lambda r: ADDRESS_BOOK.get("Tholen", "") if re.search("tholen", " ".join(map(str, r.values)), re.I) else "", axis=1)
        if len(df) > limit:
            df = df.head(limit)
        return ToolResult(True, df)
    except Exception as e:
        return ToolResult(False, None, f"tool_get_shifts error: {e}")

def tool_count_shifts(df: pd.DataFrame) -> ToolResult:
    try:
        if df.empty:
            return ToolResult(True, {"count": 0})
        df = split_van_tot(df)
        out = df.groupby(["Van"]).size().to_dict()
        return ToolResult(True, {"by_start_time": out, "count": int(len(df))})
    except Exception as e:
        return ToolResult(False, None, f"tool_count_shifts error: {e}")

def parse_capacity(s: Any) -> int:
    try:
        if pd.isna(s): return 0
        m = re.search(r"\\d+", str(s))
        return int(m.group(0)) if m else 0
    except Exception:
        return 0

def tool_assign_buses(shifts: pd.DataFrame, drivers: pd.DataFrame, time: Optional[str], location_keywords: List[str]) -> ToolResult:
    try:
        df = shifts.copy()
        drv = clean_columns(drivers).copy()
        if "Miejsca" in drv.columns:
            drv["Capacity"] = drv["Miejsca"].map(parse_capacity)
        else:
            cap_col = None
            for c in drv.columns:
                if drv[c].astype(str).str.contains(r"\\d+", regex=True, na=False).any():
                    cap_col = c; break
            drv["Capacity"] = drv[cap_col].map(parse_capacity) if cap_col else 8

        if time:
            tt = norm_time(time)
            if tt:
                df = df[df["Van"].astype(str).str.startswith(tt[:2], na=False)]
        if location_keywords:
            loc_cols = [c for c in df.columns if re.search(r"werkplek|locatie|hal|facility|miast|adres|sheet|plaats|stad", c, re.I)]
            if loc_cols:
                mask = False
                for kw in location_keywords:
                    patt = re.escape(kw)
                    mcol = False
                    for c in loc_cols:
                        mcol = mcol | df[c].astype(str).str.contains(patt, case=False, regex=True, na=False)
                    mask = mask | mcol
                df = df[mask]

        passengers = df.shape[0]
        plan = []
        remaining = passengers
        drv_sorted = drv.sort_values(by="Capacity", ascending=False).to_dict(orient="records")
        i = 0
        for d in drv_sorted:
            if remaining <= 0: break
            take = min(remaining, int(d.get("Capacity", 0) or 0))
            if take <= 0: continue
            plan.append({
                "kurs": i+1,
                "kierowca": d.get("Kierowca") or d.get("Imie") or d.get("Name"),
                "telefon": d.get("Telefon") or "",
                "pojazd": d.get("Pojazd") or "",
                "pojemnosc": int(d.get("Capacity", 0) or 0),
                "przydzielono_osob": int(take),
                "godzina": time or (df["Van"].iloc[0] if not df.empty else None),
                "miejsce": ", ".join(location_keywords) if location_keywords else (df["Sheet"].iloc[0] if not df.empty else None),
            })
            i += 1
            remaining -= take

        result = {"pasazerowie": int(passengers), "pozostalo": int(max(0, remaining)), "kursy": plan}
        return ToolResult(True, result)
    except Exception as e:
        return ToolResult(False, None, f"tool_assign_buses error: {e}")

def tool_validate_conflicts(df: pd.DataFrame) -> ToolResult:
    try:
        if df.empty:
            return ToolResult(True, {"conflicts": []})
        d = split_van_tot(df)
        if "Medewerker" not in d.columns:
            return ToolResult(True, {"conflicts": []})
        conflicts = []
        dup = d.groupby(["Medewerker","Van"]).size().reset_index(name="n")
        for _, r in dup[dup["n"]>1].iterrows():
            conflicts.append({"name": r["Medewerker"], "Van": r["Van"], "count": int(r["n"])})
        return ToolResult(True, {"conflicts": conflicts})
    except Exception as e:
        return ToolResult(False, None, f"tool_validate_conflicts error: {e}")

AGENT_STEP_SCHEMA = {
    "name": "agent_step",
    "schema": {
        "type": "object",
        "properties": {
            "step_type": {"type": "string", "enum": ["plan", "tool_call", "final"]},
            "thought": {"type": "string"},
            "tool_name": {"type": "string", "nullable": True},
            "args": {"type": "object"},
            "expected": {"type": "string", "nullable": True},
            "final_answer": {"type": "string", "nullable": True}
        },
        "required": ["step_type", "thought", "args"],
        "additionalProperties": False,
    },
    "strict": True,
}

def get_api_key() -> Optional[str]:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")

def call_llm_step(client: OpenAI, model: str, history: List[Dict[str, Any]], question: str, last_tool_out: Optional[str]) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = []
    messages.append({"role": "system", "content": SYSTEM_INSTRUCTION})
    for h in history:
        messages.append(h)
    user_payload = [
        {"type": "input_text", "text": f"Pytanie:\n{question}"},
        {"type": "input_text", "text": "Uzywaj narzedzi gdy potrzebne. Zakoncz step_type='final' z final_answer."}
    ]
    if last_tool_out:
        user_payload.append({"type":"input_text","text": f"Ostatni wynik narzedzia / blad:\n{last_tool_out}"})
    messages.append({"role":"user","content": user_payload})

    resp = client.responses.create(
        model=model,
        input=messages,
        response_format={"type":"json_schema", "json_schema": AGENT_STEP_SCHEMA},
    )
    data = json.loads(resp.output_text)
    return data

def run_agent(question: str, plan_df: pd.DataFrame, workers_df: pd.DataFrame, drivers_df: pd.DataFrame,
              model: str, max_steps: int = 6) -> str:
    api_key = get_api_key()
    if not api_key:
        return "Brak klucza API. Dodaj OPENAI_API_KEY w Secrets."

    client = OpenAI(api_key=api_key)

    history: List[Dict[str, Any]] = []
    last_tool_out: Optional[str] = None

    context_summary = f"""Kontekst (skrocony):
- plan_rows: {len(plan_df)}, workers_rows: {len(workers_df)}, drivers_rows: {len(drivers_df)}
- adresy: {json.dumps(ADDRESS_BOOK, ensure_ascii=False)}
"""
    history.append({"role":"user","content":[{"type":"input_text","text":context_summary}]})

    current_shifts = plan_df

    for step_i in range(1, max_steps+1):
        step = call_llm_step(client, model, history, question, last_tool_out)
        history.append({"role":"assistant","content":[{"type":"output_text","text":json.dumps(step, ensure_ascii=False)}]})

        if step.get("step_type") == "final":
            final = step.get("final_answer") or "(brak final_answer)"
            return final

        if step.get("step_type") == "plan":
            last_tool_out = f"Plan przyjeto: {step.get('thought','')}"
            continue

        if step.get("step_type") == "tool_call":
            tool_name = (step.get("tool_name") or "").strip()
            args = step.get("args") or {}
            try:
                if tool_name == "get_shifts":
                    t = args.get("time")
                    locs = args.get("locations") or []
                    res = tool_get_shifts(plan_df, t, locs)
                    if res.ok:
                        current_shifts = res.content
                        last_tool_out = f"get_shifts OK, rows={len(current_shifts)}"
                    else:
                        last_tool_out = res.error or "get_shifts error"
                elif tool_name == "count_shifts":
                    res = tool_count_shifts(current_shifts)
                    last_tool_out = json.dumps(res.content, ensure_ascii=False) if res.ok else (res.error or "count_shifts error")
                elif tool_name == "assign_buses":
                    t = args.get("time")
                    locs = args.get("locations") or []
                    res = tool_assign_buses(current_shifts, drivers_df, t, locs)
                    last_tool_out = json.dumps(res.content, ensure_ascii=False) if res.ok else (res.error or "assign_buses error")
                elif tool_name == "validate_conflicts":
                    res = tool_validate_conflicts(current_shifts)
                    last_tool_out = json.dumps(res.content, ensure_ascii=False) if res.ok else (res.error or "validate_conflicts error")
                elif tool_name == "export_plan":
                    path = "/mnt/data/agent_plan.csv"
                    current_shifts.to_csv(path, index=False)
                    last_tool_out = f"exported:{path}"
                else:
                    last_tool_out = f"Nieznane narzedzie: {tool_name}"
            except Exception as e:
                last_tool_out = f"BLAD przy narzedziu {tool_name}: {e}"

            history.append({"role":"user","content":[{"type":"input_text","text": f"OBSERVE:\n{last_tool_out}"}]})
            continue

    return f"Max steps ({max_steps}) reached. Ostatni stan:\n{last_tool_out or '(brak)'}"

# ---------------- UI ----------------

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

api_key = None
try:
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = os.getenv("OPENAI_API_KEY")

with st.sidebar:
    st.header("Ustawienia")
    model = st.selectbox("Model", ["gpt-4o-mini","gpt-4.1","gpt-4.1-mini","gpt-4o"], index=0)
    max_steps = st.slider("Max krokow agenta", 2, 12, 6)
    st.caption(f"API key: {'OK' if api_key else 'MISSING'}")
    st.divider()
    st.caption("Adresy stale:")
    for k,v in ADDRESS_BOOK.items():
        st.caption(f"- {k}: {v}")

uploaded = st.file_uploader("Wgraj skoroszyt Excel (Tholen, Moerdijk, Pracownicy, Kierowcy)", type=[".xlsx",".xlsm"])

@st.cache_data(show_spinner=False)
def load_all(file) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    xls = pd.ExcelFile(file)
    plan_parts = []
    for sh in xls.sheet_names:
        if sh.strip() in ("Planing","Tholen","Moerdijk"):
            raw = pd.read_excel(xls, sh)
            hdr_idx = raw.index[raw.iloc[:,0].astype(str).str.contains("Dienst", na=False)].tolist()
            start = (hdr_idx[0]+1) if hdr_idx else 5
            data = raw.iloc[start:].copy()
            mapping = {
                "Dienst": "Unnamed: 0",
                "Werkplek": "Unnamed: 2",
                "Activiteit": "Unnamed: 4",
                "VanTot": "Unnamed: 5",
                "Medewerker": "Unnamed: 6",
                "Werkgever": "Unnamed: 9",
                "Opmerking": "Unnamed: 10",
            }
            part = pd.DataFrame({k: data.get(v) for k,v in mapping.items()}).dropna(how="all")
            part["Sheet"] = sh.strip()
            part = split_van_tot(part)
            if re.search(r"tholen", sh, re.I):
                part["Adres"] = ADDRESS_BOOK["Tholen"]
            plan_parts.append(part)
    plan = pd.concat(plan_parts, ignore_index=True) if plan_parts else pd.DataFrame()

    workers = pd.DataFrame()
    for nm in ["Pracownicy","Pula","Workers","Employees","Pula pracowników"]:
        if nm in xls.sheet_names:
            workers = pd.read_excel(xls, nm); break

    if "Kierowcy" in xls.sheet_names:
        drivers = pd.read_excel(xls, "Kierowcy")
    else:
        drivers = pd.DataFrame()

    return plan, workers, drivers

if uploaded:
    try:
        plan_df, workers_df, drivers_df = load_all(uploaded)
        st.success(f"Wczytano: plan={len(plan_df)} wierszy, pracownicy={len(workers_df)}, kierowcy={len(drivers_df)}")
    except Exception as e:
        st.error(f"Blad wczytywania: {e}")
        st.stop()

    with st.expander("Podglad planu (pierwsze 400)"):
        st.dataframe(plan_df.head(400), use_container_width=True, hide_index=True)
    with st.expander("Pracownicy"):
        if workers_df.empty: st.info("Brak")
        else: st.dataframe(workers_df, use_container_width=True, hide_index=True)
    with st.expander("Kierowcy"):
        if drivers_df.empty: st.warning("Brak – agent nie ulozy kursow.")
        else: st.dataframe(drivers_df, use_container_width=True, hide_index=True)

    q = st.text_input("Pytanie / cel (np. Uloz kursy na 06:00 w Tholen i wykryj konflikty)")
    if st.button("Uruchom agenta") and q:
        with st.spinner("Agent pracuje..."):
            answer = run_agent(q, plan_df, workers_df, drivers_df, model=model, max_steps=max_steps)
        st.markdown("### Wynik agenta")
        st.write(answer)
        if "exported:" in answer:
            path = answer.split("exported:",1)[1].strip()
            st.success(f"Plan zapisany: {path}")
            st.markdown(f"[Pobierz plik]({path})")
else:
    st.info("Wgraj plik, aby uruchomic agenta.")
