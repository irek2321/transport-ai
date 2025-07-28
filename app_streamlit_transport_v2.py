import re
import os
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI

APP_TITLE = "🚐 Agent AI do transportu pracowników"
SYSTEM_INSTRUCTION = (
    "Jesteś asystentem logistycznym. Odpowiadaj zwięźle, po polsku. "
    "Priorytet: godziny (Van/Tot) i miejsce. Gdy brakuje danych, napisz wprost czego potrzebujesz."
)

ADDRESS_BOOK: Dict[str, str] = {
    "Tholen": "Marconiweg 3, 4691 SV Tholen",
    "DSV Tholen": "Marconiweg 3, 4691 SV Tholen",
}

def get_api_key() -> Optional[str]:
    """Zwraca klucz API. Jeśli jest w st.secrets – ustawia też os.environ
    tak, by klient OpenAI miał spójne źródło. Potem fallback do zmiennej środowiskowej.
    """
    key: Optional[str] = None
    try:
        # Streamlit Secrets – preferowane w chmurze
        if "OPENAI_API_KEY" in st.secrets:
            key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        key = None

    # Fallback do zmiennej środowiskowej
    if not key:
        key = os.getenv("OPENAI_API_KEY")

    # Ustaw spójnie w środowisku procesu (dla bibliotek odczytujących z env)
    if key and os.environ.get("OPENAI_API_KEY") != key:
        os.environ["OPENAI_API_KEY"] = key

    return key

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def parse_time_candidates(text: str) -> List[str]:
    patt = r"\b(\d{1,2})[:\.-]?(\d{2})?\b"
    times: List[str] = []
    for h, m in re.findall(patt, text):
        hour = int(h)
        if 0 <= hour <= 23:
            minute = int(m) if m else 0
            times.append(f"{hour:02d}:{minute:02d}")
    return list(dict.fromkeys(times))

def parse_locations(text: str) -> List[str]:
    words = re.findall(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż\-]+", text)
    locs = [w for w in words if w[:1].isupper() or "-" in w]
    stop = {"Na", "Do", "Z", "I", "Oraz", "Od", "Kto", "Który", "Kiedy"}
    locs = [w for w in locs if w.capitalize() not in stop]
    return list(dict.fromkeys(locs))

def names_from_plan(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return []
    cols = [c for c in df.columns if re.search(r"medewerk|pracown|employee|worker|osob|nazw|imi", c, re.I)]
    names: List[str] = []
    for c in cols:
        names += [str(x).strip() for x in df[c].dropna().astype(str).tolist() if str(x).strip()]
    names = [re.sub(r"\s+", " ", n) for n in names]
    return list(dict.fromkeys(names))

def enrich_addresses(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if d.empty:
        return d
    loc_cols = [c for c in d.columns if re.search(
        r"miast|lok|skad|dokad|trasa|adres|miejsc|punkt|from|to|werkplek|locatie|hal|plaats|stad|facility|sheet",
        c, re.I
    )]
    if not loc_cols:
        return d
    addr_col = "Adres" if "Adres" not in d.columns else "Adres_norm"
    d[addr_col] = d.get(addr_col, "")
    def row_addr(row: pd.Series) -> str:
        for c in loc_cols:
            val = str(row.get(c, ""))
            if re.search(r"tholen", val, re.I):
                for key, adr in ADDRESS_BOOK.items():
                    if re.search(re.escape(key), val, re.I):
                        return adr
                return ADDRESS_BOOK["Tholen"]
        return ""
    d[addr_col] = [row_addr(r) for _, r in d.iterrows()]
    return d

@st.cache_data(show_spinner=False)
def load_data(file) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    xls = pd.ExcelFile(file)
    planning_sheets = []
    for cand in ["Planing", "Tholen", "Moerdijk", "Moerdijk "]:
        if cand in xls.sheet_names:
            planning_sheets.append(cand)
    if not planning_sheets:
        raise ValueError("Brak arkusza planingu (oczekiwano: Planing / Tholen / Moerdijk).")

    plan_parts = []
    for sh in planning_sheets:
        raw = pd.read_excel(xls, sh)
        hdr_idx = raw.index[raw.iloc[:, 0].astype(str).str.contains("Dienst", na=False)].tolist()
        start = (hdr_idx[0] + 1) if hdr_idx else 5
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
        part = pd.DataFrame({k: data.get(v) for k, v in mapping.items()})
        part = part.dropna(how="all")

        vt = part["VanTot"].fillna("").astype(str).str.split("-", n=1, expand=True)
        part["Van"] = vt[0].str.strip()
        part["Tot"] = vt[1].str.strip()
        part["Sheet"] = sh.strip()

        if "tholen" in sh.lower():
            part["Adres"] = ADDRESS_BOOK["Tholen"]

        plan_parts.append(part)

    plan = pd.concat(plan_parts, ignore_index=True)

    workers = pd.DataFrame()
    for nm in ["Pracownicy", "Pula", "Workers", "Employees", "Pula pracowników"]:
        if nm in xls.sheet_names:
            workers = pd.read_excel(xls, nm)
            break

    if "Kierowcy" not in xls.sheet_names:
        raise ValueError("Brak arkusza 'Kierowcy'.")
    drivers = pd.read_excel(xls, "Kierowcy")

    return plan, workers, drivers

def filter_context(
    planing: pd.DataFrame,
    workers: pd.DataFrame,
    drivers: pd.DataFrame,
    pytanie: str,
    max_rows: int = 120,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = clean_columns(planing)
    w = clean_columns(workers)
    k = clean_columns(drivers)

    times = parse_time_candidates(pytanie)
    locs = parse_locations(pytanie)

    pf = p

    time_cols = [c for c in pf.columns if re.search(r"czas|godz|start|van|tot|begin|einde|tijd|vantot|van|tot", c, re.I)]
    if times and time_cols:
        mask_any = False
        for t in times:
            h, m = t.split(":")
            pattern = rf"\b0?{int(h)}[:\.-]?{int(m):02d}\b|\b0?{int(h)}\b"
            mcol = False
            for col in time_cols:
                mcol = mcol | pf[col].astype(str).str.contains(pattern, case=False, regex=True, na=False)
            mask_any = mask_any | mcol
        pf = pf[mask_any]

    if locs:
        loc_cols = [c for c in p.columns if re.search(
            r"miast|lok|skad|dokad|trasa|adres|miejsc|punkt|from|to|werkplek|locatie|hal|plaats|stad|facility|sheet",
            c, re.I)]
        if loc_cols:
            mask_loc = False
            for loc in locs:
                patt = rf"\b{re.escape(loc)}\b"
                mcol = False
                for col in loc_cols:
                    mcol = mcol | p[col].astype(str).str.contains(patt, case=False, regex=True, na=False)
                mask_loc = mask_loc | mcol
            pf = pf[mask_loc]

    pf = enrich_addresses(pf)

    if len(pf) > max_rows:
        pf = pf.head(max_rows)

    wanted_names = names_from_plan(pf)
    wf = w
    if wanted_names and not w.empty:
        name_cols = [c for c in w.columns if re.search(r"imie|imię|nazw|name|pracown|employee|worker", c, re.I)]
        if name_cols:
            mask_any = False
            for nm in wanted_names:
                patt = rf"\b{re.escape(nm)}\b"
                mcol = False
                for c in name_cols:
                    mcol = mcol | w[c].astype(str).str.contains(patt, case=False, regex=True, na=False)
                mask_any = mask_any | mcol
            wf = w[mask_any]

    return pf, wf, k

def df_to_compact_text(df: pd.DataFrame, max_chars: int = 20_000) -> str:
    if df.empty:
        return "(pusta tabela)"
    csv = df.to_csv(index=False)
    if len(csv) <= max_chars:
        return csv
    head = csv[:max_chars]
    return head + f"\n... (przycięto, łączna długość={len(csv)} znaków)"

def build_messages(plan_csv: str, workers_csv: str, drivers_csv: str, pytanie: str) -> List[Dict[str, Any]]:
    addresses_text = "\n".join([f"{k}: {v}" for k, v in ADDRESS_BOOK.items()]) or "(brak)"
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": (
                    "Masz odpowiedzieć na pytanie dotyczące planu transportu. "
                    "Skup się na godzinach (Van/Tot) i miejscu pracy. "
                    "Jeśli podajesz listy osób/kierowców, dołącz: godzina, skąd→dokąd (jeśli dotyczy), "
                    "lokalizacja/hal, pojazd/bus, telefon jeżeli jest."
                )},
                {"type": "input_text", "text": f"Pytanie:\n{pytanie}"},
                {"type": "input_text", "text": f"Planing CSV:\n{plan_csv}"},
                {"type": "input_text", "text": f"Pula pracowników CSV:\n{workers_csv}"},
                {"type": "input_text", "text": f"Kierowcy CSV:\n{drivers_csv}"},
                {"type": "input_text", "text": f"Stałe adresy:\n{addresses_text}"},
            ],
        },
    ]

def stream_openai_text(client: OpenAI, model: str, messages: List[Dict[str, Any]]):
    stream = client.responses.create(model=model, input=messages, stream=True)
    for event in stream:
        etype = getattr(event, "type", "")
        if etype == "response.output_text.delta":
            yield event.delta
        elif etype == "response.error":
            raise RuntimeError(getattr(event, "error", "Błąd API"))

def ask_ai(client: OpenAI, model: str, messages: List[Dict[str, Any]]) -> str:
    resp = client.responses.create(model=model, input=messages)
    return resp.output_text

# ---------------- UI ----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

api_key = get_api_key()
if not api_key:
    st.warning("Ustaw OPENAI_API_KEY w sekcjach Secrets albo w zmiennych środowiskowych.")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) if api_key else None

uploaded_file = st.file_uploader(
    "Wgraj plik Excel: Tholen + Moerdijk + Pracownicy + Kierowcy", type=[".xlsx", ".xlsm"]
)

if uploaded_file:
    try:
        planing, pracownicy, kierowcy = load_data(uploaded_file)
        st.success("Plik załadowany poprawnie.")
    except Exception as e:
        st.error(f"Błąd podczas wczytywania: {e}")
        st.stop()

    with st.expander("Podgląd danych – Planing (po scaleniu)"):
        st.dataframe(planing, use_container_width=True, hide_index=True)
    with st.expander("Podgląd danych – Pula pracowników"):
        if pracownicy.empty:
            st.info("Nie znaleziono arkusza z pulą pracowników (opcjonalny).")
        else:
            st.dataframe(pracownicy, use_container_width=True, hide_index=True)
    with st.expander("Podgląd danych – Kierowcy"):
        st.dataframe(kierowcy, use_container_width=True, hide_index=True)

    with st.sidebar:
        st.header("⚙️ Ustawienia AI")
        model = st.selectbox(
            "Model",
            options=["gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4o"],
            index=0,
            help=(
                "4o-mini – najtańszy i szybki; 4.1 – najmocniejszy (duży kontekst); "
                "4.1-mini – kompromis; 4o – multimodalny."
            ),
        )
        stream_answer = st.toggle("Strumieniuj odpowiedź", value=True)
        max_rows = st.slider("Maks. wierszy w kontekście", 40, 500, 120, 20)

    if "history" not in st.session_state:
        st.session_state.history = []

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Zadaj pytanie (np. Kto zaczyna o 06:00 w Tholen?)")
    if user_q:
        st.session_state.history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        pf, wf, kf = filter_context(planing, pracownicy, kierowcy, user_q, max_rows=max_rows)
        plan_csv = df_to_compact_text(pf)
        workers_csv = df_to_compact_text(wf)
        drivers_csv = df_to_compact_text(kf)
        messages = build_messages(plan_csv, workers_csv, drivers_csv, user_q)

        with st.chat_message("assistant"):
            if not client:
                st.error("Brak klucza API.")
                text = ""
            else:
                if stream_answer:
                    try:
                        text = st.write_stream(stream_openai_text(client, model, messages))
                    except Exception as e:
                        st.error(f"Błąd: {e}")
                        text = ""
                else:
                    try:
                        text = ask_ai(client, model, messages)
                        st.markdown(text)
                    except Exception as e:
                        st.error(f"Błąd: {e}")
                        text = ""

        if text:
            st.session_state.history.append({"role": "assistant", "content": text})
else:
    st.info("Wgraj plik, aby rozpocząć pracę z agentem.")
