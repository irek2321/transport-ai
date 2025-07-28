import re
import os
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI

# -------------------------------
# Konfiguracja i stałe
# -------------------------------
APP_TITLE = "🚐 Agent AI do transportu pracowników"
SYSTEM_INSTRUCTION = (
    "Jesteś asystentem logistycznym. Odpowiadaj zwięźle, po polsku. "
    "Gdy brakuje danych, jasno napisz czego potrzebujesz (kolumny, wartości, godziny)."
)

# -------------------------------
# Pomocnicze
# -------------------------------

def get_api_key() -> Optional[str]:
    """Preferuj st.secrets, potem zmienną środowiskową."""
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def parse_time_candidates(text: str) -> List[str]:
    """Wyłap czasy typu 6:00, 06:00, 6.00, 6 czy 6-00."""
    patt = r"\b(\d{1,2})[:\.-]?(\d{2})?\b"
    times: List[str] = []
    for h, m in re.findall(patt, text):
        hour = int(h)
        if 0 <= hour <= 23:
            minute = int(m) if m else 0
            times.append(f"{hour:02d}:{minute:02d}")
    return list(dict.fromkeys(times))  # unikalne, zachowaj kolejność


def parse_locations(text: str) -> List[str]:
    # Heurystyka: słowa z wielkiej litery, z polskimi znakami i/lub myślnikiem
    words = re.findall(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż\-]+", text)
    locs = [w for w in words if w[:1].isupper() or "-" in w]
    stop = {"Na", "Do", "Z", "I", "Oraz", "Od", "Kto", "Kiedy", "Który"}
    locs = [w for w in locs if w.capitalize() not in stop]
    return list(dict.fromkeys(locs))


def filter_context(
    planing: pd.DataFrame,
    kierowcy: pd.DataFrame,
    pytanie: str,
    max_rows: int = 120,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Zawężenie danych do tego, co prawdopodobnie potrzebne."""
    p = clean_columns(planing)
    k = clean_columns(kierowcy)

    times = parse_time_candidates(pytanie)
    locs = parse_locations(pytanie)

    pf = p

    # Filtruj po czasie
    time_cols = [c for c in pf.columns if re.search(r"czas|godz|start|wyjazd|odjazd", c, re.I)]
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

    # Filtruj po lokalizacji
    if locs:
        loc_cols = [c for c in p.columns if re.search(r"miast|lok|skad|dokad|trasa|adres|miejsc|punkt|from|to", c, re.I)]
        if loc_cols:
            mask_loc = False
            for loc in locs:
                patt = rf"\b{re.escape(loc)}\b"
                mcol = False
                for col in loc_cols:
                    mcol = mcol | p[col].astype(str).str.contains(patt, case=False, regex=True, na=False)
                mask_loc = mask_loc | mcol
            pf = pf[mask_loc]

    # Ogranicz rozmiar
    if len(pf) > max_rows:
        pf = pf.head(max_rows)

    return pf, k


def df_to_compact_text(df: pd.DataFrame, max_chars: int = 20_000) -> str:
    """CSV bez indeksu, przycięcie gdy za długie."""
    if df.empty:
        return "(pusta tabela)"
    csv = df.to_csv(index=False)
    if len(csv) <= max_chars:
        return csv
    head = csv[:max_chars]
    return head + f"\n... (przycięto, łączna długość={len(csv)} znaków)"


def build_input(plan_csv: str, drivers_csv: str, pytanie: str) -> List[Dict[str, Any]]:
    """Zbuduj wejście do Responses API z rolą user."""
    text = (
        "Masz odpowiedzieć na pytanie dotyczące planu transportu. "
        "Dane są w CSV poniżej. Jeśli podajesz listy osób/kierowców, dodaj pełne informacje "
        "potrzebne do realizacji kursu (godzina, skąd→dokąd, pojazd/bus, numer telefonu jeśli jest).\n\n"
        f"Pytanie:\n{pytanie}\n\n"
        f"Planing CSV:\n{plan_csv}\n\n"
        f"Kierowcy CSV:\n{drivers_csv}\n"
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": text}
            ],
        }
    ]


def stream_openai_text(client: OpenAI, model: str, instructions: str, input_payload: List[Dict[str, Any]]):
    """Generator tekstu do st.write_stream."""
    stream = client.responses.create(model=model, instructions=instructions, input=input_payload, stream=True)
    for event in stream:
        etype = getattr(event, "type", "")
        if etype == "response.output_text.delta":
            yield event.delta  # typ: str
        elif etype == "response.error":
            raise RuntimeError(getattr(event, "error", "Błąd API"))


def ask_ai(client: OpenAI, model: str, instructions: str, input_payload: List[Dict[str, Any]]) -> str:
    resp = client.responses.create(model=model, instructions=instructions, input=input_payload)
    return resp.output_text

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

api_key = get_api_key()
if not api_key:
    st.warning("Ustaw OPENAI_API_KEY w st.secrets lub w zmiennych środowiskowych.")

client = OpenAI(api_key=api_key) if api_key else None

uploaded_file = st.file_uploader(
    "Wgraj plik Excel z arkuszami: Planing, Kierowcy", type=[".xlsx", ".xlsm"]
)

@st.cache_data(show_spinner=False)
def load_data(file) -> Tuple[pd.DataFrame, pd.DataFrame]:
    xls = pd.ExcelFile(file)
    plan = pd.read_excel(xls, "Planing")
    drivers = pd.read_excel(xls, "Kierowcy")
    return plan, drivers

if uploaded_file:
    try:
        planing, kierowcy = load_data(uploaded_file)
        st.success("Plik załadowany poprawnie.")
    except Exception as e:
        st.error(f"Błąd podczas wczytywania: {e}")
        st.stop()

    # Podgląd
    with st.expander("Podgląd danych – Planing"):
        st.dataframe(planing, use_container_width=True, hide_index=True)
    with st.expander("Podgląd danych – Kierowcy"):
        st.dataframe(kierowcy, use_container_width=True, hide_index=True)

    # Konfiguracja modeli
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

    # Historia czatu
    if "history" not in st.session_state:
        st.session_state.history = []  # list[dict(role, content)]

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Zadaj pytanie (np. Kto jedzie na 6:00 z Roosendaal?)")
    if user_q:
        st.session_state.history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        pf, kf = filter_context(planing, kierowcy, user_q, max_rows=max_rows)
        plan_csv = df_to_compact_text(pf)
        drivers_csv = df_to_compact_text(kf)
        input_payload = build_input(plan_csv, drivers_csv, user_q)

        with st.chat_message("assistant"):
            if not client:
                st.error("Brak klucza API.")
                text = ""
            else:
                if stream_answer:
                    try:
                        text = st.write_stream(
                            stream_openai_text(client, model, SYSTEM_INSTRUCTION, input_payload)
                        )
                    except Exception as e:
                        st.error(f"Błąd: {e}")
                        text = ""
                else:
                    try:
                        text = ask_ai(client, model, SYSTEM_INSTRUCTION, input_payload)
                        st.markdown(text)
                    except Exception as e:
                        st.error(f"Błąd: {e}")
                        text = ""

        if text:
            st.session_state.history.append({"role": "assistant", "content": text})
else:
    st.info("Wgraj plik, aby rozpocząć pracę z agentem.")
