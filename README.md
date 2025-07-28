# Transport Agent v3 – z twardym fallbackiem env

- Klucz pobierany z `st.secrets["OPENAI_API_KEY"]` **i** ustawiany w `os.environ["OPENAI_API_KEY"]`.
- Jeśli w `st.secrets` brak, używany jest `os.getenv("OPENAI_API_KEY")`.
- Parser łączy arkusze **Tholen** i **Moerdijk**, rozdziela `Van - Tot` na `Van`/`Tot`, dodaje adres dla Tholen.

## Ustawienie klucza (Cloud)
W panelu Streamlit → **Settings → Secrets** wklej:
```
OPENAI_API_KEY = "sk-..."
```

## Lokalnie (opcjonalnie)
`.streamlit/secrets.toml`:
```
OPENAI_API_KEY = "sk-..."
```

Lub zmienna środowiskowa:
```
export OPENAI_API_KEY="sk-..."
```

## Start
```
pip install -r requirements.txt
streamlit run app_streamlit_transport_v2.py
```
