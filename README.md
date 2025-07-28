# Transport Agent – Tholen + Moerdijk

- Wczytuje arkusze **Tholen**, **Moerdijk** (także `Moerdijk `), scala je w jeden *Planing*.
- Rozdziela kolumnę `Van - Tot` na `Van` i `Tot`.
- Dla Tholen automatycznie dodaje adres: **Marconiweg 3, 4691 SV Tholen**.
- Wczytuje arkusze **Pracownicy** (opcjonalny) i **Kierowcy** (wymagany).
- Filtruje kontekst po godzinie i lokalizacji, minimalizując koszty API.
- Responses API z **streamingiem** odpowiedzi.

## Uruchom w Streamlit Cloud
1. Prześlij pliki do repo na GitHub.
2. Deploy: *Create app* → `app_streamlit_transport_v2.py`.
3. W **Secrets** dodaj:
   ```
   OPENAI_API_KEY = "sk-..."
   ```

## Lokalnie (opcjonalnie)
```
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app_streamlit_transport_v2.py
```
