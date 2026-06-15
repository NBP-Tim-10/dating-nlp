# Dubinska analiza podataka — NLP za dating aplikacije

Projekat za predmet _Dubinska analiza podataka_ (ETF Sarajevo).  
Implementira šest NLP rješenja za unapređenje korisničkog iskustva i sigurnosti dating aplikacija.

---

## NLP taskovi

| # | Task | Dataset | Metode |
|---|------|---------|--------|
| 1 | Sistem za preporučivanje profila | OkCupid Profiles (Kaggle) | TF-IDF, SBERT + cosine similarity |
| 2 | Detekcija govora mržnje | Davidson 2017 + `tweet_eval/hate` (HF) | Klasifikacijski modeli |
| 3 | Detekcija prevara i botova | SMS Spam Collection + sintetički scam profili | TF-IDF + feature ekstrakcija |
| 4 | Analiza sentimenta recenzija | Google Play recenzije (Tinder/Bumble/Hinge/CMB) | Sentiment klasifikacija |
| 5 | Analiza emocija u razgovorima | Sintetički razgovori s progresijom emocija | Klasifikacija emocija |
| 6 | Generisanje icebreaker poruka | PersonaChat (HF) + Claude Haiku 4.5 generacija | QLoRA fine-tuning Llama-3.2-3B |

---

## Struktura projekta

```
dap/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/              # sirovi preuzeti / scraped fajlovi
│   ├── processed/        # očišćeni fajlovi spremni za modeliranje
│   │   └── icebreaker/   # personas.jsonl, dataset_train/eval.jsonl, retrieval fajlovi
│   └── synthetic/        # sintetički generisani fajlovi
├── docs/                 # LaTeX dokumentacija
│   ├── main.tex
│   ├── Uvod.tex
│   ├── Skup_podataka.tex
│   ├── Icebreaker.tex
│   └── references.bib
├── notebooks/
│   ├── 01_eda.ipynb
│   └── icebreaker_finetuning.ipynb   # QLoRA trening + evaluacija (Colab T4)
├── reports/
│   └── recommendation/
└── src/
    ├── icebreaker/
    │   ├── config.py
    │   ├── extract_personas.py       # Faza 0: ekstrakcija persona iz PersonaChat
    │   ├── generate_dataset.py       # Faza 1: generacija bio+icebreaker parova (Claude API)
    │   └── retrieve_icebreakers.py   # Faza 0.5: TF-IDF vs MiniLM retrieval baseline
    ├── recommendation/
    │   ├── bio_recommender.py
    │   ├── run_bio_recommendation_demo.py
    │   └── evaluate_bio_recommender.py
    ├── ui/
    │   ├── recommendation_app.py     # Streamlit: preporučivanje profila
    │   ├── hate_speech_app.py        # Streamlit: detekcija govora mržnje
    │   ├── scam_detection_app.py     # Streamlit: detekcija prevara
    │   ├── sentiment_emotions_app.py # Streamlit: sentiment i emocije
    │   └── icebreaker_demo.ipynb     # Gradio demo (pokreće se na Colab)
    ├── data_collection/
    │   ├── download_okcupid.py
    │   ├── download_hate_speech.py
    │   ├── download_sms_spam.py
    │   ├── scrape_tinder_reviews.py
    │   └── generate_synthetic.py
    ├── preprocessing/
    │   ├── text_cleaning.py
    │   ├── preprocess_bios.py
    │   ├── preprocess_hate_speech.py
    │   ├── preprocess_bot_detection.py
    │   └── preprocess_sentiment.py
    ├── utils/
    │   ├── paths.py
    │   └── nltk_setup.py
    └── run_pipeline.py
```

---

## Postavljanje okruženja

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.utils.nltk_setup
```

### Kaggle kredencijali (OkCupid)

Prihvatiti uslove dataseta na [andrewmvd/okcupid-profiles](https://www.kaggle.com/datasets/andrewmvd/okcupid-profiles), zatim:

```powershell
copy .env.example .env
# upisati KAGGLE_USERNAME i KAGGLE_KEY u .env
```

### API ključevi (icebreaker task)

```powershell
# u .env fajl dodati:
ANTHROPIC_API_KEY=...   # generacija trening skupa (Claude Haiku 4.5)
OPENAI_API_KEY=...      # evaluacija (GPT-4o-mini sudija)
```

---

## Pokretanje pipeline-a (taskovi 1–5)

```powershell
# sve faze: collect → synthetic → preprocess
python -m src.run_pipeline

# preskoči Google Play scraping
python -m src.run_pipeline --skip-scrape

# pojedinačne faze
python -m src.run_pipeline --only collect
python -m src.run_pipeline --only synthetic
python -m src.run_pipeline --only preprocess
```

---

## Icebreaker task (Task 6)

Pipeline se izvodi u četiri faze:

**Faza 0 — Ekstrakcija persona**
```powershell
python -m src.icebreaker.extract_personas
# izlaz: data/processed/icebreaker/personas.jsonl (500 persona)
```

**Faza 1 — Generacija dataseta (Claude Haiku 4.5)**
```powershell
python -m src.icebreaker.generate_dataset
# izlaz: dataset_train.jsonl (1272 primjera) + dataset_eval.jsonl (225 primjera)
```

**Faza 0.5 — Retrieval baseline (TF-IDF vs MiniLM)**
```powershell
python -m src.icebreaker.retrieve_icebreakers
# ispisuje komparativnu tabelu: P@k, R@k, MRR za obje reprezentacije
```

**Faza 2 & 3 — QLoRA fine-tuning i evaluacija**  
Pokrenuti na Google Colab T4 GPU: `notebooks/icebreaker_finetuning.ipynb`  
Zahtijeva: Runtime → Change runtime type → T4 GPU

---

## Streamlit UI (taskovi 1–5)

```powershell
streamlit run src/ui/recommendation_app.py
streamlit run src/ui/hate_speech_app.py
streamlit run src/ui/scam_detection_app.py
streamlit run src/ui/sentiment_emotions_app.py
```

## Gradio demo (Task 6 — icebreaker)

Demo pokrenut na Colabu: `src/ui/icebreaker_demo.ipynb`  
Prikazuje side-by-side poređenje tri pristupa za unos bio teksta:
- **TF-IDF retrieval** — top-k iz banke po leksičkoj sličnosti
- **MiniLM retrieval** — top-k iz banke po semantičkoj sličnosti  
- **Fine-tuned Llama-3.2-3B** — slobodna generacija (nije ograničena bankom)

---

## Rezultati (icebreaker evaluacija, n=225)

| Metrika | Vrijednost |
|---------|-----------|
| GPT-4o-mini sudija — ukupno | **4.40 / 5** |
| GPT-4o-mini sudija — ton | 4.91 / 5 |
| GPT-4o-mini sudija — specifičnost | 4.65 / 5 |
| Perpleksnost (eval, n=50) | 3.55 |
| MiniLM retrieval MRR | 0.5217 |
| TF-IDF retrieval MRR | 0.3390 |

---

## Etika i privatnost

- **OkCupid**: javno anonimiziran dataset; pri preprocesiranju se uklanjaju e-mailovi i telefoni.
- **Google Play recenzije**: zadržavaju se samo `app`, `rating` i `text`.
- **Sintetički scam profili**: generirani Faker-om — nisu stvarni podaci korisnika.
- **PersonaChat**: izmišljeni karakteri, MIT licenca — bez scrape-ovanih ličnih podataka.

---

## Reproducibilnost

Sve random operacije koriste `seed=42`. Za potpunu reproducibilnost:

```powershell
pip freeze > requirements.lock.txt
```
