# Dubinska analiza podataka — NLP za dating aplikacije

Projekat za predmet _Dubinska analiza podataka_ (ETF Sarajevo).

Cilj: primjena pet NLP rješenja na korisničko iskustvo i sigurnost
dating aplikacija (Tinder/Bumble/Hinge…).

Repozitorij pokriva prikupljanje, sintetičko stvaranje i preprocesiranje
podataka za sve NLP taskove.

---

## Mapiranje NLP taskova → izvori podataka

| #   | NLP Task                                  | Primarni dataset                                | Dopunski / sintetički                       |
| --- | ----------------------------------------- | ----------------------------------------------- | ------------------------------------------- |
| 1   | Sistem za preporučivanje (bio embeddings) | OkCupid Profiles (Kaggle)                       | —                                           |
| 2   | Identifikacija govora mržnje              | Davidson 2017 (GitHub) + `tweet_eval/hate` (HF) | —                                           |
| 3   | Modeliranje tema / Icebreakers            | OkCupid Profiles (Kaggle)                       | Sintetički icebreaker parovi                |
| 4   | Detekcija prevara / botova                | SMS Spam Collection (UCI)                       | Sintetički romance-scam profili             |
| 5   | Dinamička analiza sentimenta i emocija    | Google Play recenzije (Tinder/Bumble/Hinge/CMB) | Sintetički razgovori sa progresijom emocija |

Više datasetova je nužno — _jedan dataset ne može pokriti sve taskove_
(bot detekcija traži kratke poruke, preporuke traže duge bio sekcije).

---

## Struktura projekta

```
dap/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/         # sirovi preuzeti / scraped fajlovi
│   ├── processed/   # očišćeni CSV-ovi spremni za modeliranje
│   └── synthetic/   # sintetički generisani fajlovi
├── notebooks/
│   └── 01_eda.ipynb
├── reports/         # statistike, distribucije, kasnije evaluacije
│   └── recommendation/
│       ├── bio_recommendation_demo_results.csv
│       ├── bio_recommendation_evaluation_details.csv
│       └── bio_recommendation_evaluation_summary.csv    
└── src/
    ├── recommendation/
    │   ├── bio_recommender.py
    │   ├── run_bio_recommendation_demo.py
    │   └── evaluate_bio_recommender.py
    ├── ui/
    │   └── recommendation_app.py
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
# 1) (preporuka) virtualno okruzenje
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) zavisnosti
pip install -r requirements.txt

# 3) NLTK resursi (stopwords, punkt, wordnet)
python -m src.utils.nltk_setup
```

### Kaggle kredencijali (OkCupid)

OkCupid dataset se preuzima preko Kaggle API-ja. Potrebno je jednom prihvatiti uslove dataseta na
[andrewmvd/okcupid-profiles](https://www.kaggle.com/datasets/andrewmvd/okcupid-profiles)
(kliknuti **I Understand and Accept**), a kredencijale postaviti na jedan od dva načina:

**Opcija A — `.env` fajl**

```powershell
copy .env.example .env
# otvoriti .env i upisati username i key
```

Username i API key se nalaze na [kaggle.com](https://www.kaggle.com/) → **Settings** → **API**.

**Opcija B — `kaggle.json`**

Preuzeti token sa iste stranice (**Create New API Token**) i smjestiti ga na:
`C:\Users\<ime>\.kaggle\kaggle.json`

---

## Pokretanje cijelog pipeline-a

```powershell
# sve faze (collect → synthetic → preprocess)
python -m src.run_pipeline

# bez Google Play scraping-a (npr. ako nema interneta ili je blokiran)
python -m src.run_pipeline --skip-scrape

# samo jedna faza
python -m src.run_pipeline --only collect
python -m src.run_pipeline --only synthetic
python -m src.run_pipeline --only preprocess
```

Svaka skripta se može pokrenuti i samostalno, npr.:

```powershell
python -m src.data_collection.download_okcupid
python -m src.preprocessing.preprocess_hate_speech
```

---

## Faze pipeline-a

### 1. Prikupljanje (`src/data_collection/`)

| Skripta                    | Izvor                                          | Izlaz                                         |
| -------------------------- | ---------------------------------------------- | --------------------------------------------- |
| `download_okcupid.py`      | Kaggle: `andrewmvd/okcupid-profiles`           | `data/raw/okcupid/okcupid_profiles.csv`       |
| `download_hate_speech.py`  | GitHub raw + Hugging Face `tweet_eval`         | `data/raw/hate_speech/hate_combined.csv`      |
| `download_sms_spam.py`     | UCI ML Repository                              | `data/raw/sms_spam/sms_spam.csv`              |
| `scrape_tinder_reviews.py` | Google Play Store (Tinder, Bumble, Hinge, CMB) | `data/raw/app_reviews/dating_app_reviews.csv` |

### 2. Sintetičko generisanje (`generate_synthetic.py`)

Razlozi:

- Nedostaju javne baze sa romance scam profilima iz dating aplikacija.
- Trebamo _kontrolisane_ labele za fine-grained emocije
  (frustracija, sarkazam, pad interesovanja).

Output:

- `data/synthetic/scam_profiles.csv` — 1500 legit + 1500 scam bio
- `data/synthetic/icebreaker_pairs.csv` — 1200 (bio_a, bio_b, icebreaker)
- `data/synthetic/conversations.csv` — ~600 razgovora po šablonima

### 3. Preprocesiranje (`src/preprocessing/`)

Sve skripte koriste zajednički modul `text_cleaning.py` koji nudi tri
pipelines (različiti taskovi traže različit tretman):

- `clean_for_embeddings` — preporuka + topic modeling
  (zadržava semantiku, prevodi emojije u tekst)
- `clean_for_classification` — govor mržnje + bot detekcija
  (agresivnija normalizacija, brojeve mijenja sa `<NUM>`)
- `clean_for_sentiment` — recenzije i razgovori
  (zadržava `!`, `?`, negacije i emojije — sve nose signal)

Sve tri pipelines rade:

1. `ftfy` + `unicodedata.NFKC` (popravljanje pokvarene Unicode kodacije)
2. Zamjena URL-ova / e-mail-ova / telefona placeholder tokenima
3. Demojizacija ili uklanjanje emojija
4. Smanjenje ponovljenih karaktera (`soooo` → `soo`)
5. Skidanje interpunkcije (osim u sentiment pipeline-u)
6. NLTK tokenizacija + stop-words filter (engleska osnova,
   _zadržavamo negacije_ tipa `not`, `never`)
7. WordNet lematizacija

Output (u `data/processed/`):

| Task                | Fajl                                                    |
| ------------------- | ------------------------------------------------------- |
| Preporuke           | `bios_for_embeddings.csv`                               |
| Topics              | `bios_for_topics.csv`                                   |
| Govor mržnje        | `hate_speech_clean.csv` + `_train/_val/_test.csv`       |
| Scam/bot            | `scam_detection_clean.csv` + `_train/_val/_test.csv`    |
| Sentiment recenzije | `sentiment_reviews_clean.csv` + `_train/_val/_test.csv` |
| Sentiment razgovori | `sentiment_conversations_clean.csv`                     |

Stratifikovani 70/15/15 split koristi `sklearn.model_selection.train_test_split`
sa `random_state=42` zbog reproducibilnosti.

### 4. EDA

```powershell
jupyter notebook notebooks/01_eda.ipynb
```

Notebook prikazuje:

- distribucije dužine teksta i broj tokena
- balans klasa po taskovima i izvorima
- top frekventne tokene u bios
- razlike u dužini / URL / telefon između legitimnih i scam poruka
- raspodjelu sentimenta po aplikaciji

---

## Etika i privatnost podataka

- **OkCupid**: dataset je već anonimiziran, javno objavljen 2016. Bio
  sekcije su slobodne forme — pri tokenizaciji se uklanjaju imena,
  e-mailovi i telefoni preko regex-a u `text_cleaning.py`.
- **Google Play recenzije**: ne sadrže osjetljive podatke, ali u
  `userName` koloni može biti puno ime. U procesiranom CSV-u zadržavamo
  samo `app`, `rating`, `text` i izvedene kolone.
- **Sintetički scam tekstovi**: koriste Faker-ovo nasumično generisanje
  (`name`, `phone`, `email`) — _nisu_ stvarni ljudski podaci.
- **Hate speech (Davidson)**: tweet-ovi su javni i već anonimizirani od
  strane autora dataseta.

---

## Reproducibilnost

- Sve random operacije postavljaju `random_state=42` (sklearn) ili
  `Faker.seed(42) / random.Random(42)`.
- `requirements.txt` fiksira _minimalne_ verzije — za 100% repro
  preporučljivo je generisati `pip freeze > requirements.lock.txt` na
  kraju ciklusa.

---

## Mapiranje članova tima na NLP taskove

Svaki član tima radi na svom NLP tasku, koristeći odgovarajuće
`data/processed/*.csv` fajlove kao ulaz:

| Član                         | Task                                                            | Ulazni fajl(ovi) |
| ---------------------------- | --------------------------------------------------------------- | ---------------- |
| Sistem preporuka             | `bios_for_embeddings.csv`                                       |
| Govor mržnje                 | `hate_speech_{train,val,test}.csv`                              |
| Topic modeling / icebreakers | `bios_for_topics.csv` + `icebreaker_pairs.csv`                  |
| Bot/scam detekcija           | `scam_detection_{train,val,test}.csv`                           |
| Sentiment + emocije          | `sentiment_reviews_*.csv` + `sentiment_conversations_clean.csv` |

---

## Task 1: Sistem za preporučivanje profila na osnovu bio embeddings

Ovaj task implementira content-based sistem za preporučivanje profila u dating aplikaciji. Sistem koristi tekstualne bio opise korisnika iz OkCupid Profiles dataseta i vraća top-N najsličnijih profila.

Za reprezentaciju teksta koriste se dvije forme prikaza:

1. **TF-IDF reprezentacija** — klasična sparse reprezentacija teksta.
2. **SBERT sentence embeddings** — duboko kontekstualna reprezentacija teksta.

Obje metode koriste **cosine similarity** za rangiranje kandidata.

### Ulazni fajl

Preporučivački sistem koristi preprocesirani fajl:

```text
data/processed/bios_for_embeddings.csv
Ovaj fajl nastaje pokretanjem:

python -m src.preprocessing.preprocess_bios

ili kroz kompletan preprocessing pipeline:

python -m src.run_pipeline --only preprocess
Pokretanje recommender sistema iz terminala

Primjer preporuka za postojeći profil:

python -m src.recommendation.bio_recommender --method both --index 0 --top-k 5 --max-profiles 1000

Dostupne metode su:

tfidf
sbert
both

Primjer sa samo TF-IDF metodom:

python -m src.recommendation.bio_recommender --method tfidf --index 0 --top-k 5 --max-profiles 5000

Primjer sa SBERT metodom:

python -m src.recommendation.bio_recommender --method sbert --index 0 --top-k 5 --max-profiles 1000
Demo rezultati

Za generisanje CSV fajla sa demo preporukama:

python -m src.recommendation.run_bio_recommendation_demo

Output se snima u:

reports/recommendation/bio_recommendation_demo_results.csv
Evaluacija recommender sistema

Pošto OkCupid dataset ne sadrži stvarne match/rating labele, koristi se proxy evaluacija. Sistem se poredi sa random baseline-om kroz metapodatke preporučenih profila, npr. sličnost po gradu, statusu, orijentaciji, obrazovanju, poslu i prosječnoj razlici u godinama.

Pokretanje evaluacije:

python -m src.recommendation.evaluate_bio_recommender --max-profiles 1000 --n-queries 50 --top-k 5

Output fajlovi:

reports/recommendation/bio_recommendation_evaluation_details.csv
reports/recommendation/bio_recommendation_evaluation_summary.csv
Streamlit UI

Za demonstraciju taska kroz grafički interfejs:

python -m streamlit run src/ui/recommendation_app.py

UI omogućava:

izbor postojećeg profila iz dataseta,
unos novog bio teksta,
izbor metode: TF-IDF, SBERT ili obje metode,
prikaz top-N preporučenih profila.

Za prezentaciju je preporučeno koristiti:

Broj profila za demo: 1000
Broj preporuka: 5
Metoda: Obje metode
Način unosa: Postojeći profil
Index profila: 0
```
