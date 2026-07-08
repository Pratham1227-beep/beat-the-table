# ⚽ Beat the Table — AI Football Analytics Dashboard

> A premium AI-powered football analytics platform that predicts match outcomes, recommends tactical lineups, detects opponent mismatches, and runs Monte Carlo tournament projections — all in a sleek, modern Streamlit dashboard.

---

## 🚀 Features

| Feature | Description |
|---|---|
| **Match Prediction** | XGBoost model outputs Win / Draw / Loss probabilities using ELO, form, xG, H2H data |
| **AI Commentary** | SHAP-powered natural language explanation of every prediction |
| **Starting XI Advisor** | Tactically-optimised 11-player lineup vs opponent style with rating bars |
| **Portrait Pitch View** | Vertical green football pitch with players placed at exact formation positions |
| **Weak Link Detector** | Scans opponent threats and highlights your 3 biggest defensive mismatches |
| **Formation Advisor** | Try any formation and see updated win % and tactical notes |
| **Tournament Projection** | Monte Carlo simulation (1,000 runs) of bracket advancement probability per stage |
| **Historical Database** | Full 2018–2026 World Cup prediction backtest browser |
| **Streamlit Caching** | Model, squad data, and simulations are cached — near-instant repeat queries |

---

## 📁 Project Structure

```
beat-the-table/
├── app.py                        # Streamlit dashboard (UI, routing, pitch rendering)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── models/
│   └── model.pkl                 # Trained XGBoost model + feature names
├── data/
│   ├── raw/
│   │   ├── results.csv           # International match results (1872–2024)
│   │   └── players_fifa23.csv    # FIFA 23 player dataset (cached)
│   └── processed/
│       ├── team_stats.csv        # ELO, form, xG/xGA per team
│       └── world_cup_predictions_2018_2026.csv  # Historical backtest results
└── src/
    ├── data_pipeline.py          # ELO scraper, stat aggregator, normalization
    ├── predict_engine.py         # XGBoost training + cached predict_match()
    ├── explain.py                # SHAP explainer + natural-language generator
    ├── lineup_engine.py          # Tactical Starting XI selector (cached)
    ├── tournament_sim.py         # Monte Carlo bracket simulator (cached)
    ├── weak_link_detector.py     # Mismatch scorer vs opponent threats
    ├── backtest.py               # Historical backtest engine (2018 & 2022 WCs)
    └── compile_predictions.py    # Compiles all backtest predictions into a CSV
```

---

## 🛠️ Setup & Running

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Requires Python 3.10+ and the packages: `streamlit`, `xgboost`, `scikit-learn`, `pandas`, `numpy`, `shap`, `plotly`, `matplotlib`.

### 2. Run the app

```bash
python -m streamlit run app.py
```

> Use `python -m streamlit` (not just `streamlit`) to ensure the correct Python environment is used.

The dashboard opens at **http://localhost:8501**

---

## 📊 Datasets Used

| # | Dataset | Source |
|---|---|---|
| 1 | International match results (1872–2024) | [Kaggle — martj42](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) |
| 2 | FIFA World Cup historical data | [Kaggle — abecklas](https://www.kaggle.com/datasets/abecklas/fifa-world-cup) |
| 3 | Live ELO ratings | [eloratings.net/World.tsv](https://www.eloratings.net/World.tsv) |
| 4 | Player ratings & attributes | [FIFA 23 Complete Player Dataset](https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset) |

---

## 🧠 Model Architecture

### Prediction: XGBoost Multiclass Classifier

The win/draw/loss prediction is powered by an XGBoost model trained on international match results:

**Why XGBoost?**
- Tabular features (ELO diff, form diff, xG diff) have non-linear relationships — gradient boosted trees fit these far better than linear models
- Native `multi:softprob` objective returns calibrated Win/Draw/Loss probabilities that sum to 1.0
- Built-in L1/L2 regularisation prevents overfitting on correlated features (ELO and squad rating tend to correlate)
- Naturally models decision thresholds (e.g. "ELO gap > 200 → near-certain win")

**Features used:**

| Feature | Description |
|---|---|
| `elo_diff` | ELO rating gap between Team A and Team B |
| `form_diff` | Weighted recent form score difference (last 10 games) |
| `xg_diff` | Attacking strength gap (goals scored / xG estimate) |
| `xga_diff` | Defensive strength gap (goals conceded / xGA estimate) |
| `squad_rating_diff` | FIFA 23 top-23 average overall rating difference |
| `h2h_home` | Historical H2H win rate for Team A |
| `h2h_draw` | Historical H2H draw rate |
| `h2h_away` | Historical H2H win rate for Team B |

### Lineup Engine: Rule-Based Tactical Selector

**Why rules, not ML?**
- No public dataset links coach decisions to opponent style profiles
- Tactical rules are interpretable, explainable, and cold-start capable for any team
- Each pick includes a plain-English rationale for full transparency

**6 Tactical Adjustment Rules:**

| Rule | Position | Triggers When | Effect |
|---|---|---|---|
| 1 | CB | Opponent = aerial threat | +5 if heading > 78 / -3 if defending < 72 |
| 2 | CM / CDM / CAM | Opponent = high press | +4 if composure or dribbling > 78 / -2 otherwise |
| 3 | CB / LB / RB | Opponent = counter pace | +5 if pace > 78 / -4 if pace < 65 |
| 4 | MID + ATT | Opponent = possession heavy | +3 if passing or dribbling > 78 |
| 5 | CB / MID | Opponent = high press | +3 if physic or stamina > 80 |
| 6 | ST / LW / RW | Opponent = possession heavy | +3 if shooting > 80 |

---

## 🖼️ UI Design Highlights

- **Design system:** Outfit (headings, 900 weight) + Inter (body) via Google Fonts
- **Colour palette:** Deep navy `#080e1c` background, blue `#60a5fa` accent, red `#f43f5e` opponent, amber `#f59e0b` alerts
- **Portrait pitch:** Matplotlib-rendered vertical football pitch with grass green (`#0d7a3e`) and full pitch markings (penalty area, 6-yard box, centre circle)
- **Premium tables:** Custom HTML tables with position-coloured badges (🟡 GK / 🔵 DEF / 🟢 MID / 🔴 ATT) and mini rating bars
- **Glassmorphism sidebar:** `linear-gradient(160deg, #0f172a, #0a1628)` with subtle border
- **Weak Link cards:** No image placeholders — clean name-vs-name stat pill badges with severity colour coding
- **Caching:** `@st.cache_resource` for the model, `@st.cache_data` for CSVs and simulations

---

## 📈 Historical Backtest Results

Tested against **actual** 2018 and 2022 World Cup results using data frozen *before* each tournament:

| Tournament | Model Accuracy | Naive Rank Baseline |
|---|---|---|
| **2018 World Cup** | 43.8% | 50.0% |
| **2022 World Cup** | 48.4% | 57.8% |

> The model predicts all three outcomes (Win/Draw/Loss). The naive baseline only picks the higher-ranked team. Despite lower headline accuracy, the model correctly flags major upsets.

**Notable correctly predicted upsets:**
- 🇷🇺 Russia def. Egypt (predicted 45.9% Win)
- 🇳🇬 Nigeria def. Iceland (predicted 52.3% Win)
- 🇭🇷 Croatia def. England (predicted 39.2% Win)
- 🇰🇷 Korea Republic def. Portugal (predicted 38.7% Win)
- 🇲🇦 Morocco def. Portugal (predicted 36.6% Win)

---

## ⚠️ Known Limitations

1. **FBRef scraping blocked (Cloudflare):** The pipeline falls back to estimating xG from average goals if `fbref_countries.html` is not manually placed in `data/raw/`.
2. **Kaggle API:** ELO ratings are pulled directly from `eloratings.net` — no Kaggle login required.
3. **Squad data staleness:** Player attributes are based on the FIFA 23 dataset (2022–23 season). Players who emerged or declined after that season are not reflected.
4. **Neutral venue assumption:** All matches are modelled as neutral-venue fixtures. Home-ground advantage is not separately accounted for in World Cup group stage games.
5. **Thin H2H:** If two teams have fewer than 5 historical meetings, H2H features default to 33.3% per outcome.


Built for hackathon purposes. All data sourced from publicly available datasets under their respective Kaggle licences.
