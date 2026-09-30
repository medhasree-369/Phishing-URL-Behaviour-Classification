# Phishing URL Behaviour Classification

**Track 1 · Final Capstone · Problem 02**
Learn Depth Academy LLP | Domain: Cybersecurity | ML Task: Binary Classification

## Problem Statement
Classify web addresses as **legitimate** or **suspicious (phishing)** using structural
and lexical URL characteristics, without relying on paid security APIs.

## Dataset
- **Source:** [UCI Machine Learning Repository — Phishing Websites](https://archive.ics.uci.edu/dataset/327/phishing) (Mohammad, R. & McCluskey, L., 2012)
- **Size:** 11,055 labeled websites, 30 categorical/lexical features (-1 / 0 / 1 encoded)
- **License:** CC BY 4.0
- **File used:** `data/phishing_dataset.csv` (header row added for readability)

## Project Structure
```
phishing_project/
├── data/
│   └── phishing_dataset.csv        # Raw dataset (with headers)
├── notebook/
│   └── train_pipeline.py           # Full EDA + training + evaluation pipeline
├── model/
│   ├── best_model.pkl              # Trained classifier (best of 3)
│   ├── scaler.pkl                  # StandardScaler fit on training data
│   └── feature_names.pkl           # Ordered feature list expected by the model
├── app/
│   └── app.py                      # Streamlit prediction app
├── outputs/                        # EDA plots, confusion matrices, ROC curves, JSON report
├── requirements.txt
└── README.md
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Reproduce the EDA + training pipeline
```bash
python notebook/train_pipeline.py
```
This regenerates all plots in `outputs/`, retrains all three models, and saves the
best-performing one to `model/`.

### 3. Launch the Streamlit app
```bash
streamlit run app/app.py
```
Paste any URL and get an instant Legitimate/Phishing prediction with a risk score.

## Methodology Summary
- **Preprocessing:** No missing values found. Duplicate rows (5,206 of 11,055) were
  **retained** — with only 30 low-cardinality categorical features, many distinct
  websites legitimately share identical feature vectors; this is a known property of
  this dataset, not a data-entry error.
- **Data leakage check:** No single feature exceeds |correlation| > 0.9 with the
  target, so no leakage was found. Strongest predictors: `sslfinal_state` (-0.72),
  `url_of_anchor` (-0.69), `prefix_suffix` (-0.35), `web_traffic` (-0.35).
- **Split:** 80/20 stratified train-test split, features scaled with `StandardScaler`.
- **Models trained:** Logistic Regression, Decision Tree (max_depth=8), KNN (k=7).

## Results (test set, n=2,211)

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.928 | 0.934 | 0.901 | 0.917 | 0.979 |
| Decision Tree | 0.935 | 0.942 | 0.909 | 0.925 | 0.985 |
| **KNN (k=7)** | **0.945** | **0.953** | **0.921** | **0.937** | **0.988** |

**Selected model: KNN (k=7)** — best F1-score and ROC-AUC, indicating the strongest
overall balance between catching phishing sites (recall) and not over-flagging
legitimate ones (precision).

## Limitations
- 5 of the 30 original features (SSL certificate state, domain age, web traffic
  rank, Google index, page rank) normally require live external lookups (WHOIS,
  Alexa/traffic ranking, Google index API). To keep the app **zero-cost and
  offline-friendly** per the project constraints, the Streamlit app lets the user
  manually set these, defaulting to a neutral value.
- This is an educational prototype trained on a 2012-era static dataset; phishing
  tactics evolve continuously, so periodic retraining would be needed for any
  real-world use.

## Deliverables Checklist
- [x] Source code & pipeline script
- [x] Dataset + source citation
- [x] Trained model (`.pkl`)
- [x] `requirements.txt`
- [x] README (this file)
- [x] Streamlit application
- [ ] Technical paper (see `Phishing_URL_Technical_Paper.docx`)
- [ ] Presentation slides
- [ ] Final demo + viva

## Credits
Dataset: Mohammad, R. & McCluskey, L. (2012). *Phishing Websites* [Dataset]. UCI
Machine Learning Repository. https://doi.org/10.24432/C51W2X
