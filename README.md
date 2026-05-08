# US Flight Delay Analysis — Software Packages Project

Analysis of US domestic flight performance in 2024 using Python.  
**Organisation:** US Airline Industry (BTS On-Time Performance Data)  
**Course:** Software Packages — 6th semester, CSIE Faculty

---

## Business question

> Which airlines and routes suffer the worst delays in 2024, and can we predict flight cancellations before they happen?

---

## Dataset

| | |
|---|---|
| Source | [US Bureau of Transportation Statistics](https://www.bts.gov/) |
| Period | Full year 2024 |
| Raw rows | ~7 million flights |
| Sample used | 100 000 flights (random) |
| Airport reference | OpenFlights — lat/lon for 7 000+ airports |

The `data/` folder is not tracked in git. Place `flight_data_2024.csv` and `Airports-Only.csv` inside it before running.

---

## Python facilities

| # | Facility | File |
|---|---|---|
| 1 | Streamlit — multi-page app | `app.py` |
| 2 | GeoPandas — airport map | `app.py` → page 3 |
| 3 | Missing & extreme values | `src/data_loader.py` |
| 4 | Encoding (OHE + label) | `src/features.py` |
| 5 | Scaling (StandardScaler) | `src/features.py` |
| 6 | pandas groupby & aggregation | `src/analysis.py` |
| 7 | merge / join | `src/analysis.py` |
| 8 | matplotlib charts | `src/visualizations.py` |
| 9 | scikit-learn KMeans clustering | `src/models.py` |
| 10 | scikit-learn Logistic Regression | `src/models.py` |
| 11 | statsmodels OLS regression | `src/models.py` |

---

## Project structure

```
├── app.py                  # Streamlit application (entry point)
├── src/
│   ├── data_loader.py      # Loading, sampling, missing values, outliers
│   ├── features.py         # Encoding and scaling
│   ├── analysis.py         # Aggregations and joins
│   ├── visualizations.py   # matplotlib figures
│   └── models.py           # KMeans, LogisticRegression, OLS
├── data/                   # (not tracked) raw CSV files
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

> GeoPandas can be tricky on Windows. If `pip` fails, use:
> `conda install -c conda-forge geopandas`

---

## Run the app

```bash
streamlit run app.py
```

---

## Key findings

- **Frontier (F9) and American (AA)** have the highest average arrival delays (> 11 min).
- **Alaska (AS) and Hawaiian (HA)** are the most punctual carriers.
- **July–August** are the worst months (avg dep delay > 18 min); October is the best.
- NAS (Air Traffic Control) and late arriving aircraft cause the majority of delay minutes.
- A logistic regression model achieves **AUC 0.999** for predicting cancellations using departure delay as the main signal.
