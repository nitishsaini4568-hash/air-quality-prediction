# Air Quality Prediction (India)

Machine learning project jo Indian cities ke AQI ko do tarah se predict karta hai:
1. **Current-day estimate:** aaj ke pollutant readings se aaj ka AQI
2. **Next-day forecast:** aaj ke pollutants aur AQI se kal ka AQI

## Dataset
[Air Quality Data in India (2015-2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) - `city_day.csv`, 26 cities.

## Approach
- Cleaning: duplicates check, Xylene (61% missing) drop, AQI_Bucket drop (leakage)
- Features: 11 pollutants, Month, DayOfWeek, City (one-hot); next-day task mein aaj ka AQI bhi
- Next-day target: har city ke andar agli date ka AQI (sirf consecutive din)
- **Time-based split** (80:20): train 2015-01-01 to 2019-12-06, test 2019-12-06 to 2020-07-01
- Imputer aur scaler pipeline ke andar (train data se hi fit)
- Models: Linear Regression, Random Forest, XGBoost (XGBoost deployed)

## Results (test set)
| Task | Model | MAE | RMSE | R2 |
|---|---|---|---|---|
| Current-day | XGBoost | 17.11 | 28.23 | 0.902 |
| Next-day | XGBoost | 18.12 | 34.66 | 0.852 |
| Next-day | Baseline (aaj = kal) | 20.88 | 43.11 | 0.771 |

Note: test period mein COVID-19 lockdown (2020) shamil hai, jisse pollution achanak gira.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `air_quality_project.ipynb` - poora notebook (EDA, training, evaluation)
- `app.py` - Streamlit web app
- `saved_models/` - trained models aur metadata
- `requirements.txt`

## Limitations
- Historical data par bana demo hai, live AQI service nahi
- Sirf 2015-2020 ka data; naye saalon mein pattern badal sakte hain
