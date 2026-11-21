# Air Quality Prediction - Streamlit App
import json
import os
from datetime import date

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Air Quality Prediction", page_icon="🌿",
                   layout="wide", initial_sidebar_state="expanded")

MODEL_DIR = "saved_models"

st.markdown("""
<style>
.hero-section {
    background: linear-gradient(120deg, #103C42, #087E75);
    padding: 36px; border-radius: 18px; color: white; margin-bottom: 24px;
}
.hero-section h1 { color: white !important; font-size: 36px; margin: 6px 0; }
.hero-section p { color: #D7F3EC; font-size: 16px; margin: 2px 0; }
div.stButton > button { border-radius: 10px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_assets():
    current_model = joblib.load(os.path.join(MODEL_DIR, "current_day_model.pkl"))
    next_model = joblib.load(os.path.join(MODEL_DIR, "next_day_model.pkl"))
    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        meta = json.load(f)
    return current_model, next_model, meta


try:
    current_model, next_model, meta = load_assets()
except Exception as e:
    st.error("Models load nahi ho paaye.")
    st.code(str(e))
    st.info("Pehle notebook run karo taaki `saved_models/` folder ban jaye "
            "(current_day_model.pkl, next_day_model.pkl, metadata.json).")
    st.stop()

CITIES = meta["cities"]
POLLUTANTS = meta["pollutants"]
DEFAULTS = meta["defaults"]
METRICS = meta["metrics"]


# ---------------- HELPERS ----------------
def get_category(aqi):
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


ADVICE = {
    "Good": "Air quality achhi hai. Bahar ki activities aaram se kar sakte ho.",
    "Satisfactory": "Sensitive logon ko halki saans ki takleef ho sakti hai.",
    "Moderate": "Sensitive log (bachche, bujurg, asthma) lambi outdoor activity kam karein.",
    "Poor": "Lambe samay tak bahar rehne se bachein, zaroorat ho to mask lagayein.",
    "Very Poor": "Bahar kam nikle aur health precautions follow karein.",
    "Severe": "Outdoor physical activity avoid karein aur local health advisory follow karein.",
}

COLORS = {"Good": "🟢", "Satisfactory": "🟡", "Moderate": "🟠",
          "Poor": "🔴", "Very Poor": "🟣", "Severe": "🟤"}


def build_row(features, values, city, the_date, today_aqi=None):
    """Model ke liye ek row banata hai. Missing pollutant = NaN (imputer sambhal lega)."""
    row = {f: 0.0 for f in features}
    for p in POLLUTANTS:
        if p in row:
            row[p] = values[p]
    row["Month"] = the_date.month
    row["DayOfWeek"] = the_date.weekday()
    city_col = "City_" + city
    if city_col in row:
        row[city_col] = 1.0
    if today_aqi is not None and "AQI" in row:
        row["AQI"] = today_aqi
    return pd.DataFrame([row], columns=features)


def metrics_table(task):
    return pd.DataFrame(METRICS[task]).T.reset_index().rename(columns={"index": "Model"})


# ---------------- SIDEBAR ----------------
st.sidebar.title("🌿 Air Quality Prediction")
st.sidebar.caption("Machine Learning Project")
page = st.sidebar.radio("Navigation", [
    "Dashboard", "Prediction", "Analytics", "Model Performance", "Health Recommendations"])
st.sidebar.divider()
st.sidebar.caption("Data: Air Quality Data in India (2015-2020)")


# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.markdown("""
    <div class="hero-section">
        <p>MACHINE LEARNING PROJECT</p>
        <h1>🌿 Air Quality Prediction</h1>
        <p>Aaj ka AQI estimate karo. Kal ka AQI forecast karo.</p>
    </div>
    """, unsafe_allow_html=True)

    best_cur = METRICS["current_day"]["XGBoost"]
    best_next = METRICS["next_day"]["XGBoost"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cities", len(CITIES))
    c2.metric("Current-day R²", f"{best_cur['R2']:.3f}")
    c3.metric("Next-day R²", f"{best_next['R2']:.3f}")
    c4.metric("Next-day MAE", f"{best_next['MAE']:.1f} AQI")

    st.divider()
    st.subheader("Project Overview")
    st.write("""
    Is project mein historical air pollution data (2015-2020, 26 Indian cities) se
    machine learning models train kiye gaye hain.

    - **Current-day estimate:** aaj ke pollutant readings se aaj ka AQI
    - **Next-day forecast:** aaj ke pollutants aur AQI se kal ka AQI
    - Split time ke hisaab se kiya gaya hai (purana data train, naya data test), isliye results honest hain
    """)
    st.warning("Ye historical data par bana ML demonstration hai, live AQI monitoring service nahi.")


# ---------------- PREDICTION ----------------
elif page == "Prediction":
    st.title("🌍 Air Quality Prediction")

    c1, c2 = st.columns(2)
    city = c1.selectbox("City", CITIES, index=CITIES.index("Delhi") if "Delhi" in CITIES else 0)
    the_date = c2.date_input("Date", date.today())

    mode = st.radio("Kya chahiye?", ["Current-day AQI estimate", "Next-day AQI forecast", "Dono"],
                    horizontal=True)

    st.subheader("Pollution Measurements (µg/m³; CO mg/m³)")
    missing = st.multiselect("Jin pollutants ki value pata nahi hai, unhe yahan chuno",
                             POLLUTANTS,
                             help="Inki value model khud estimate kar lega (training jaisa).")

    values = {}
    cols = st.columns(3)
    for i, p in enumerate(POLLUTANTS):
        with cols[i % 3]:
            if p in missing:
                st.number_input(p, value=float(DEFAULTS[p]), disabled=True, key="in_" + p)
                values[p] = np.nan
            else:
                values[p] = st.number_input(p, min_value=0.0, value=float(DEFAULTS[p]),
                                            step=1.0, key="in_" + p)

    known_today = False
    today_aqi = None
    if mode != "Current-day AQI estimate":
        known_today = st.checkbox("Mujhe aaj ka asli AQI pata hai")
        if known_today:
            today_aqi = st.number_input("Aaj ka AQI", min_value=0.0, max_value=500.0,
                                        value=float(DEFAULTS["AQI"]), step=1.0)

    if st.button("Predict Air Quality", type="primary"):
        try:
            cur_row = build_row(meta["current_features"], values, city, the_date)
            current_aqi = max(0.0, float(current_model.predict(cur_row)[0]))

            results = {}
            if mode in ("Current-day AQI estimate", "Dono"):
                results["Current Day"] = current_aqi

            if mode in ("Next-day AQI forecast", "Dono"):
                base_aqi = today_aqi if known_today else current_aqi
                nxt_row = build_row(meta["next_features"], values, city, the_date, base_aqi)
                results["Next Day"] = max(0.0, float(next_model.predict(nxt_row)[0]))

            st.success("Prediction ho gayi!")
            cols = st.columns(len(results))
            for col, (label, val) in zip(cols, results.items()):
                cat = get_category(val)
                col.metric(f"{label} AQI", f"{val:.0f}")
                col.subheader(f"{COLORS[cat]} {cat}")
                col.caption(ADVICE[cat])

            if "Next Day" in results and not known_today:
                st.info("Next-day forecast mein aaj ka AQI model ne khud estimate kiya hai. "
                        "Asli AQI pata ho to upar tick karke daalo, forecast behtar hoga.")

            mae_cur = METRICS["current_day"]["XGBoost"]["MAE"]
            mae_nxt = METRICS["next_day"]["XGBoost"]["MAE"]
            st.caption(f"Model ki typical galti: current-day ±{mae_cur:.0f} AQI, "
                       f"next-day ±{mae_nxt:.0f} AQI (test data par MAE).")

            if len(results) > 1:
                chart_df = pd.DataFrame({"Day": list(results.keys()), "AQI": list(results.values())})
                fig = px.bar(chart_df, x="Day", y="AQI", color="Day", title="Predicted AQI")
                st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Prediction error: {e}")


# ---------------- ANALYTICS ----------------
elif page == "Analytics":
    st.title("📊 Air Quality Analytics")

    df = None
    if os.path.exists("city_day.csv"):
        df = pd.read_csv("city_day.csv")
    else:
        up = st.file_uploader("city_day.csv upload karo", type=["csv"])
        if up is not None:
            df = pd.read_csv(up)

    if df is None:
        st.info("Analytics dekhne ke liye city_day.csv upload karo.")
    else:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))

        st.subheader("AQI Distribution")
        st.plotly_chart(px.histogram(df.dropna(subset=["AQI"]), x="AQI", nbins=40))

        st.subheader("Top 15 Cities by Average AQI")
        city_avg = (df.groupby("City")["AQI"].mean().reset_index()
                    .sort_values("AQI", ascending=False).head(15))
        st.plotly_chart(px.bar(city_avg, x="City", y="AQI"))

        st.subheader("Month-wise Average AQI")
        month_avg = df.dropna(subset=["AQI", "Date"]).groupby(df["Date"].dt.month)["AQI"].mean().reset_index()
        month_avg.columns = ["Month", "AQI"]
        st.plotly_chart(px.bar(month_avg, x="Month", y="AQI"))

        st.subheader("AQI Trend for a City")
        sel = st.selectbox("City chuno", sorted(df["City"].dropna().unique()))
        trend = df[(df["City"] == sel)].dropna(subset=["Date", "AQI"]).sort_values("Date")
        st.plotly_chart(px.line(trend, x="Date", y="AQI", title=f"{sel}: AQI over time"))


# ---------------- MODEL PERFORMANCE ----------------
elif page == "Model Performance":
    st.title("📈 Model Performance")
    st.write(f"Train: {meta['train_period']}  |  Test: {meta['test_period']} "
             "(time ke hisaab se split, random nahi)")

    for task, title in [("current_day", "Current-Day AQI Estimate"),
                        ("next_day", "Next-Day AQI Forecast")]:
        st.subheader(title)
        tbl = metrics_table(task)
        st.dataframe(tbl.round(3), hide_index=True)
        st.plotly_chart(px.bar(tbl, x="Model", y="R2", color="Model", title=f"{title}: R² score"))

    st.info("Baseline (aaj = kal) wo simple tareeka hai jisme aaj ka AQI hi kal ka AQI maan liya jaata hai. "
            "Model ko isse behtar hona chahiye.")
    st.caption("Test period (Jan-Jul 2020) mein COVID lockdown tha, jisse pollution achanak gira. "
               "Isliye test thoda mushkil hai.")
    st.write("Deployed models:", type(current_model.named_steps['model']).__name__,
             "(current) /", type(next_model.named_steps['model']).__name__, "(next-day)")


# ---------------- HEALTH RECOMMENDATIONS ----------------
elif page == "Health Recommendations":
    st.title("💚 Health Recommendations")
    ranges = [("Good", "0-50"), ("Satisfactory", "51-100"), ("Moderate", "101-200"),
              ("Poor", "201-300"), ("Very Poor", "301-400"), ("Severe", "401+")]
    for name, rng in ranges:
        with st.expander(f"{COLORS[name]} {name} AQI ({rng})"):
            st.write(ADVICE[name])
    st.warning("Ye general precautions hain, personal medical advice nahi.")
