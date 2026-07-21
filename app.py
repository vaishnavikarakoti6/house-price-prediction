import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import joblib
import time

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="HouseAI Dashboard",
    layout="wide",
    page_icon="🏡"
)

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/AmesHousing.csv")

@st.cache_resource
def load_model():
    return pickle.load(open("models/best_model.pkl", "rb"))

@st.cache_resource
def load_results():
    return joblib.load("models/model_results.pkl")

@st.cache_resource
def load_template():
    return joblib.load("models/X_template.pkl")


df = load_data()
model = load_model()
results = load_results()
template = load_template()

# ------------------ CSS ------------------
st.markdown("""
<style>

.main {
    background: #f7f9fc;
    font-family: 'Segoe UI', sans-serif;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#eef2ff,#f0fdfa);
}

.metric-card {
    padding:22px;
    border-radius:16px;
    color:#1f2937;
    background: linear-gradient(135deg,#e0f2fe,#ede9fe);
    text-align:center;
    box-shadow:0px 6px 20px rgba(0,0,0,0.05);
    transition: all 0.25s ease;
}

.metric-card:hover {
    transform: translateY(-6px);
}

.section-title {
    font-size:22px;
    font-weight:600;
    margin:20px 0 10px 0;
    color:#111827;
}

.stButton>button {
    background: linear-gradient(135deg,#93c5fd,#c4b5fd);
    color:#111827;
    border:none;
    padding:10px 18px;
    border-radius:12px;
    font-weight:600;
}

h1 {
    font-weight:700;
    color:#111827;
}

</style>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.title("🏡 HouseAI")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Analytics", "Model Performance", "Predict"]
)

# ================= DASHBOARD =================
if page == "Dashboard":

    st.title("🏡 House Price Intelligence")

    col1, col2, col3 = st.columns(3)

    col1.markdown(
        f'<div class="metric-card"><h3>Total Houses</h3><h1>{df.shape[0]}</h1></div>',
        unsafe_allow_html=True
    )

    col2.markdown(
        f'<div class="metric-card"><h3>Features</h3><h1>{df.shape[1]}</h1></div>',
        unsafe_allow_html=True
    )

    col3.markdown(
        f'<div class="metric-card"><h3>Average Price</h3><h1>${int(df["SalePrice"].mean())}</h1></div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------- Price Distribution --------
    st.markdown('<div class="section-title">📊 Price Distribution</div>', unsafe_allow_html=True)

    fig = px.histogram(df, x="SalePrice", color_discrete_sequence=["#6366f1"])
    st.plotly_chart(fig, use_container_width=True)

    # -------- Dataset Preview --------
    st.markdown('<div class="section-title">📄 Dataset Preview</div>', unsafe_allow_html=True)

    st.dataframe(df.head(), use_container_width=True)


# ================= ANALYTICS =================
elif page == "Analytics":

    st.title("📊 Data Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Sale Price Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x="SalePrice", color_discrete_sequence=["#7c3aed"])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Living Area vs Price</div>', unsafe_allow_html=True)
        fig = px.scatter(df, x="Gr Liv Area", y="SalePrice", color="Overall Qual")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)

    corr = df.corr(numeric_only=True)
    fig = px.imshow(corr, height=700)
    st.plotly_chart(fig, use_container_width=True)


# ================= MODEL PERFORMANCE =================
elif page == "Model Performance":

    st.title("🤖 Model Performance")

    df_results = pd.DataFrame(results).T
    best_model = df_results["RMSE"].idxmin()

    col1, col2, col3 = st.columns(3)

    col1.metric("Best Model", best_model)
    col2.metric("Lowest RMSE", round(df_results["RMSE"].min(), 4))
    col3.metric("Best R²", round(df_results["R2"].max(), 4))

    st.markdown('<div class="section-title">RMSE Comparison</div>', unsafe_allow_html=True)

    fig = px.bar(df_results, y="RMSE", color=df_results.index)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Model Metrics</div>', unsafe_allow_html=True)

    st.dataframe(df_results, use_container_width=True)


# ================= PREDICT =================
elif page == "Predict":

    st.title("🔮 Predict House Price")

    st.markdown('<div class="section-title">Enter Property Details</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        area = st.number_input("Living Area (sqft)", 100, 5000, 1500)
        quality = st.number_input("Overall Quality", 1, 10, 5)

    with col2:
        year = st.number_input("Year Built", 1900, 2023, 2000)
        rooms = st.number_input("Rooms", 1, 12, 6)

    with col3:
        garage = st.number_input("Garage Cars", 0, 4, 2)

    if st.button("Predict Price"):

            input_df = template.copy()

            input_df["Gr Liv Area"] = area
            input_df["Overall Qual"] = quality
            input_df["Year Built"] = year
            input_df["TotRms AbvGrd"] = rooms
            input_df["Garage Cars"] = garage

            with st.spinner("Predicting..."):
                time.sleep(1)
                prediction = model.predict(input_df)
                price = np.expm1(prediction)[0]

            usd_to_inr = 83
            price_inr = price * usd_to_inr

            st.markdown(f"""
            ### 💰 Estimated Price
            - USD: **${price:,.0f}**
            - INR: **₹{price_inr:,.0f}**
            """)