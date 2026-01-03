!pip install streamlit prophet

import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt



st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

st.title("📊 AI-Powered Sales Forecasting Dashboard")
st.markdown("**Future Interns | Machine Learning Task 1**")

uploaded_file = st.file_uploader("Upload sales_data.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df['Sale_Date'] = pd.to_datetime(df['Sale_Date'])

    # Aggregate daily sales
    daily_sales = df.groupby('Sale_Date')['Sales_Amount'].sum().reset_index()
    daily_sales = daily_sales.rename(columns={
        'Sale_Date': 'ds',
        'Sales_Amount': 'y'
    })

    st.subheader("📁 Aggregated Daily Sales")
    st.dataframe(daily_sales.head())

    periods = st.slider("Forecast period (days)", 30, 365, 90)

    model = Prophet()
    model.fit(daily_sales)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    st.subheader("📈 Sales Forecast")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    st.subheader("📉 Trend & Seasonality")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    st.success("Forecast generated successfully!")

else:
    st.info("Upload the sales_data.csv file to generate forecasts.")
