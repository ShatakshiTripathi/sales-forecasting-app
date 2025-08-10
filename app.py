import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt
import pmdarima as pm

sns.set()

st.title("📊 Sales Forecasting Engine")

uploaded_file = st.file_uploader("Upload your sales CSV with 'Date' and 'Sales' columns", type=["csv"])

if uploaded_file is not None:
    try:
        # Try reading with utf-8 encoding first
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        # If utf-8 fails, try latin1 encoding
        uploaded_file.seek(0)  # Reset file pointer to start
        df = pd.read_csv(uploaded_file, encoding='latin1')

    # Select date and sales columns
    date_col = st.selectbox("Select the Date column", df.columns)
    sales_col = st.selectbox("Select the Sales column", df.columns)

    # --- Data Cleaning ---

    # 1. Convert date column to datetime, coercing errors
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=False)

    # 2. Drop rows with invalid dates
    df = df.dropna(subset=[date_col])

    # 3. Convert sales column to numeric, coercing errors to NaN
    df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce')

    # 4. Forward-fill missing sales values
    df[sales_col] = df[sales_col].fillna(method='ffill')

    # 5. Aggregate duplicates by date (sum sales)
    df = df.groupby(date_col)[sales_col].sum().reset_index()

    # 6. Sort by date and set index
    df = df.sort_values(date_col)
    df.set_index(date_col, inplace=True)

    # --- End cleaning ---

    # Convert to monthly frequency (start of month)
    sales_ts = df[sales_col].asfreq('MS')

    st.subheader("Sales Time Series")
    st.line_chart(sales_ts)

    if sales_ts.isnull().sum() > 0:
        sales_ts = sales_ts.fillna(method='ffill')
        st.warning("Missing sales data found and forward-filled.")

    st.subheader("Seasonal Decomposition")
    decomposition = seasonal_decompose(sales_ts, model='multiplicative')
    fig = decomposition.plot()
    st.pyplot(fig)

    train = sales_ts.iloc[:-12]
    test = sales_ts.iloc[-12:]

    st.write(f"Train data points: {len(train)} months")
    st.write(f"Test data points: {len(test)} months")

    with st.spinner("Finding best SARIMA model parameters..."):
        auto_model = pm.auto_arima(train,
                                   seasonal=True,
                                   m=12,
                                   suppress_warnings=True,
                                   stepwise=True)

    st.subheader("SARIMA Model Summary")
    st.text(auto_model.summary())

    model = SARIMAX(train,
                    order=auto_model.order,
                    seasonal_order=auto_model.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    model_fit = model.fit(disp=False)

    forecast = model_fit.get_forecast(steps=12)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    rmse = sqrt(mean_squared_error(test, forecast_mean))
    st.write(f"Test RMSE: {rmse:.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Test')
    plt.plot(test.index, forecast_mean, label='Forecast')
    plt.fill_between(test.index,
                     conf_int.iloc[:, 0],
                     conf_int.iloc[:, 1],
                     color='pink', alpha=0.3)
    plt.legend()
    plt.title(f'Sales Forecast (RMSE: {rmse:.2f})')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    st.pyplot(plt)

else:
    st.info("Please upload a CSV file with 'Date' and 'Sales' columns to start.")
