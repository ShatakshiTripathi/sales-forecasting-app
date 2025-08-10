# Sales Forecasting Engine with Mixed Date Format Support

This is a Streamlit app for sales forecasting using SARIMA models.  
It supports CSV files with mixed date formats and provides seasonal decomposition and forecasts.

---

## Features
- Upload CSV files with **Date** and **Sales** columns (supports mixed date formats)  
- Visualize sales time series and seasonal components  
- Automatic SARIMA model parameter selection  
- Forecast sales for next 12 months  
- Root Mean Squared Error (RMSE) calculation on test data  
- Interactive plots using Streamlit and Matplotlib  

---

## How to Run Locally

### Prerequisites
- Python 3.7 or later  
- Install dependencies:  
  ```bash
  pip install streamlit pandas matplotlib seaborn statsmodels scikit-learn pmdarima
Run the app
From the project directory, run:

bash
Copy
Edit
streamlit run app.py
Access the app
Open your browser and go to:
http://localhost:8501

Upload your sales CSV
Your CSV file should have at least two columns:

A Date column (dates can be in mixed formats)

A Sales column (numeric sales data)

The app will parse dates, clean data, and forecast sales.
