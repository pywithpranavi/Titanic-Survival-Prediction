import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Page Config
st.set_page_config(page_title="Stock Analytics Dashboard",
                   page_icon="📈",
                   layout="wide")

st.title("📈 Stock Price Analytics Dashboard")
st.markdown("AI-Powered Stock Trend & Price Prediction System")

# Sidebar
st.sidebar.header("📁 Upload Stock Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        st.error("Dataset must contain Date, Open, High, Low, Close columns.")
        st.stop()

    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

    # KPI Metrics Row
    st.subheader("📊 Key Market Indicators")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Latest Close Price", f"₹ {round(df['Close'].iloc[-1],2)}")
    col2.metric("Highest Price", f"₹ {round(df['High'].max(),2)}")
    col3.metric("Lowest Price", f"₹ {round(df['Low'].min(),2)}")
    col4.metric("Total Records", len(df))

    # Main Trend Chart
    st.subheader("📉 Closing Price Trend")
    st.line_chart(df.set_index('Date')['Close'])

    # Feature Engineering
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    features = ['Open', 'High', 'Low', 'Day', 'Month', 'Year']
    if 'Volume' in df.columns:
        features.insert(3, 'Volume')

    X = df[features]
    y = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Model Performance Section
    st.subheader("📈 Model Performance")
    col5, col6 = st.columns(2)

    col5.metric("Mean Squared Error", round(mse, 2))
    col6.metric("R² Score", round(r2, 2))

    # Actual vs Predicted Chart
    st.subheader("📊 Actual vs Predicted Prices")
    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions)
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    st.pyplot(fig)

    # Divider
    st.divider()

    # Prediction Panel
    st.subheader("🔮 Predict Next Day Closing Price")

    col7, col8, col9 = st.columns(3)

    open_price = col7.number_input("Open Price")
    high_price = col8.number_input("High Price")
    low_price = col9.number_input("Low Price")

    col10, col11, col12 = st.columns(3)

    volume = col10.number_input("Volume") if 'Volume' in df.columns else 0
    day = col11.number_input("Day", 1, 31)
    month = col12.number_input("Month", 1, 12)
    year = st.number_input("Year", 2000, 2100)

    if st.button("🚀 Predict Price"):
        if 'Volume' in df.columns:
            input_data = np.array([[open_price, high_price, low_price,
                                    volume, day, month, year]])
        else:
            input_data = np.array([[open_price, high_price, low_price,
                                    day, month, year]])

        predicted_price = model.predict(input_data)
        st.success(f"Predicted Closing Price: ₹ {round(predicted_price[0],2)}")

else:
    st.info("Upload a stock dataset to begin analysis.")