import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Page Config
st.set_page_config(page_title="Stock Price Prediction", page_icon="📈", layout="wide")

st.title("📈 Stock Price Prediction App")
st.markdown("""
Upload historical stock data (CSV) to analyze trends and predict future prices.
""")

# Sidebar
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload Stock CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head())

        # Check required columns
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
        else:
            # Convert Date
            df['Date'] = pd.to_datetime(df['Date'])
            df.sort_values('Date', inplace=True)

            # Line Chart
            st.subheader("📉 Closing Price Trend")
            st.line_chart(df.set_index('Date')['Close'])

            # Feature Engineering
            df['Day'] = df['Date'].dt.day
            df['Month'] = df['Date'].dt.month
            df['Year'] = df['Date'].dt.year

            X = df[['Open', 'High', 'Low', 'Volume', 'Day', 'Month', 'Year']]
            y = df['Close']

            # Handle missing Volume
            if 'Volume' not in df.columns:
                X = df[['Open', 'High', 'Low', 'Day', 'Month', 'Year']]

            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Train Model
            model = LinearRegression()
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            # Metrics
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            st.subheader("📈 Model Performance")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Mean Squared Error", round(mse, 2))
            with col2:
                st.metric("R2 Score", round(r2, 2))

            # Actual vs Predicted Graph
            st.subheader("📊 Actual vs Predicted Prices")

            fig, ax = plt.subplots()
            ax.scatter(y_test, predictions)
            ax.set_xlabel("Actual Price")
            ax.set_ylabel("Predicted Price")
            st.pyplot(fig)

            # Prediction Section
            st.divider()
            st.subheader("🔮 Predict Next Day Price")

            open_price = st.number_input("Open Price")
            high_price = st.number_input("High Price")
            low_price = st.number_input("Low Price")
            volume = st.number_input("Volume")
            day = st.number_input("Day (1-31)", 1, 31)
            month = st.number_input("Month (1-12)", 1, 12)
            year = st.number_input("Year", 2000, 2100)

            if st.button("Predict Price"):
                input_data = np.array([[open_price, high_price, low_price,
                                        volume, day, month, year]])
                predicted_price = model.predict(input_data)
                st.success(f"Predicted Closing Price: ₹ {round(predicted_price[0],2)}")

    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    st.info("👈 Upload a stock dataset to begin.")