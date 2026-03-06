import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Page Configuration
st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢", layout="wide")

st.title("🚢 Titanic Survival Classification")
st.markdown("""
This application predicts passenger survival using selected demographic information.
Upload your Titanic dataset file to get started!
""")

# Sidebar Upload
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload Titanic CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Load dataset
        df = pd.read_csv(uploaded_file)

        # Remove unwanted columns
        df = df.drop(['SibSp', 'Parch', 'Fare', 'Ticket'], axis=1, errors='ignore')

        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        # ======================
        # SEARCH SECTION
        # ======================
        st.subheader("🔍 Search Passengers")

        col1, col2 = st.columns(2)
        with col1:
            search_name = st.text_input("Search by Name:")
        with col2:
            search_passenger_id = st.text_input("Search by Passenger ID:")

        if search_name and 'Name' in df.columns:
            results = df[df['Name'].str.contains(search_name, case=False, na=False)]
            st.dataframe(results)

        if search_passenger_id and 'PassengerId' in df.columns:
            try:
                pid = int(search_passenger_id)
                result = df[df['PassengerId'] == pid]
                st.dataframe(result)
            except:
                st.error("Enter valid Passenger ID")

        # ======================
        # PREPROCESSING
        # ======================
        if 'Survived' not in df.columns:
            st.error("Dataset must contain 'Survived' column")
        else:

            st.subheader("🔧 Data Preprocessing")

            def preprocess_data(df):
                df_processed = df.copy()

                # Fill missing Age
                if 'Age' in df_processed.columns:
                    df_processed['Age'] = df_processed['Age'].fillna(
                        df_processed['Age'].median()
                    )

                # Fill missing Embarked
                if 'Embarked' in df_processed.columns:
                    df_processed['Embarked'] = df_processed['Embarked'].fillna(
                        df_processed['Embarked'].mode()[0]
                    )

                # Drop unnecessary columns
                df_processed = df_processed.drop(
                    ['Name', 'Cabin', 'PassengerId'],
                    axis=1,
                    errors='ignore'
                )

                # Encode Sex
                if 'Sex' in df_processed.columns:
                    df_processed['Sex'] = df_processed['Sex'].map(
                        {'male': 0, 'female': 1}
                    )

                # One-hot encode Embarked
                if 'Embarked' in df_processed.columns:
                    df_processed = pd.get_dummies(
                        df_processed,
                        columns=['Embarked'],
                        drop_first=True
                    )

                return df_processed

            df_processed = preprocess_data(df)

            st.success("✅ Data preprocessing completed!")
            st.dataframe(df_processed.head(), use_container_width=True)

            # ======================
            # MODEL TRAINING
            # ======================
            X = df_processed.drop('Survived', axis=1)
            y = df_processed['Survived']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "KNN": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Naive Bayes": GaussianNB(),
                "SVM": SVC(probability=True),
                "Gradient Boosting": GradientBoostingClassifier()
            }

            st.subheader("⚙️ Model Training & Comparison")

            if st.button("🚀 Train Models"):

                results = []

                for name, model in models.items():
                    model.fit(X_train_scaled, y_train)
                    preds = model.predict(X_test_scaled)

                    accuracy = accuracy_score(y_test, preds)
                    precision = precision_score(y_test, preds)
                    recall = recall_score(y_test, preds)

                    results.append({
                        "Model": name,
                        "Accuracy (%)": round(accuracy * 100, 2),
                        "Precision (%)": round(precision * 100, 2),
                        "Recall (%)": round(recall * 100, 2)
                    })

                results_df = pd.DataFrame(results).sort_values(
                    by="Accuracy (%)", ascending=False
                )

                st.dataframe(results_df, use_container_width=True)
                st.bar_chart(results_df.set_index("Model")["Accuracy (%)"])

                best_model = results_df.iloc[0]
                best_model_name = best_model["Model"]
                best_model_obj = models[best_model_name]
                best_model_obj.fit(X_train_scaled, y_train)

                st.success(
                    f"🏆 Best Model: {best_model_name} "
                    f"with {best_model['Accuracy (%)']}% accuracy"
                )

                # ======================
                # CUSTOM PREDICTION
                # ======================
                st.subheader("🎯 Predict Survival for Custom Passenger")

                col1, col2 = st.columns(2)

                with col1:
                    pclass = st.selectbox("Passenger Class", [1, 2, 3])
                    sex = st.selectbox("Sex", ["male", "female"])

                with col2:
                    age = st.number_input("Age", min_value=0, max_value=100, value=25)
                    embarked = st.selectbox("Embarked", ["C", "Q", "S"])

                # Encode inputs
                sex = 0 if sex == "male" else 1
                embarked_C = 1 if embarked == "C" else 0
                embarked_Q = 1 if embarked == "Q" else 0

                input_data = np.array([[pclass, sex, age, embarked_C, embarked_Q]])
                input_scaled = scaler.transform(input_data)

                if st.button("🔮 Predict Survival"):

                    prediction = best_model_obj.predict(input_scaled)[0]
                    probability = best_model_obj.predict_proba(input_scaled)[0][1]

                    if prediction == 1:
                        st.success("🎉 Passenger Survived")
                    else:
                        st.error("❌ Passenger Did Not Survive")

                    st.info(f"📊 Survival Probability: {probability*100:.2f}%")
                    st.progress(int(probability * 100))

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

else:
    st.info("👈 Please upload your Titanic dataset file to begin.")