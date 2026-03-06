import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.impute import SimpleImputer

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
This application predicts passenger survival using demographic and ticket information from the Titanic dataset.
Upload your Titanic dataset file to get started!
""")

# Sidebar for file upload
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload Titanic CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Load the dataset from uploaded file
        df = pd.read_csv(uploaded_file)
        
        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head())
        
        # Add search functionality
        st.subheader("🔍 Search Passengers")
        
        col1, col2 = st.columns(2)
        with col1:
            search_name = st.text_input("Search by Name:")
        with col2:
            search_passenger_id = st.text_input("Search by Passenger ID:")
        
        # Search by name
        if search_name:
            search_results = df[df['Name'].str.contains(search_name, case=False, na=False)]
            
            if not search_results.empty:
                st.success(f"Found {len(search_results)} passenger(s) matching '{search_name}'")
                st.dataframe(search_results)
                
                # Show survival statistics for search results
                survived_count = search_results['Survived'].sum()
                total_count = len(search_results)
                survival_rate = (survived_count / total_count) * 100 if total_count > 0 else 0
                st.info(f"📊 Survival Rate: {survived_count}/{total_count} ({survival_rate:.1f}%)")
            else:
                st.warning(f"No passengers found matching '{search_name}'")
        
        # Search by Passenger ID
        if search_passenger_id:
            try:
                passenger_id = int(search_passenger_id)
                passenger_result = df[df['PassengerId'] == passenger_id]
                
                if not passenger_result.empty:
                    st.success(f"Found Passenger ID: {passenger_id}")
                    st.dataframe(passenger_result)
                    
                    # Show survival status
                    survived = passenger_result['Survived'].iloc[0]
                    if survived == 1:
                        st.success("✅ This passenger SURVIVED")
                    else:
                        st.error("❌ This passenger did NOT survive")
                else:
                    st.warning(f"No passenger found with ID: {passenger_id}")
            except ValueError:
                st.error("Please enter a valid Passenger ID (numbers only)")
        
        # Quick stats buttons
        st.subheader("📈 Quick Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Show All Survivors"):
                survivors = df[df['Survived'] == 1]
                st.success(f"Total Survivors: {len(survivors)}")
                st.dataframe(survivors)
        
        with col2:
            if st.button("Show All Non-Survivors"):
                non_survivors = df[df['Survived'] == 0]
                st.error(f"Total Non-Survivors: {len(non_survivors)}")
                st.dataframe(non_survivors)
        
        with col3:
            if st.button("Show by Class"):
                class_stats = df.groupby('Pclass')['Survived'].agg(['count', 'sum']).reset_index()
                class_stats.columns = ['Class', 'Total', 'Survived']
                class_stats['Survival Rate'] = (class_stats['Survived'] / class_stats['Total'] * 100).round(1)
                st.dataframe(class_stats)
        
        st.divider()
        
        # Show dataset info
        st.subheader("📋 Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
            st.write(f"**Columns:** {list(df.columns)}")
        with col2:
            st.write("**Missing Values:**")
            missing_data = df.isnull().sum()
            st.dataframe(missing_data[missing_data > 0])
        
        # Check if required columns exist
        required_columns = ['Survived']
        missing_required = [col for col in required_columns if col not in df.columns]
        
        # Check if dataset has sufficient features for training
        feature_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        available_features = [col for col in feature_columns if col in df.columns]
        
        if not missing_required:
            if len(available_features) < 3:
                st.error("⚠️ Insufficient features for model training!")
                st.info(f"""
                Your dataset only has {len(available_features)} feature columns: {available_features}
                
                **To fix this issue:**
                1. Use the `titanic_dataset.csv` file I created for you (recommended)
                2. Or download a complete Titanic dataset from Kaggle
                3. Your dataset must include at least 3 of these columns: {feature_columns}
                
                **Current dataset columns:** {list(df.columns)}
                """)
            else:
                # Data Preprocessing
                st.subheader("🔧 Handling Missing Values & Encoding Categorical Variables")
                
                # Function to preprocess data
                def preprocess_data(df):
                    # Make a copy to avoid modifying original
                    df_processed = df.copy()
                    
                    # Handle missing values
                    # Fill missing Age with median
                    if 'Age' in df_processed.columns:
                        df_processed['Age'].fillna(df_processed['Age'].median(), inplace=True)
                    
                    # Fill missing Embarked with mode
                    if 'Embarked' in df_processed.columns:
                        df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0], inplace=True)
                    
                    # Fill missing Fare with median
                    if 'Fare' in df_processed.columns:
                        df_processed['Fare'].fillna(df_processed['Fare'].median(), inplace=True)
                    
                    # Drop columns that won't be useful for prediction
                    columns_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']
                    df_processed = df_processed.drop([col for col in columns_to_drop if col in df_processed.columns], axis=1)
                    
                    # Convert categorical variables to numerical
                    le = LabelEncoder()
                    
                    # Encode Sex
                    if 'Sex' in df_processed.columns:
                        df_processed['Sex'] = le.fit_transform(df_processed['Sex'])
                    
                    # Encode Embarked
                    if 'Embarked' in df_processed.columns:
                        df_processed['Embarked'] = le.fit_transform(df_processed['Embarked'])
                    
                    return df_processed
            
            # Preprocess the data
            df_processed = preprocess_data(df)
            
            st.success("✅ Data preprocessing completed!")
            st.dataframe(df_processed.head())
            
            # Separate Features (X) and Target (y)
            X = df_processed.drop('Survived', axis=1)
            y = df_processed['Survived']
            
            # Train-Test Split (80% training, 20% testing)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Feature Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Define Models
            models = {
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Naive Bayes": GaussianNB(),
                "Support Vector Classifier": SVC(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42)
            }
            
            st.divider()
            st.subheader("⚙️ Model Training & Comparison")
            
            # Button to trigger training
            if st.button("🚀 Train Models & Compare Metrics"):
                with st.spinner('Training models... Please wait!'):
                    results = []
                    
                    # Train and Evaluate each model
                    for name, model in models.items():
                        model.fit(X_train_scaled, y_train)
                        predictions = model.predict(X_test_scaled)
                        
                        # Calculate Metrics
                        accuracy = accuracy_score(y_test, predictions)
                        precision = precision_score(y_test, predictions, zero_division=0)
                        recall = recall_score(y_test, predictions, zero_division=0)
                        
                        results.append({
                            "Model": name, 
                            "Accuracy (%)": round(accuracy * 100, 2),
                            "Precision (%)": round(precision * 100, 2),
                            "Recall (%)": round(recall * 100, 2)
                        })
                    
                    # Convert results to DataFrame and sort by Accuracy
                    results_df = pd.DataFrame(results).sort_values(by="Accuracy (%)", ascending=False)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("### 📈 Performance Metrics")
                        st.dataframe(results_df, use_container_width=True)
                    
                    with col2:
                        st.write("### 📊 Accuracy Comparison")
                        st.bar_chart(results_df.set_index("Model")['Accuracy (%)'])
                    
                    # Show best model
                    best_model = results_df.iloc[0]
                    st.success(f"🏆 Best Model: {best_model['Model']} with {best_model['Accuracy (%)']}% accuracy")
                    
                    # Feature importance for tree-based models
                    if st.checkbox("🔍 Show Feature Importance"):
                        tree_models = ["Random Forest", "Gradient Boosting", "Decision Tree"]
                        selected_model = st.selectbox("Select model for feature importance:", tree_models)
                        
                        if selected_model in models:
                            model = models[selected_model]
                            if hasattr(model, 'feature_importances_'):
                                importance_df = pd.DataFrame({
                                    'Feature': X.columns,
                                    'Importance': model.feature_importances_
                                }).sort_values('Importance', ascending=False)
                                
                                st.write(f"### Feature Importance - {selected_model}")
                                st.bar_chart(importance_df.set_index('Feature')['Importance'])
                
                st.success("✅ Models trained successfully!")
            
        else:
            st.error(f"⚠️ Required columns missing: {missing_required}")
            st.info("Please ensure your dataset contains a 'Survived' column with values 0 (did not survive) and 1 (survived).")
            
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure you're uploading a valid Titanic dataset CSV file.")
        
else:
    st.info("👈 Please upload your Titanic dataset file to begin analysis.")
    
    # Show expected format
    with st.expander("📋 Expected Dataset Format"):
        st.write("""
        Your dataset should contain the following columns:
        - **Survived**: Target variable (0 = No, 1 = Yes)
        - **Pclass**: Passenger class (1, 2, 3)
        - **Sex**: Gender (male, female)
        - **Age**: Age in years
        - **SibSp**: Number of siblings/spouses aboard
        - **Parch**: Number of parents/children aboard
        - **Fare**: Passenger fare
        - **Embarked**: Port of embarkation (C, Q, S)
        - **Name**: Passenger name (optional)
        - **Ticket**: Ticket number (optional)
        - **Cabin**: Cabin number (optional)
        - **PassengerId**: Unique identifier (optional)
        """)