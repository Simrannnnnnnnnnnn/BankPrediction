import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit Page Config
st.set_page_config(page_title="Bank Churn Prediction", layout="wide")

@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv("BankChurners.csv")
    return df

def preprocess_data(df, is_train=True, encoders=None):
    columns_to_drop = ['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                       'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',
                       'CLIENTNUM']
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
    df['Attrition_Flag'] = df['Attrition_Flag'].map({'Attrited Customer': 1, 'Existing Customer': 0})
    categorical_columns = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    
    if encoders is None:
        encoders = {col: LabelEncoder().fit(df[col]) for col in categorical_columns}
    
    for column in categorical_columns:
        if column in df.columns:
            df[column] = encoders[column].transform(df[column])
    
    return df, encoders

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

df = load_data()
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Visualization", "Explore Dataset", "Predict Churn"])

if page == "Home":
    st.title("Welcome to Bank Churn Prediction Web App")
    st.write("""
    - **Explore the Dataset**: Understand the statistics and structure of the data.
    - **Visualize Correlations**: Analyze relationships between variables using interactive charts.
    - **Predict Customer Churn**: Use a Machine Learning Model to predict whether a customer will churn or not.
    """)
    
    st.subheader("Train the Model")
    if st.button("Train Model"):
        df, encoders = preprocess_data(df, is_train=True)
        X = df.drop(columns=["Attrition_Flag"])
        y = df["Attrition_Flag"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)

        st.session_state['model'] = model
        st.session_state['features'] = X.columns.tolist()
        st.session_state['encoders'] = encoders
        y_pred = model.predict(X_test)
        
        st.write(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

elif page == "Predict Churn":
    st.title("Predict Customer Churn")
    
    if "model" not in st.session_state:
        st.warning("Please train the model first from the Home page.")
    else:
        input_data = {}
        
        for col in st.session_state['features']:
            if col in df.select_dtypes('object').columns:
                options = df[col].unique().tolist()
                input_data[col] = st.radio(f"Select {col}", options)
            else:
                input_data[col] = st.slider(f"Select {col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        
        input_df = pd.DataFrame([input_data])
        
        if st.button("Predict"):
            try:
                for col in st.session_state['features']:
                    if col in df.select_dtypes('number').columns:
                        input_df[col] = float(input_df[col])
                input_df, _ = preprocess_data(input_df, is_train=False, encoders=st.session_state['encoders'])
                prediction = st.session_state['model'].predict(input_df)[0]
                st.write(f"The customer is predicted to be: {'Churn' if prediction == 1 else 'Not Churn'}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

elif page == "Visualization":
    st.title("Data Visualization")
    visualization_options = ["Correlation Heatmap", "Distribution Plots", "Box Plots", "Scatter Plots"]
    selected_visualization = st.sidebar.selectbox("Choose a visualization", visualization_options)
    
    if selected_visualization == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")
        numeric_columns = df.select_dtypes('number').columns.tolist()
        if "Attrition_Flag" in numeric_columns:
            numeric_columns.remove("Attrition_Flag")
        
        if numeric_columns:
            corr_matrix = df[numeric_columns].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
            plt.close()
        else:
            st.write("No numeric columns available for correlation matrix.")

elif page == "Explore Dataset":
    st.title("Explore Dataset")
    st.write(""" Here, you can explore the dataset and view its statistics. """)
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.subheader("Dataset Statistics")
    st.write(df.describe())
    st.subheader("Missing Values")
    st.write(df.isnull().sum())
