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

# Caching dataset
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    df = pd.read_csv("BankChurners.csv")
    return df

# Preprocessing function
def preprocess_data(df, is_train=True, encoders=None):
    columns_to_drop = [
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',
        'CLIENTNUM'
    ]
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
    
    # Encode target variable explicitly
    if 'Attrition_Flag' in df.columns:
        df['Attrition_Flag'] = df['Attrition_Flag'].map({'Attrited Customer': 1, 'Existing Customer': 0})
    
    # Define categorical columns
    categorical_columns = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    
    if encoders is None:
        encoders = {col: LabelEncoder().fit(df[col]) for col in categorical_columns}
    
    for column in categorical_columns:
        if column in df.columns:
            df[column] = encoders[column].transform(df[column])
    
    # Drop 'Attrition_Flag' during prediction
    if not is_train and 'Attrition_Flag' in df.columns:
        df.drop(columns=['Attrition_Flag'], inplace=True)
    
    return df, encoders

# Model training function
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Load data
df = load_data()

# Navigation Bar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Visualization", "Explore Dataset", "Predict Churn"])

# Home Page
if page == "Home":
    st.title("Welcome to Bank Churn Prediction Web App")
    st.write("""
    This web application allows you to:
    - **Explore the dataset**: Understand the statistics and structure of the data.
    - **Visualize correlations**: Analyze relationships between variables using interactive charts.
    - **Predict customer churn**: Use a machine learning model to predict whether a customer will churn or not.
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

# Visualization Page
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

    elif selected_visualization == "Distribution Plots":
        st.subheader("Distribution Plots")
        st.write("""
        A distribution plot shows the distribution of a numeric variable. 
        It helps you understand the spread, skewness, and outliers in the data.
        """)
        
        # Select numeric column for distribution plot
        numeric_columns = df.select_dtypes('number').columns.tolist()
        selected_column = st.selectbox("Select a numeric column for distribution plot", numeric_columns)
        
        if selected_column:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(df[selected_column], kde=True, color="blue", ax=ax)
            ax.set_title(f"Distribution of {selected_column}")
            st.pyplot(fig)
            plt.close()

    elif selected_visualization == "Box Plots":
        st.subheader("Box Plots")
        st.write("""
        A box plot shows the distribution of a numeric variable across different categories. 
        It helps you compare the spread and identify outliers.
        """)
        
        # Select numeric and categorical columns for box plot
        numeric_columns = df.select_dtypes('number').columns.tolist()
        categorical_columns = df.select_dtypes('object').columns.tolist()
        
        selected_numeric = st.selectbox("Select a numeric column for box plot", numeric_columns)
        selected_categorical = st.selectbox("Select a categorical column for box plot", categorical_columns)
        
        if selected_numeric and selected_categorical:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x=selected_categorical, y=selected_numeric, data=df, ax=ax)
            ax.set_title(f"Box Plot of {selected_numeric} by {selected_categorical}")
            st.pyplot(fig)
            plt.close()

    elif selected_visualization == "Scatter Plots":
        st.subheader("Scatter Plots")
        st.write("""
        A scatter plot shows the relationship between two numeric variables. 
        It helps you identify patterns, trends, and outliers.
        """)
        
        # Select numeric columns for scatter plot
        numeric_columns = df.select_dtypes('number').columns.tolist()
        x_axis = st.selectbox("Select X-axis", numeric_columns)
        y_axis = st.selectbox("Select Y-axis", numeric_columns)
        
        if x_axis and y_axis:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x=x_axis, y=y_axis, data=df, hue="Attrition_Flag", palette="cool", ax=ax)
            ax.set_title(f"Scatter Plot: {x_axis} vs {y_axis}")
            st.pyplot(fig)
            plt.close()

# Explore Dataset Page
elif page == "Explore Dataset":
    st.title("Explore Dataset")
    st.write("Here, you can explore the dataset and view its statistics.")
    
    st.subheader("Dataset Overview")
    st.write(df.head())
    
    st.subheader("Dataset Statistics")
    st.write(df.describe())
    
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

# Predict Churn Page
elif page == "Predict Churn":
    st.title("Predict Customer Churn")
    
    if "model" not in st.session_state:
        st.warning("Please train the model first from the Home page.")
    else:
        input_data = {}
        
        for col in st.session_state['features']:
            if col in df.select_dtypes('object').columns:
                # Use unique values from the training data for categorical columns
                options = df[col].unique().tolist()
                input_data[col] = st.selectbox(f"Select {col}", options)
            else:
                # Use sliders for numeric columns
                input_data[col] = st.slider(f"Select {col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        
        input_df = pd.DataFrame([input_data])
        
        if st.button("Predict"):
            try:
                # Convert input data to correct types
                for col in st.session_state['features']:
                    if col in df.select_dtypes('number').columns:
                        input_df[col] = float(input_df[col])
                
                # Preprocess input data
                input_df, _ = preprocess_data(input_df, is_train=False, encoders=st.session_state['encoders'])
                
                # Ensure all required columns are present
                missing_cols = set(st.session_state['features']) - set(input_df.columns)
                if missing_cols:
                    st.error(f"Missing columns in input data: {missing_cols}")
                else:
                    # Make prediction
                    prediction = st.session_state['model'].predict(input_df)[0]
                    st.write(f"The customer is predicted to be: {'Churn' if prediction == 1 else 'Not Churn'}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
