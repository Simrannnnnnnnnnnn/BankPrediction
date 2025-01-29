import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Bank Churn Prediction", layout="wide")

@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv("BankChurners.csv")
    return df

# Preprocessing Function
def preprocess_data(df, is_train=True, encoders=None):
    columns_to_drop = [
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',
        'CLIENTNUM'
    ]
    if is_train:
        df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    df['Attrition_Flag'] = df['Attrition_Flag'].map({'Attrited Customer': 1, 'Existing Customer': 0})

    categorical_columns = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']

    if encoders is None:
        encoders = {col: LabelEncoder().fit(df[col]) for col in categorical_columns}

    for column in categorical_columns:
        if column in df.columns:
            df[column] = encoders[column].transform(df[column])

    return df, encoders

# Model Training Function
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Load Data
df = load_data()
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Visualization", "Explore Dataset", "Predict Churn"])

# Home Page
if page == "Home":
    st.title("Welcome to Bank Churn Prediction Web App")
    st.write("""
    This Web Application allows you to:
    - **Explore the Dataset**: Understand the statistics and structure of the data.
    - **Visualize Correlations**: Analyze relationships between variables using interactive charts.
    - **Predict Customer Churn**: Use a machine learning model to predict customer churn.
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
if page == "Visualization":
    st.title("Data Visualization")
    st.write("Select a visualization type from the sidebar to explore relationships and patterns in the dataset.")

    visualization_options = ["Correlation Heatmap", "Distribution Plots", "Box Plots", "Scatter Plots"]
    selected_visualization = st.sidebar.selectbox("Choose a visualization", visualization_options)

    if selected_visualization == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")
        selected_columns = st.multiselect("Select columns for correlation matrix", options=df.select_dtypes('number').columns)
        if selected_columns:
            corr_matrix = df[selected_columns].corr()
            fig, ax = plt.subplots(figsize=(10,8))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
            plt.close()

    elif selected_visualization == "Distribution Plots":
        st.subheader("Distribution Plots")
        numeric_columns = df.select_dtypes('number').columns
        selected_dist_column = st.selectbox("Select Column for Distribution Plot", options=numeric_columns)
        if selected_dist_column:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(df[selected_dist_column], kde=True, color="blue", ax=ax)
            ax.set_title(f"Distribution of {selected_dist_column}")
            st.pyplot(fig)
            plt.close()

    elif selected_visualization == "Box Plots":
        st.subheader("Box Plots")
        numeric_columns = df.select_dtypes('number').columns
        categorical_columns = df.select_dtypes('object').columns
        selected_box_column = st.selectbox("Select Numeric Column for Box Plot", options=numeric_columns)
        selected_cat_column = st.selectbox("Select Categorical Column for Box Plot", options=categorical_columns)
        if selected_box_column and selected_cat_column:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x=selected_cat_column, y=selected_box_column, data=df, ax=ax)
            ax.set_title(f"Box Plot of {selected_box_column} by {selected_cat_column}")
            st.pyplot(fig)
            plt.close()

    elif selected_visualization == "Scatter Plots":
        st.subheader("Scatter Plots")
        numeric_columns = df.select_dtypes('number').columns
        x_axis = st.selectbox("Select X-axis", numeric_columns)
        y_axis = st.selectbox("Select Y-axis", numeric_columns)
        if x_axis and y_axis:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x=x_axis, y=y_axis, data=df, hue="Attrition_Flag", palette="cool", ax=ax)
            ax.set_title(f"Scatter Plot: {x_axis} vs {y_axis}")
            st.pyplot(fig)
            plt.close()

# Explore Dataset Page
if page == "Explore Dataset":
    st.title("Explore Dataset")
    st.subheader("Dataset Overview")
    st.write(df.head())

    st.subheader("Dataset Statistics")
    st.write(df.describe())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

# Predict Churn Page
# Predict Churn Page
if page == "Predict Churn":
    st.title("Predict Customer Churn")
    st.write("Use the trained model to predict whether a customer will churn or not.")

    if "model" not in st.session_state:
        st.warning("Please train the model first from the Home page.")
    else:
        input_data = {}

        # Separate categorical and numerical columns
        categorical_columns = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
        numeric_columns = df.select_dtypes('number').columns.tolist()
        numeric_columns.remove("Attrition_Flag")  # Remove target column

        # Get unique values for categorical columns from the dataset
        for col in categorical_columns:
            options = df[col].unique().tolist()
            input_data[col] = st.radio(f"Select {col}", options)

        # Get sliders for numeric columns
        for col in numeric_columns:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            avg_val = float(df[col].median())  # Default to median value
            input_data[col] = st.slider(f"Select {col}", min_val, max_val, avg_val)

        # Convert input data into a DataFrame
        input_df = pd.DataFrame([input_data])

        if st.button("Predict"):
            try:
                # Encode categorical values using stored encoders
                for col in categorical_columns:
                    encoder = st.session_state['encoders'][col]
                    input_df[col] = encoder.transform([input_df[col][0]])

                # Ensure numeric values are correctly formatted
                input_df = input_df.astype(float)

                # Predict churn
                prediction = st.session_state['model'].predict(input_df)[0]
                st.write(f"### The customer is predicted to be: **{'Churn' if prediction == 1 else 'Not Churn'}**")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
