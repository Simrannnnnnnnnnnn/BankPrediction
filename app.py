import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Bank Churn Prediction Analysis")
st.write("Explore the dataset, visualize correlations and, predict customer churn interactively!")

# Caching dataset
@st.cache_data
def load_data():
    df = pd.read_csv("BankChurners.csv")
    return df

# Label encoding and preprocessing
def preprocess_data(df, is_train=True, encoders=None):
    columns_to_drop = [
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',
        'CLIENTNUM'
    ]
    if is_train:
        df.drop(columns=columns_to_drop, inplace=True)

    #label_encoder = LabelEncoder()
    df['Attrition_Flag'] = df['Attrition_Flag'].map({'Attrited Customer' : 1, 'Existing Customer':0})
    
    categorical_columns = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']

    if encoders is None:
        encoders = {col: LabelEncoder().fit(df[col]) for col in categorical_columns}

    for column in categorical_columns:
        if column in df.columns:
            df[column] = encoders[column].transform(df[column])

    return df, encoders

# Model training function
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Load data
df = load_data()
st.write(df.head())

# Sidebar options for filtering
st.sidebar.title("Customize Your Analysis")
selected_gender = st.sidebar.selectbox("Gender", options=["All"] + df['Gender'].unique().tolist())
selected_income = st.sidebar.selectbox("Income Category", options=["All"] + df['Income_Category'].unique().tolist())

if selected_gender != "All":
    df = df[df['Gender'] == selected_gender]

if selected_income != "All":
    df = df[df['Income_Category'] == selected_income]
st.write(df.shape)
# Heatmap
st.sidebar.title("Visualizations")
visualization_options =["Correlation Heatmap", "Distribution Plots","Box plots","Scatter Plots"]
selected_visualization = st.sidebar.radio("Choose a visualization", visualization_options)

if selected_visualization =="Correlation Heatmap":
    st.subheader("Correaltion Heatmap")
    selected_columns = st.multiselect("Select Columns for correlation Matrix", options = df.select_dtypes('number').columns)
    if selected_columns:
        corr_matrix = df[selected_columns].corr()
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(corr_matrix, annot = True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        plt.close()
elif selected_visualization == "Distribution Plots":
    st.subheader("Distribution Plots")
    numeric_columns = df.select_dtypes('number').columns
    selected_dist_column = st.sidebar.selectbox("Select column for Distribution Plot", options = numeric_columns)
    if selected_dist_column:
        fig,ax = plt.subplots(figsize = (8,6))
        sns.histplot(df[selected_dist_column], kde = True, color = "blue", ax=ax)
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
    x_axis = st.sidebar.selectbox("Select X-axis", numeric_columns)
    y_axis = st.sidebar.selectbox("Select Y-axis", numeric_columns)
    if x_axis and y_axis:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=x_axis, y=y_axis, data=df, hue="Attrition_Flag", palette="cool", ax=ax)
        ax.set_title(f"Scatter Plot: {x_axis} vs {y_axis}")
        st.pyplot(fig)
        plt.close()
# Model Training Secti
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
    st.text(classification_report(y_test, y_pred))

# Prediction Section
st.subheader("Predict Customer Churn")
if "model" in st.session_state:
    input_data = {col: st.text_input(col, "") for col in st.session_state['features']}
    input_df = pd.DataFrame([input_data])

    if st.button("Predict"):
        input_df, _ = preprocess_data(input_df, is_train=False, encoders=st.session_state['encoders'])
        prediction = st.session_state['model'].predict(input_df)[0]
        st.write(f"The customer is predicted to be: {'Churn' if prediction == 1 else 'Not Churn'}")
else:
    st.warning("Please train the model first.")
