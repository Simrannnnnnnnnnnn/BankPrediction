import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Preprocess the data
def preprocess_data(df, is_train=True):
    columns_to_drop = [
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',
        'CLIENTNUM'
    ]
    if is_train:
        df.drop(columns=columns_to_drop, inplace=True)
    label_encoder = LabelEncoder()
    categorical_columns = ['Attrition_Flag', 'Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    for column in categorical_columns:
        if column in df.columns:
            df[column] = label_encoder.fit_transform(df[column])
    return df

# Load data
def load_data():
    return pd.read_csv("C:/Users/kaurs/Downloads/Internship/4. Bank Churn Prediction Analysis/BankChurners.csv")

# Train model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Predict churn
def predict_churn(model, input_df, features):
    input_df = preprocess_data(input_df, is_train=False)
    if 'Attrition_Flag' in input_df.columns:
        input_df.drop(columns=['Attrition_Flag'], inplace=True)
    input_df = input_df[features]
    prediction = model.predict(input_df)
    return "Churn" if prediction[0] else "Not Churn"

# Main function
def main():
    st.title('Bank Churn Prediction Analysis')
    data = load_data()
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Select Option", ("Data Overview", "EDA", "Model Training", "Predict Churn"))

    if options == "Data Overview":
        st.subheader("Dataset Overview:")
        st.write(data.head())
        st.subheader("Summary Statistics:")
        st.write(data.describe())

    elif options == "EDA":
        st.subheader("Correlation Matrix:")
        numeric_data = data.select_dtypes(include='number')
        corr_matrix = numeric_data.corr()
        st.write(corr_matrix)
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        st.pyplot(plt)
        st.subheader("Distribution of Attrition Flag:")
        st.bar_chart(data['Attrition_Flag'].value_counts())

    elif options == "Model Training":
        st.subheader("Model Training:")
        data = preprocess_data(data)
        X = data.drop(columns=['Attrition_Flag'])
        y = data['Attrition_Flag']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write("Accuracy:", accuracy)
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))
        st.session_state['model'] = model
        st.session_state['features'] = X.columns.tolist()

    elif options == "Predict Churn":
        st.subheader("Enter customer details to predict churn:")
        input_data = {
            'Gender': st.selectbox('Gender', ['Male', 'Female']),
            'Customer_Age': st.number_input('Customer Age', min_value=18, max_value=100),
            'Dependent_count': st.number_input('Dependent Count', min_value=0, max_value=10),
            'Education_Level': st.selectbox('Education Level', ['Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate']),
            'Marital_Status': st.selectbox('Marital Status', ['Single', 'Married', 'Divorced']),
            'Income_Category': st.selectbox('Income Category', ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +']),
            'Card_Category': st.selectbox('Card Category', ['Blue', 'Silver', 'Gold', 'Platinum']),
            'Months_on_book': st.number_input('Months on book', min_value=0, max_value=100),
            'Total_Relationship_Count': st.number_input('Total Relationship Count', min_value=0, max_value=10),
            'Months_Inactive_12_mon': st.number_input('Months Inactive in last 12 months', min_value=0, max_value=12),
            'Contacts_Count_12_mon': st.number_input('Contacts Count in last 12 months', min_value=0, max_value=20),
            'Credit_Limit': st.number_input('Credit Limit', min_value=0.0, step=0.1),
            'Total_Revolving_Bal': st.number_input('Total Revolving Balance', min_value=0.0, step=0.1),
            'Avg_Open_To_Buy': st.number_input('Average Open To Buy', min_value=0.0, step=0.1),
            'Total_Amt_Chng_Q4_Q1': st.number_input('Total Amount Change Q4 to Q1', min_value=0.0, step=0.1),
            'Total_Trans_Amt': st.number_input('Total Transaction Amount', min_value=0.0, step=0.1),
            'Total_Trans_Ct': st.number_input('Total Transaction Count', min_value=0),
            'Total_Ct_Chng_Q4_Q1': st.number_input('Total Count Change Q4 to Q1', min_value=0.0, step=0.1),
            'Avg_Utilization_Ratio': st.number_input('Average Utilization Ratio', min_value=0.0, step=0.1)
        }
        input_df = pd.DataFrame([input_data])
        if 'model' in st.session_state and 'features' in st.session_state:
            if st.button('Predict Churn'):
                model = st.session_state['model']
                features = st.session_state['features']
                prediction = predict_churn(model, input_df, features)
                st.subheader("Prediction:")
                st.write("The customer is predicted to be:", prediction)
        else:
            st.write("Please train the model first in the 'Model Training' section.")

if __name__ == "__main__":
    main()
