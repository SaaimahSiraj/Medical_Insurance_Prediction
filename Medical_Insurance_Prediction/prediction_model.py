import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

# Load dataset and preprocess
def load_data(file_path):
    data = pd.read_csv(file_path)

    # Map categorical variables to numeric
    data['sex'] = data['sex'].map({'male': 0, 'female': 1})
    data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})
    data = pd.get_dummies(data, columns=['region'], drop_first=True)
    return data

# Train model
def train_model(data):
    X = data.drop(columns=['expenses'])
    y = data['expenses']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the trained model and feature columns
    joblib.dump(model, 'insurance_model.pkl')
    joblib.dump(X.columns.tolist(), 'model_columns.pkl')  # Save feature column names
    return model

# Predict expenses
def predict_expenses(input_data):
    model_path = 'insurance_model.pkl'
    columns_path = 'model_columns.pkl'

    # Train the model if it doesn't exist
    if not os.path.exists(model_path) or not os.path.exists(columns_path):
        data = load_data('insurance.csv')
        train_model(data)

    # Load the model and feature columns
    model = joblib.load(model_path)
    model_columns = joblib.load(columns_path)

    # Create a DataFrame for input data to match feature columns
    input_df = pd.DataFrame([input_data], columns=model_columns)

    # Ensure missing columns are filled with 0
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Check for NaN values and fill them
    input_df = input_df.fillna(0)

    # Predict and return result
    return model.predict(input_df)
