import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prediction_model import load_data, predict_expenses

# Set page configuration
st.set_page_config(page_title="Medical Insurance Expense Prediction", page_icon="💰", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f9f9f9;
    }
    h1 {
        font-family: 'Arial Black', sans-serif;
        color: #2E86C1;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background-color: #D6EAF8;
    }
    .stButton > button {
        background-color: #2E86C1;
        color: white;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.title("💰 Medical Insurance Expense Prediction Dashboard")
st.markdown(
    """This application predicts medical insurance expenses based on user-provided information 
    such as age, BMI, number of children, smoking habits, and region of residence. """

   
)
st.markdown(
    """### How It Helps:
    - Financial Planning: Understand and estimate your medical insurance expenses to plan your finances better.
    - Transparency: Know how lifestyle factors like smoking or BMI can impact your insurance costs.
    - Informed Decisions: Gain insights into the factors affecting your premiums, enabling smarter health and financial choices. """

   
)
st.markdown(
    """
     ### How It Works:
    - The application uses a **Linear Regression Model** trained on a dataset of insurance records to predict expenses.
    - It leverages historical data and key features to estimate medical insurance expenses with high accuracy.
    - The model is dynamic and adapts to user inputs, ensuring personalized and accurate predictions.

    ---
    **Developed and Maintained by Saaimah Siraj**
    
    """
)

# Sidebar for navigation
st.sidebar.title("DashBoard")
page = st.sidebar.radio("Go to", ["Home", "Data Insights", "Prediction"])

# Load dataset
data_file = "insurance.csv"
data = pd.read_csv(data_file)
data = load_data(data_file)

if page == "Home":
    st.header("Welcome to the Expense Prediction Dashboard")
    st.markdown("Navigate through the app using the sidebar.")
    st.image("home.jpeg", use_container_width=True)

elif page == "Data Insights":
    st.header("Dataset Insights")

    # Show dataset preview
    st.markdown("### Quick View of the Dataset")
    st.dataframe(data.head(10))

    # Distribution of expenses
    st.markdown("### Distribution of Expenses")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data['expenses'], kde=True, ax=ax, color="#2E86C1")
    st.pyplot(fig)

    # Correlation heatmap
    st.markdown("### Correlation Heatmap")
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

elif page == "Prediction":
    st.header("Enter Details for Prediction")

    # Input fields for user data
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 18, 100, 30)
    with col2:
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    with col3:
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)

    col1, col2 = st.columns(2)
    with col1:
        smoker = st.selectbox("Smoker", ["Yes", "No"])
    with col2:
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    # Prepare input for prediction
    if st.button("Predict"):
        # Convert inputs to numeric format
        smoker = 1 if smoker == "Yes" else 0
        input_data = {
            'age': age,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'region_northwest': 1 if region == 'northwest' else 0,
            'region_southeast': 1 if region == 'southeast' else 0,
            'region_southwest': 1 if region == 'southwest' else 0
        }

        expenses = predict_expenses(input_data)

        # Output section
        st.markdown("---")
        st.subheader("Prediction Result")
        st.success(f"🎉 The predicted insurance expense is **${expenses[0]:,.2f}**.")
