# Import necessary packages
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Step 1: Generate dummy data
def generate_dummy_data():
    """Generate dummy data for marks and grades."""
    np.random.seed(42)
    data = {
        "marks": np.random.randint(30, 100, 20),
        "grades": np.random.choice(['A', 'B', 'C', 'D', 'E'], 20)
    }
    df = pd.DataFrame(data)
    return df


# Step 2: Convert grades to numeric values
def preprocess_data(df):
    """Convert grades to numeric values."""
    label_encoder = LabelEncoder()
    df['grades_numeric'] = label_encoder.fit_transform(df['grades'])
    return df, label_encoder


# Step 3: Train the linear regression model
def train_model(df):
    """Train a Linear Regression model on the dummy data."""
    X = df[['marks']].values
    y = df['grades_numeric'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# Step 4: Build Streamlit web app
def main():
    # App title
    st.title("Student Grade Prediction App")
    st.write("This app predicts the grade of a student based on their marks.")

    # Input field for marks
    marks = st.number_input("Enter the marks (0-100):", min_value=0, max_value=100, step=1)

    # Predict button
    if st.button("Predict"):
        # Generate and preprocess dummy data
        with st.spinner("Generating data and training the model..."):
            df = generate_dummy_data()
            df, label_encoder = preprocess_data(df)
            model = train_model(df)

        # Make prediction
        predicted_grade_numeric = model.predict([[marks]])
        predicted_grade_numeric = round(predicted_grade_numeric[0])
        predicted_grade_numeric = np.clip(predicted_grade_numeric, 0, len(label_encoder.classes_) - 1)

        # Convert prediction back to grade
        predicted_grade = label_encoder.inverse_transform([predicted_grade_numeric])[0]
        st.success(f"The predicted grade is: {predicted_grade}")

    # Debugging and displaying dummy data (optional for development)
    if st.checkbox("Show dummy data"):
        st.write(generate_dummy_data())


# Run the app
if __name__ == "__main__":
    main()

