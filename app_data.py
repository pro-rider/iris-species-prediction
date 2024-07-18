
import pandas as pd
import joblib
import streamlit as st

# Load the model and feature names
model = joblib.load(r'C:\Users\DELL\Desktop\new project\git hub\iris_model.pkl')  # Path to the saved model
features = joblib.load(r'C:\Users\DELL\Desktop\new project\git hub\iris_features.pkl')  # Path to the saved features

# Define the predict function
def predict(new_data):
    p = model.predict(new_data)
    return p

# Get user inputs
sepal_length = st.number_input("Enter sepal length")
sepal_width = st.number_input("Enter sepal width")
petal_length = st.number_input("Enter petal length")
petal_width = st.number_input("Enter petal width")

# Convert input data into DataFrame
input_data = pd.DataFrame({
    'sepal length (cm)': [sepal_length],
    'sepal width (cm)': [sepal_width],
    'petal length (cm)': [petal_length],
    'petal width (cm)': [petal_width]
})

# Predict button
if st.button('Predict'):
    prediction = predict(input_data)
    st.write('Predicted Species:', prediction)
    if prediction == 0:
        st.write('This flower belongs to Setosa species')
    elif prediction == 1:
        st.write('This flower belongs to Versicolor species')
    elif prediction == 2:
        st.write('This flower belongs to Virginica species')
