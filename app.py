import streamlit as st 
import pandas as pd
import joblib

# Set the webpage title
st.set_page_config(page_title="Iris Project")

# Load the model
model = joblib.load("notebook/model.joblib")

# Function to predict results
def predict_results(model, sep_len, sep_wid, pet_len, pet_wid):
    # Get the data in dictionary format
    d = {
        "sepal_length": [sep_len],
        "sepal_width": [sep_wid],
        "petal_length": [pet_len],
        "petal_width": [pet_wid]
    }

    # Convert dictionary to dataframe
    xnew = pd.DataFrame(d)

    # Predict the results with probability
    preds = model.predict(xnew)
    probs = model.predict_proba(xnew)

    # Probs save as dictionary
    classes = model.classes_

    # Apply for loop and save results in dictionary format
    prob_d = {}
    for c, p in zip(classes, probs.flatten()):
        prob_d[c] = float(p)

    # Return prediction with prob
    return preds[0], prob_d

# Creating the web app interface
st.title("Iris Prediction Project")
st.subheader("by Utkarsh Gaikwad")

# Take inputs from user
sep_len = st.number_input("Sepal Length", min_value=0.00, step=0.01)
sep_wid = st.number_input("Sepal Width", min_value=0.00, step=0.01)
pet_len = st.number_input("Petal Length", min_value=0.00, step=0.01)
pet_wid = st.number_input("Petal Width", min_value=0.00, step=0.01)

# Create a button for predicting results
submit = st.button("predict", type="primary")

# After pressing submit button
if submit:
    pred, prob = predict_results(model, sep_len, sep_wid, pet_len, pet_wid)
    st.subheader(f"Prediction : {pred}")

    for c, p in prob.items():
        st.subheader(f"{c} : {p:.4f}")
        st.progress(p)
