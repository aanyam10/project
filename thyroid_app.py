import streamlit as st
import joblib
import requests
from io import BytesIO

@st.cache_resource
def load_model():
    url = "https://raw.githubusercontent.com/aanyam10/project/main/thyroid_model_final.pkl"
    response = requests.get(url)
    if response.status_code == 200:
        return joblib.load(BytesIO(response.content))
    else:
        raise Exception(f"Failed to load model. Status code: {response.status_code}")

model = load_model()

def main():
    st.title("Thyroid Cancer Recurrence Prediction")
    st.write("Provide inputs for each feature:")

    # Feature: Age
    age = st.number_input("Age:", min_value=0, max_value=120, step=1)

    # Feature: Stage (I: 0, II: 1, III: 2, IVB: 3, IVA: 4)
    stage = st.selectbox("Stage:", options=[0, 1, 2, 3, 4], format_func=lambda x: {
        0: "I", 1: "II", 2: "III", 3: "IVB", 4: "IVA"
    }[x])

    # Feature: T (Tumor size/stage) (T1a: 0, T1b: 1, T2: 2, T3a: 3, T3b: 4, T4a: 5, T4b: 6)
    t = st.selectbox("T (Tumor size/stage):", options=[0, 1, 2, 3, 4, 5, 6], format_func=lambda x: {
        0: "T1a", 1: "T1b", 2: "T2", 3: "T3a", 4: "T3b", 5: "T4a", 6: "T4b"
    }[x])

    # Feature: N (Node involvement) (N0: 0, N1b: 1, N1a: 2)
    n = st.selectbox("N (Node involvement):", options=[0, 1, 2], format_func=lambda x: {
        0: "N0", 1: "N1b", 2: "N1a"
    }[x])

   # Feature: Adenopathy ('No':0,'Right':1, 'Extensive':2, 'Left':3, 'Bilateral':4, 'Posterior':5)
    adenopathy = st.selectbox("adenopathy:", options=[0, 1, 2, 3, 4, 5], format_func=lambda x: {
        0: "No", 1: "Right", 2: "Extensive", 3: "Left", 4: "Bilateral", 5: "Posterior"}[x])

    # Feature: Response ('Excellent':0,'Indeterminate':1, 'Structural Incomplete':2, 'Biochemical Incomplete':3)
    response = st.selectbox("response:", options=[0, 1, 2, 3], format_func=lambda x: {
        0: "Excellent", 1: "Indeterminate", 2: "Structural Incomplete", 3: "Biochemical Incomplete"}[x])

    # Combine all features into a single input array
    input_data = [[age, stage, t, n, adenopathy, response]]

    # Add this right before making the prediction in the Streamlit app
    st.write(f"Input data: {input_data}")

    # Predict button
    if st.button("Predict"):
        probabilities = model.predict_proba(input_data)
        confidence_no_recurrence = probabilities[0][0] * 100
        confidence_recurrence = probabilities[0][1] * 100

        # Display results as confidence percentages
        st.write(f"**Confidence in Recurrence:** {confidence_recurrence:.2f}%")

if __name__ == "__main__":
    main()
