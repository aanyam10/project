import streamlit as st
import joblib
import requests
from io import BytesIO

@st.cache_resource
def load_model():
    model_url = "https://raw.githubusercontent.com/aanyam10/project/main/thyroid_model_1.pkl"  # Update with your model URL
    try:
        # Download the model file from the URL
        response = requests.get(model_url)
        response.raise_for_status()  # Raise an error for bad status codes (404, 500, etc.)
        
        # Load the model from the response content
        model = joblib.load(BytesIO(response.content))
        st.write("Model loaded successfully.")
        return model
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading model: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

def main():
    st.title("Thyroid Cancer Recurrence Prediction")
    st.write("Provide inputs for each feature:")

    # Feature: Age
    age = st.number_input("Age:", min_value=0, max_value=120, step=1)

    # Feature: Smoking (Yes: 1, No: 0)
    smoking = st.selectbox("Smoking:", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    # Feature: N (Node involvement) (N0: 0, N1b: 1, N1a: 2)
    n = st.selectbox("N (Node involvement):", options=[0, 1, 2], format_func=lambda x: {
        0: "N0", 1: "N1b", 2: "N1a"
    }[x])

    # Feature: T (Tumor size/stage) (T1a: 0, T1b: 1, T2: 2, T3a: 3, T3b: 4, T4a: 5, T4b: 6)
    t = st.selectbox("T (Tumor size/stage):", options=[0, 1, 2, 3, 4, 5, 6], format_func=lambda x: {
        0: "T1a", 1: "T1b", 2: "T2", 3: "T3a", 4: "T3b", 5: "T4a", 6: "T4b"
    }[x])

    # Feature: Stage (I: 0, II: 1, III: 2, IVB: 3, IVA: 4)
    stage = st.selectbox("Stage:", options=[0, 1, 2, 3, 4], format_func=lambda x: {
        0: "I", 1: "II", 2: "III", 3: "IVB", 4: "IVA"
    }[x])

    # Feature: Hx Smoking (Yes: 1, No: 0)
    hx_smoking = st.selectbox("Hx Smoking:", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    # Combine all features into a single input array
    input_data = [[age, smoking, n, t, stage, hx_smoking]]

    # Predict button
    if st.button("Predict"):
        probabilities = model.predict_proba(input_data)
        confidence_no_recurrence = probabilities[0][0] * 100
        confidence_recurrence = probabilities[0][1] * 100

        # Display results as confidence percentages
        st.write(f"**Confidence in Recurrence:** {confidence_recurrence:.2f}%")

if __name__ == "__main__":
    main()
