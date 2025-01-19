import streamlit as st
import joblib
import requests
from io import BytesIO
import matplotlib.pyplot as plt

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
    # Centered Title with Break Line and Bold Italic Styling
    st.markdown(
        """
        <h1 style='text-align: center; color: #0000FF; font-size: 34px;'>
        WWSEF 2025: Thyroid Cancer Recurrence Prediction Machine Learning Model<br>
        <b><i style="font-size: 18px;">Aanya Mendapara</i></b>
        </h1>
        """,
        unsafe_allow_html=True
    )
    
    # Sentence with Bold Italic, Custom Font Size, and Color
    st.markdown(
        """
        <p style='color: #ff0000; font-size: 22px;'>
        <b>Select inputs for each feature below:</b>
        </p>
        """,
        unsafe_allow_html=True
    )

    # Custom CSS for removing extra spacing (try again with a different approach)
    st.markdown(
        """
        <style>
        .stTextInput, .stSelectbox, .stNumberInput {
            padding-top: 0px;
            padding-bottom: 0px;
            margin-top: 0px;
            margin-bottom: 0px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Using a form to group elements
    with st.form(key='input_form'):
        # Feature: Age
        st.markdown("<p>Age:</p>", unsafe_allow_html=True)
        age = st.number_input("", min_value=0, max_value=120, step=1)

        # Feature: Stage
        st.markdown("<p>Stage:</p>", unsafe_allow_html=True)
        stage = st.selectbox("", options=[0, 1, 2, 3, 4], format_func=lambda x: {
            0: "I", 1: "II", 2: "III", 3: "IVB", 4: "IVA"
        }[x])

        # Feature: T (Tumor size)
        st.markdown("<p>T (Tumor size):</p>", unsafe_allow_html=True)
        t = st.selectbox("", options=[0, 1, 2, 3, 4, 5, 6], format_func=lambda x: {
            0: "T1a", 1: "T1b", 2: "T2", 3: "T3a", 4: "T3b", 5: "T4a", 6: "T4b"
        }[x])

        # Feature: N (Node involvement)
        st.markdown("<p>N (Node involvement):</p>", unsafe_allow_html=True)
        n = st.selectbox("", options=[0, 1, 2], format_func=lambda x: {
            0: "N0", 1: "N1b", 2: "N1a"
        }[x])

        # Feature: Adenopathy
        st.markdown("<p>Adenopathy:</p>", unsafe_allow_html=True)
        adenopathy = st.selectbox("", options=[0, 1, 2, 3, 4, 5], format_func=lambda x: {
            0: "No", 1: "Right", 2: "Extensive", 3: "Left", 4: "Bilateral", 5: "Posterior"}[x])

        # Feature: Response
        st.markdown("<p>Response:</p>", unsafe_allow_html=True)
        response = st.selectbox("", options=[0, 1, 2, 3], format_func=lambda x: {
            0: "Excellent", 1: "Indeterminate", 2: "Structural Incomplete", 3: "Biochemical Incomplete"}[x])

        # Submit button inside form
        submit_button = st.form_submit_button("Predict")

    if submit_button:
        input_data = [[age, stage, t, n, adenopathy, response]]

        # Display input data for debugging purposes
        st.write(f"Input data: {input_data}")

        probabilities = model.predict_proba(input_data)
        confidence_no_recurrence = probabilities[0][0] * 100
        confidence_recurrence = probabilities[0][1] * 100

        # Display results as confidence percentages with custom styles
        st.markdown(
            f"""
            <p style='text-align: center; font-size: 20px; color: red;'>
            <b>Confidence in Recurrence:</b> {confidence_recurrence:.2f}%
            </p>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <p style='text-align: center; font-size: 20px; color: green;'>
            <b>Confidence in No Recurrence:</b> {confidence_no_recurrence:.2f}%
            </p>
            """,
            unsafe_allow_html=True
        )

        # Visualization with Matplotlib
        fig, ax = plt.subplots()
        ax.bar(
            ["No Recurrence", "Recurrence"],
            [confidence_no_recurrence, confidence_recurrence],
            color=["green", "red"],
        )
        ax.set_ylabel("Confidence Percentage")
        ax.set_title("Prediction Confidence")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
