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
        <b>Select inputs for each feature from dropdown list:</b>
        </p>
        """,
        unsafe_allow_html=True
    )

    # Feature: Age
    age = st.number_input("Age:", min_value=0, max_value=120, step=1)

    # Feature: Stage (I: 0, II: 1, III: 2, IVB: 3, IVA: 4)
    stage = st.selectbox("Stage:", options=[0, 1, 2, 3, 4], format_func=lambda x: {
        0: "I", 1: "II", 2: "III", 3: "IVB", 4: "IVA"
    }[x])

    # Feature: T (Tumor size) (T1a: 0, T1b: 1, T2: 2, T3a: 3, T3b: 4, T4a: 5, T4b: 6)
    t = st.selectbox("T (Tumor size):", options=[0, 1, 2, 3, 4, 5, 6], format_func=lambda x: {
        0: "T1a", 1: "T1b", 2: "T2", 3: "T3a", 4: "T3b", 5: "T4a", 6: "T4b"
    }[x])

    # Feature: N (Node involvement) (N0: 0, N1b: 1, N1a: 2)
    n = st.selectbox("N (Node involvement):", options=[0, 1, 2], format_func=lambda x: {
        0: "N0", 1: "N1b", 2: "N1a"
    }[x])

    # Feature: Adenopathy ('No':0,'Right':1,'Extensive':2,'Left':3,'Bilateral':4,'Posterior':5)
    adenopathy = st.selectbox("Adenopathy:", options=[0, 1, 2, 3, 4, 5], format_func=lambda x: {
        0: "No", 1: "Right", 2: "Extensive", 3: "Left", 4: "Bilateral", 5: "Posterior"}[x])

    # Feature: Response ('Excellent':0,'Indeterminate':1,'Structural Incomplete':2,'Biochemical Incomplete':3)
    response = st.selectbox("Response:", options=[0, 1, 2, 3], format_func=lambda x: {
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
        #st.write(f"**Confidence in No Recurrence:** {confidence_no_recurrence:.2f}%")

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
