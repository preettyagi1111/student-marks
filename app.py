import streamlit as st
import joblib
import pandas as pd


# ==============================
# PAGE SETTINGS
# ==============================

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)


# ==============================
# LOAD TRAINED MODEL
# ==============================

model_data = joblib.load("model/student_model.pkl")

model = model_data["model"]
features = model_data["features"]


# ==============================
# WEBSITE TITLE
# ==============================

st.title("🎓 Student Performance Predictor")

st.write(
    "Enter the student's academic information "
    "to predict final performance using Machine Learning."
)

st.divider()


# ==============================
# STUDENT INPUT
# ==============================

st.header("📋 Student Information")

col1, col2 = st.columns(2)


with col1:

    attendance = st.number_input(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
        step=1.0
    )

    study_hours = st.number_input(
        "Study Hours per Day",
        min_value=0.0,
        max_value=24.0,
        value=4.0,
        step=0.5
    )

    previous_marks = st.number_input(
        "Previous Marks",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        step=1.0
    )

    assignment_score = st.number_input(
        "Assignment Score",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0
    )


with col2:

    internal_marks = st.number_input(
        "Internal Marks",
        min_value=0.0,
        max_value=100.0,
        value=65.0,
        step=1.0
    )

    practical_marks = st.number_input(
        "Practical Marks",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0
    )

    backlogs = st.number_input(
        "Backlogs",
        min_value=0,
        max_value=20,
        value=0,
        step=1
    )


st.divider()


# ==============================
# PREDICTION BUTTON
# ==============================

if st.button(
    "🔮 Predict Performance",
    use_container_width=True
):

    input_data = pd.DataFrame(
        [[
            attendance,
            study_hours,
            previous_marks,
            assignment_score,
            internal_marks,
            practical_marks,
            backlogs
        ]],
        columns=features
    )

    prediction = model.predict(input_data)

    predicted_marks = float(prediction[0])

    st.success("Prediction generated successfully!")

    st.metric(
        "Predicted Final Marks",
        f"{predicted_marks:.2f}"
    )