import streamlit as st
import pandas as pd
import ast

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Health Care Center",
    page_icon="🩺",
    layout="wide"
)


# --------------------------------------------------
# LOAD ALL DATASETS
# --------------------------------------------------

@st.cache_data
def load_data():
    training = pd.read_csv("Training.csv")
    description = pd.read_csv("description.csv")
    precautions = pd.read_csv("precautions_df.csv")
    medications = pd.read_csv("medications.csv")
    diets = pd.read_csv("diets.csv")
    workout = pd.read_csv("workout_df.csv")
    return training, description, precautions, medications, diets, workout


df, description_df, precautions_df, medications_df, diets_df, workout_df = load_data()


# --------------------------------------------------
# PREPARE DATA FOR MODEL
# --------------------------------------------------

X = df.drop("prognosis", axis=1)
y = df["prognosis"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# --------------------------------------------------
# TRAIN SVM MODEL
# --------------------------------------------------

@st.cache_resource
def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=20
    )

    model = SVC(
        kernel="linear",
        probability=True,
        random_state=20
    )

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    return model, accuracy


model, accuracy = train_model(X, y_encoded)


# --------------------------------------------------
# HELPER FUNCTIONS - fetch info for a predicted disease
# --------------------------------------------------

def get_description(disease):
    row = description_df[description_df["Disease"] == disease]
    if row.empty:
        return "No description available."
    return row["Description"].values[0]


def get_precautions(disease):
    row = precautions_df[precautions_df["Disease"] == disease]
    if row.empty:
        return []
    cols = ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]
    values = row[cols].values[0]
    return [v for v in values if pd.notna(v) and str(v).strip() != ""]


def parse_list_string(value):
    """Handles columns like "['A', 'B', 'C']" stored as text."""
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, SyntaxError):
        pass
    return [value]


def get_medications(disease):
    row = medications_df[medications_df["Disease"] == disease]
    if row.empty:
        return []
    return parse_list_string(row["Medication"].values[0])


def get_diets(disease):
    row = diets_df[diets_df["Disease"] == disease]
    if row.empty:
        return []
    return parse_list_string(row["Diet"].values[0])


def get_workout(disease):
    # workout_df uses lowercase "disease" column and has one row per tip
    rows = workout_df[workout_df["disease"] == disease]
    if rows.empty:
        return []
    return rows["workout"].tolist()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🩺 Health Care Center")

st.sidebar.info(
    "Select your symptoms and click Predict Disease."
)

st.sidebar.write(f"Number of symptoms: {len(X.columns)}")
st.sidebar.write(f"Number of diseases: {len(label_encoder.classes_)}")
st.sidebar.write(f"Model accuracy: {accuracy * 100:.2f}%")


# --------------------------------------------------
# MAIN TITLE
# --------------------------------------------------

st.title("🩺 Health Care Center")
st.subheader("Disease Prediction & Recommendation System")
st.write("Select the symptoms you are experiencing.")


# --------------------------------------------------
# SYMPTOM SELECTION
# --------------------------------------------------

symptoms = list(X.columns)

selected_symptoms = st.multiselect(
    "Select Symptoms",
    symptoms,
    placeholder="Choose your symptoms..."
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("🔍 Predict Disease", type="primary"):

    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom.")

    else:
        # Create input with all symptoms = 0
        input_data = pd.DataFrame(0, index=[0], columns=symptoms)

        # Selected symptoms = 1
        for symptom in selected_symptoms:
            input_data[symptom] = 1

        # Prediction
        prediction = model.predict(input_data)
        disease = label_encoder.inverse_transform(prediction)[0]

        # Probability
        probabilities = model.predict_proba(input_data)[0]
        confidence = probabilities.max() * 100

        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        st.success("Prediction completed.")
        st.markdown(f"## 🩺 Predicted Disease: {disease}")
        st.info(f"Model confidence: {confidence:.2f}%")

        st.subheader("✅ Selected Symptoms")
        for symptom in selected_symptoms:
            st.write(f"✓ {symptom.replace('_', ' ').title()}")

        st.divider()

        # Tabs for all the recommendation info
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📝 Description", "⚠️ Precautions", "💊 Medications", "🥗 Diets", "🏋️ Workout"]
        )

        with tab1:
            st.write(get_description(disease))

        with tab2:
            precautions = get_precautions(disease)
            if precautions:
                for i, p in enumerate(precautions, start=1):
                    st.write(f"{i}. {p.capitalize()}")
            else:
                st.write("No precautions found.")

        with tab3:
            meds = get_medications(disease)
            if meds:
                for m in meds:
                    st.write(f"- {m}")
            else:
                st.write("No medication info found.")

        with tab4:
            diet_items = get_diets(disease)
            if diet_items:
                for d in diet_items:
                    st.write(f"- {d}")
            else:
                st.write("No diet info found.")

        with tab5:
            workouts = get_workout(disease)
            if workouts:
                for w in workouts:
                    st.write(f"- {w}")
            else:
                st.write("No workout info found.")

        st.warning(
            "This is an educational machine-learning demonstration "
            "and should not be used as a medical diagnosis or "
            "as a substitute for a qualified healthcare professional."
        )


# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

with st.expander("📊 Dataset Information"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", df.shape[0])
    with col2:
        st.metric("Symptoms", X.shape[1])
    with col3:
        st.metric("Diseases", len(label_encoder.classes_))


with st.expander("🔬 Model Information"):
    st.write("Algorithm: Support Vector Machine (SVM)")
    st.write("Kernel: Linear")
    st.write(f"Training records: {len(df)}")
    st.write(f"Number of symptoms: {X.shape[1]}")
    st.write(f"Number of diseases: {len(label_encoder.classes_)}")
    st.write(f"Test accuracy: {accuracy * 100:.2f}%")