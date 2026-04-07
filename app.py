import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Student Mental Health Predictor", page_icon="🧠", layout="centered")

# Load and prepare data
df = pd.read_csv("Student Mental health.csv")
df.columns = ['Timestamp', 'Gender', 'Age', 'Course', 'Year', 'CGPA', 'Married', 'Depression', 'Anxiety', 'Panic_Attack', 'Sought_Help']
df['Age'] = df['Age'].fillna(df['Age'].mean())

df_ml = df[['Gender', 'Age', 'Year', 'CGPA', 'Anxiety', 'Panic_Attack', 'Married', 'Depression']].copy()
df_ml['Gender'] = df_ml['Gender'].map({'Female': 0, 'Male': 1})
df_ml['Anxiety'] = df_ml['Anxiety'].map({'No': 0, 'Yes': 1})
df_ml['Panic_Attack'] = df_ml['Panic_Attack'].map({'No': 0, 'Yes': 1})
df_ml['Married'] = df_ml['Married'].map({'No': 0, 'Yes': 1})
df_ml['Depression'] = df_ml['Depression'].map({'No': 0, 'Yes': 1})
df_ml['CGPA'] = LabelEncoder().fit_transform(df_ml['CGPA'].astype(str))
df_ml['Year'] = LabelEncoder().fit_transform(df_ml['Year'].astype(str))

X = df_ml[['Gender', 'Age', 'Year', 'CGPA', 'Anxiety', 'Panic_Attack', 'Married']]
y = df_ml['Depression']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Header
st.title("🧠 Student Mental Health Predictor")
st.markdown("Fill in the details below to check your depression risk level.")
st.divider()

# Metric cards
col1, col2, col3 = st.columns(3)
col1.metric("Students Surveyed", "101")
col2.metric("Model Accuracy", "71%")
col3.metric("Risk Factors", "7")
st.divider()

# Input form
st.subheader("Student Details")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    year = st.selectbox("Year of Study", ["year 1", "year 2", "year 3", "year 4"])
    anxiety = st.selectbox("Do you have Anxiety?", ["No", "Yes"])
    married = st.selectbox("Marital Status", ["No", "Yes"])

with col2:
    age = st.slider("Age", 18, 30, 20)
    cgpa = st.selectbox("CGPA Range", ["3.50 - 4.00", "3.00 - 3.49", "2.50 - 2.99", "2.00 - 2.49", "0 - 1.99"])
    panic = st.selectbox("Do you have Panic Attacks?", ["No", "Yes"])

st.divider()

# Encode inputs
gender_val = 0 if gender == "Female" else 1
anxiety_val = 0 if anxiety == "No" else 1
panic_val = 0 if panic == "No" else 1
married_val = 0 if married == "No" else 1
cgpa_map = {"0 - 1.99": 0, "2.00 - 2.49": 1, "2.50 - 2.99": 2, "3.00 - 3.49": 3, "3.50 - 4.00": 4}
year_map = {"year 1": 3, "year 2": 4, "year 3": 5, "year 4": 6}

input_data = pd.DataFrame([[gender_val, age, year_map[year], cgpa_map[cgpa], anxiety_val, panic_val, married_val]],
                           columns=['Gender', 'Age', 'Year', 'CGPA', 'Anxiety', 'Panic_Attack', 'Married'])

# Predict button
if st.button("Check Risk Level", type="primary", use_container_width=True):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ High Risk of Depression Detected — {probability*100:.1f}% probability")
        st.progress(probability)
        st.markdown("**Suggestions:**")
        st.markdown("- Talk to a counselor or trusted person")
        st.markdown("- Maintain a regular sleep schedule")
        st.markdown("- Take breaks and practice mindfulness")
        st.markdown("- Stay connected with friends and family")
    else:
        st.success(f"✅ Low Risk of Depression — {probability*100:.1f}% probability")
        st.progress(probability)
        st.markdown("**Keep it up!**")
        st.markdown("- Keep maintaining your healthy habits")
        st.markdown("- Regular exercise helps mental health")
        st.markdown("- Stay connected with friends and family")