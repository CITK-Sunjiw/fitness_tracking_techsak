import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time


st.set_page_config(page_title="Fitness Tracker", layout="wide")


st.markdown(
    """
    <style>
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .stApp {
            background-color: #87a8ea; color: white;
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            color: white;
        }
        
        .big-font { font-size: 24px !important; font-weight: bold; animation: fadeIn 1.5s ease-in-out; }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            color: black;
            transition: transform 0.3s;
        }
        .metric-card:hover {
            transform: scale(1.05);
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
        }
        
        .sidebar .sidebar-content { background-color: #ffffff; }
        
        .assistant-image {
            display: block;
            margin: auto;
            max-width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True
)


st.sidebar.image("assistant.jpg", caption="Your Fitness Assistant", use_container_width=True)
st.sidebar.header("User Input Parameters")

def user_input_features():
    age = st.sidebar.number_input("Age:", 10, 100, 30)
    bmi = st.sidebar.number_input("BMI:", 15, 40, 20)
    duration = st.sidebar.slider("Duration (min):", 0, 35, 15)
    heart_rate = st.sidebar.slider("Heart Rate:", 60, 130, 80)
    body_temp = st.sidebar.slider("Body Temperature (C):", 36, 42, 38)
    gender_button = st.sidebar.radio("Gender:", ("Male", "Female"))
    gender = 1 if gender_button == "Male" else 0
    return pd.DataFrame({"Age": [age], "BMI": [bmi], "Duration": [duration], "Heart_Rate": [heart_rate], "Body_Temp": [body_temp], "Gender_male": [gender]})

df = user_input_features()

st.markdown("<h1 class='big-font'>🏋️ Personal Fitness Tracker</h1>", unsafe_allow_html=True)
st.write("### Predict your calories burned based on personal metrics")
st.markdown("---")


calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")
exercise_df = exercise.merge(calories, on="User_ID").drop(columns="User_ID")
exercise_df["BMI"] = round(exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2), 2)

data = exercise_df[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
data = pd.get_dummies(data, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(data.drop("Calories", axis=1), data["Calories"], test_size=0.2, random_state=1)


model = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
model.fit(X_train, y_train)
df = df.reindex(columns=X_train.columns, fill_value=0)
prediction = model.predict(df)


col1, col2 = st.columns(2)
with col1:
    st.subheader("Your Input Parameters")
    st.write(df)
with col2:
    st.subheader("🔥 Predicted Calories Burned")
    st.markdown(f"<div class='metric-card'><h2>{round(prediction[0], 2)} kcal</h2></div>", unsafe_allow_html=True)

st.markdown("---")
st.subheader("📊 Similar Results")
calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]
st.dataframe(similar_data.sample(5))

st.markdown("---")
st.subheader("📈 Data Visualization")
fig = px.histogram(exercise_df, x="Calories", nbins=30, title="Calorie Distribution")
st.plotly_chart(fig, use_container_width=True)
