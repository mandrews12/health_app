from utils import *

# Title Section
st.title("Heart Disease Risk Predictor")
st.write("Provide your details below to assess your risk of heart disease.")

# Input Form
with st.form("risk_form"):
    # User Inputs
    age = st.slider("Enter your age", 0, 100, 53)
    sex = st.selectbox("Select your gender", ["Male", "Female"])
    chest_pain_type = st.selectbox("Select your chest pain type", [1, 2, 3, 4])
    bps = st.slider("Enter your resting blood pressure (mmHg)", 0, 200, 132)
    cholesterol = st.slider("Enter your total cholesterol level (mg/dL)", 0, 700, 210)
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dL?", ["Yes (1)", "No (0)"])
    ecg = st.selectbox("Resting ECG results", [0, 1, 2])
    max_hr = st.slider("Enter your maximum heart rate (bpm)", 0, 250, 140)
    exerc_angina = st.selectbox("Do you experience exercise-induced angina?", ["Yes (1)", "No (0)"])
    old_peak = st.slider("Enter ST depression induced by exercise (Old Peak)", 0.0, 10.0, 2.0)
    st_slope = st.selectbox("Enter the slope of the peak exercise ST segment", [0, 1, 2])

    # Submit Button
    submitted = st.form_submit_button("Predict")
    if submitted:
        # get input values and send to model predictor
        input_data = {
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "chest pain type": chest_pain_type,
        "resting bp s": bps,
        "cholesterol": cholesterol,
        "fasting blood sugar": 1 if fbs == "Yes (1)" else 0,
        "resting ecg": ecg,
        "max heart rate": max_hr,
        "exercise angina": 1 if exerc_angina == "Yes (1)" else 0,
        "oldpeak": old_peak,
        "ST slope": st_slope
        }
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        #Call the prediction function
        prediction = predict(input_df)
        # Display the prediction result
        if prediction[0] == 1:
            st.write("You are at risk of heart disease.")
        else:
            st.write("You are not at risk of heart disease.")
