# Health App - Determining the risk of Heart Disease

This is a streamlit application built to determin if a user is at risk of heart disease based on user inputs. This application takes the inputs of 
* Age: Patient's Age in years (Numeric)
* Sex: Patient's Gender Male as 1 Female as 0 (Nominal)
* Chest Pain Type: Type of chest pain categorized into 1 typical, 2 typical angina, 3 non-anginal pain, 4 asymptomatic(Categorical)
* Resting BPs: Level of blood pressure at resting mode in mm/HG (Numerical)
* Cholesterol: Serum cholestrol in mg/dl (Numeric)
* Fasting Blood Sugar: Blood sugar levels on fasting > 120 mg/dl represents as 1 in case of true and 0 as false
* Resting ECG: result of electrocardiogram while at rest are represented in 3 distinct values 0 : Normal 1: Abnormality in ST-T wave 2: Left ventricular hypertrophy
* Max Heart Rate: Maximum heart rate achieved (Numeric)
* Exercise Angimia: Angina induced by exercise 0 depicting NO 1 depicting Yes (Nominal)
* Old Peak: Exercise induced ST-depression in comparison with the state of rest (Numeric)
* ST Slope: ST segment measured in terms of slope during peak exercise 0: Normal 1: Upsloping 2: Flat 3: Downsloping

The inputs are then converted to a panda dataframe and is then run against a predictive model to determin if the user is at risk of heart disease. The model that is utilized is a Decision Tree trained on a dataset of heart risk factors.

The dataset the model is trained on is linked in the github repo for viewing.

Summary statistics, datacleaning, model building, model interpretation and documentation can be found on the linked colab document.

To run the app it can accessed through the following link:

https://healthapp-3fytbcnbbxtwdgy2owexhc.streamlit.app/

Or the code can be downloaded and follow the following steps for local deployment after making sure python and pip are installed

```
pip install -r requirments
```

```
$ streamlit run main.py
```

Sources ued to complete the project:

* https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall
* https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
