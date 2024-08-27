from flask import Flask, request, render_template

import pickle
import numpy as np
import pandas as pd
# import h5py
import joblib

# Initialize the Flask application

with open('gbm_classifier_state_unknown.pkl','r') as file:
    model = pickle.load(file)
app = flask(__name__)
# @app.route('/')
# def man():
#     return render_template('index.html')
# with open('gbm_classifier.pkl') as f:
#     gbm_classifier = pickle.load(f)
# model_filename='gbm_classifier.h5'
# with h5py.File(model_filename, 'r') as h5file:
#     # Load the model data
#     model_data = h5file['model'][()]
#     gbm_classifier_loaded = joblib.loads(model_data.tobytes())

    # # Load the additional information
    # accuracy = h5file.attrs['accuracy']
    # print(f"Accuracy: {accuracy}")

    # conf_matrix = np.array(h5file['confusion_matrix'])
    # print("Confusion Matrix:")
    # print(conf_matrix)

# # Load mappings for categorical values
# BCC_mappings = pd.read_csv('Breast_Cancer_Diagnosis_Code_Mapping.csv')
# BCD_mappings=pd.read_csv('Breast_Cancer_Diagnosis_Description_Mapping.csv')
# MCC_mappings=pd.read_csv('Metastatic_Cancer_Diagnosis_Code_Mapping.csv')
# Function to get numerical code from the CSV mappings
df1=pd.read_csv('Breast_Cancer_Diagnosis_Code_Mapping.csv')

BCC_mappings = {}

# Read the CSV file and populate the dictionary
with open(df1, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row if it exists
    for row in reader:
        Original_Value, Encoded_Label = row
        BCC_mappings[Original_Value] = Encoded_Label

df2=pd.read_csv('Breast_Cancer_Diagnosis_Description_Mapping.csv')

BCD_mappings = {}

# Read the CSV file and populate the dictionary
with open(df2, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row if it exists
    for row in reader:
        Original_Value, Encoded_Label = row
        BCD_mappings[Original_Value] = Encoded_Label

df3=pd.read_csv('Metastatic_Cancer_Diagnosis_Code_Mapping.csv')

MCC_mappings = {}

# Read the CSV file and populate the dictionary
with open(df3, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row if it exists
    for row in reader:
        Original_Value, Encoded_Label = row
        MCC_mappings[Original_Value] = Encoded_Label

df4=pd.read_csv('patient_state_Mapping(1).csv')

PS_mappings = {}

# Read the CSV file and populate the dictionary
with open(df4, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row if it exists
    for row in reader:
        Original_Value, Encoded_Label = row
        PS_mappings[Original_Value] = Encoded_Label

def get_BCcode(code):
    return BCC_mappings['code']
def get_BCDcode(code):
    return BCD_mappings['code']
def get_MCcode(code):
    return MCC_mappings['code']
def get_PScode(code):
    return PS_mappings['code']
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index', methods=['POST'])
def predict():
    age = request.form['age']
    bmi = float(request.form['bmi']).round(2)
    breast_cancer_code = request.form['breast-cancer-code']
    breast_cancer_desc = request.form['breast-cancer-desc']
    metastatic_cancer_code = request.form['metastatic-cancer-code']
    ozone = float(request.form['ozone']).round(2)
    pm25 = float(request.form['PM25']).round(2)
    no2 = float(request.form['NO2']).round(2)
    ps=request.form['patient_state']
    zip=request.form['zip']
    
    # Map the categorical values to their numerical codes
    breast_cancer_code = get_BCcode(breast_cancer_code)
    breast_cancer_desc = get_BCDcode(breast_cancer_desc)
    metastatic_cancer_code=get_MCcode(metastatic_cancer_code)
    patient_state=get_PScode(ps)
    
    # Create the feature array
    features = np.array([[patient_state,zip,age, bmi,breast_cancer_code,breast_cancer_desc,metastatic_cancer_code, ozone, pm25, no2]])

    # Make a prediction
    predict = model.predict(features)[0]

    return render_template('index.html', prediction=predict)

if __name__ == "__main__":
    app.run(debug=True)
