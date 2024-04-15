from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('diabetes_model.sav', 'rb'))
scaler = pickle.load(open('diabetes_model_scaler.sav', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    # Get form data
    data = {}
    data['Pregnancies'] = int(request.form.get('Pregnancies'))
    data['Glucose'] = int(request.form.get('Glucose'))
    data['BloodPressure'] = int(request.form.get('BloodPressure'))  
    data['SkinThickness'] = int(request.form.get('SkinThickness')) 
    data['Insulin'] = int(request.form.get('Insulin'))
    data['BMI'] = float(request.form.get('BMI'))
    data['DiabetesPedigreeFunction'] = float(request.form.get('DiabetesPedigreeFunction'))
    data['Age'] = int(request.form.get('Age'))


    df = pd.DataFrame([data])

    df =pd.DataFrame(scaler.transform(df),columns=df.columns)



    pred = model.predict(df)[0]

    if pred == 1:
        pred = 'Diabetes, Please visit your Dr'
    else:
        pred = 'Not  Diabetes'
        
    return render_template('index.html', prediction=pred)

if __name__ == "__main__":
    app.run(debug=True)