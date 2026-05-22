import joblib
import numpy as np
from flask import Flask, request, jsonify 
model = joblib.load('loan_approval_pipeline.pkl')
app = Flask(__name__) # creating flask name
@app.route('/') # route define the prediction function
def home():
    return '<h1>Loan Approval Prediction using Flask API</h1>'
@app.route('/predict',methods=['POST'])
def predict():
    data = request.json
    feature =np.array([[
        data['no_of_dependents'],
        data['education'],
        data['self_employed'], 
        data['income_annum'], 
        data['loan_amount'],
        data['loan_term'], 
        data['cibil_score'],
        data['residential_assets_value'],
        data['commercial_assets_value'],
        data['luxury_assets_value'],
        data['bank_asset_value'], 
    ]])
    prediction = model.predict(feature)
    probability = model.predict_proba(feature) # this will give the confidence score for each class
    result ='Loan Approved'
    if prediction[0] == 0:
        result = 'Loan Rejected'
    confidence_score = round(np.max(probability)*100, 2) # get the maximum probability and round it to 2 decimal places
    return jsonify({
        'prediction': result,
        'confidence': confidence_score
    })
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )