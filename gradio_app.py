import gradio as gr
import joblib
import numpy as np
model = joblib.load('loan_approval_pipeline.pkl')
def predict_loan(
    no_of_dependents,
    education,
    self_employed,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value
):
    data=np.array([[
        no_of_dependents,
        education,
        self_employed,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value

    ]])
    prediction_loan  = model.predict(data)
    probability_loan = model.predict_proba(data)
    confidence = round(np.max(probability_loan)*100, 2)
    if prediction_loan[0] == 1:
        return f'Loan Approved with confidence {confidence}%'
    else:
        return f'Loan Rejected with confidence {confidence}%'
interface = gr.Interface(
    fn= predict_loan,
    inputs=[gr.Number(label='no of dependents'),
            gr.Number(label='education'),
            gr.Number(label='self-employed'),
            gr.Number(label='income_annum'),
            gr.Number(label='loan_amount'),
            gr.Number(label='loan_term'),
            gr.Number(label='cibil_score'),
            gr.Number(label='residential_assets_value'),
            gr.Number(label='commercial_assets_value'),
            gr.Number(label='luxury_assets_value'),
            gr.Number(label='bank_asset_value')],
    outputs= [gr.Textbox(label='Predicted Loan staus')],
    title='Loan Approval Prediction using Gradio'
    
) 

interface.launch()
