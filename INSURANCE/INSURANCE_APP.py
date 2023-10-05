from pickle import FALSE
import gradio as gr
from joblib import load 
import numpy as np
import pandas as pd
from traitlets import default

def predict_expenses(
    age,sex,bmi,
    children,smoker,region
    ):

    # Load the model
    model=load("insurancepredict.jb")

    # Create dict array from parameters
    data={
        'age':[age],
        'sex':[sex],
        'bmi':[bmi],
        'children':[children],
        'smoker':[smoker],
        'region':[region],
    }

    xinp=pd.DataFrame(data)
    # print(xinp)

    expenses=model.predict(xinp)
    return expenses[0]

ui=gr.Interface(
    fn=predict_expenses,
    inputs=[
        gr.inputs.Textbox(placeholder="Age",default=20,numeric=True,label="AGE"),
        gr.Radio(['male','female'],label="GENDER"),
        gr.inputs.Textbox(placeholder="BMI",default=25,numeric=True,label="BMI"),
        gr.inputs.Textbox(placeholder="Childrens",default=2,label="CHILDRENS"),
        gr.Radio(['yes','no'],label="SMOKER"),
        gr.Dropdown(['southwest','southeast','northwest','northeast'],label="REGION"),

    ],
    title="INSURANCE PREDICTOR",

    outputs="text",

    examples=[[19,"female",27.9,0,"yes","southwest",16884.92],
    [61,"male",36.3,1,"yes","southwest",47403.88]],
    
    # theme="darkdefault",

    css= """body {body-color: blue}"""

)

if __name__=="__main__":
    ui.launch(share=True)


