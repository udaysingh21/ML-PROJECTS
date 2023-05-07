from ast import dump
from cProfile import label
from logging import PlaceHolder
from operator import mod
import pandas as pd
import numpy as np
import gradio as gr
from joblib import load,dump

def purchase(
    UserID,Gender,Age,EstimatedSalary
):

    model=load("purchased.jb")

    data={
        "UserID":[UserID],
        "Gender":[Gender],
        "Age":[Age],
        "EstimatedSalary":[EstimatedSalary]
    }

    xin=pd.DataFrame(data)
    purchased=model.predict(xin)
    return purchased[0]

ui=gr.Interface(
    fn=purchase,
    inputs=[
        gr.inputs.Textbox(placeholder="user_id",numeric=True,label="USER ID"),
        gr.Radio(["Male","Female"],label="GENDER"),
        gr.inputs.Textbox(placeholder="age",numeric=True,label="AGE"),
        gr.inputs.Textbox(placeholder="estimated_salary",numeric=True,label="ESTIMATED SALARY"),
    ],

    title="PURCHASED OR NOT ?",
    outputs="text",
    examples=[[15624510,"Male",19,19000,0],
    [15694829,"Female",32,150000,1
]]

)

if __name__=="__main__":
    ui.launch()