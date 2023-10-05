from cProfile import label
from logging import PlaceHolder
from operator import mod
import gradio as gr
import pandas as pd
import numpy as np
from joblib import load 

def predict_bodymass(
    FlipperLength,
):
    model=load("penguin_predictor.jb")

    # Create dict array from parameters
    data={
        "FlipperLength":[FlipperLength]
    }

    xin=pd.DataFrame(data)
    bodymass=model.predict(xin)
    return bodymass[0]

ui=gr.Interface(
    fn=predict_bodymass,
    inputs=[
        gr.inputs.Textbox(placeholder="Enter Flipper Length(mm)",numeric=True,label="FLIPPER LENGTH")
    ],

    title="PENGUIN REGRESSION",
    outputs="text",
    examples=[[195],
    [183]]

)

if __name__=="__main__":
    ui.launch(share=True)