from importlib.abc import Loader
from pickle import TRUE
from unittest import loader
import pandas as pd 
import numpy as np
import gradio as gr
# from yaml import load 
from joblib import load

def predict_price(
    carat,cut,color,clarity,depth,table,x,y,z
):
    model=load("DIAMONDS_\diamonds.jb")
    # Create dict array from parameters
    data={ 
        "carat":[carat],
        "cut":[cut],
        "color":[color],
        "clarity":[clarity],
        "depth":[depth],
        "table":[table],
        "x":[x],
        "y":[y],
        "z":[z]
    }

    xin=pd.DataFrame(data)
    price=model.predict(xin)
    return price[0]

ui=gr.Interface( 
    fn=predict_price,
    inputs=[ 
        gr.inputs.Textbox(placeholder="carat",numeric=True,label="CARAT"),
        gr.inputs.Textbox(placeholder="cut",label="CUT"),
        gr.inputs.Textbox(placeholder="color",label="COLOR"),
        gr.inputs.Textbox(placeholder="clarity",label="CLARITY"),
        gr.inputs.Textbox(placeholder="depth",numeric=True,label="DEPTH"),
        gr.inputs.Textbox(placeholder="table",numeric=True,label="TABLE"),
        gr.inputs.Textbox(placeholder="x",numeric=True,label="X"),
        gr.inputs.Textbox(placeholder="y",numeric=True,label="Y"),
        gr.inputs.Textbox(placeholder="z",numeric=True,label="Z"),
    ],

    title="DIAMONDS PRICE PREDICTOR",
    outputs="text",
    examples=[[0.23,"Ideal","E","SI2",61.5,55,3.95,3.98,2.43],
    [0.23,"Good","E","VS1",56.9,65,4.05,4.07,2.31]]
)

if __name__=="__main__":
    ui.launch()