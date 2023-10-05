from unicodedata import numeric
import gradio as gr
import pandas as pd
import numpy as np
from joblib import load

def predict_price(
    Mendacium,Depth
):

    model=load("oil_predictor.jb")

    # Create dict array from parameters
    data={
       "Mendacium":[Mendacium],
       "Depth":[Depth]
    }

    xin=pd.DataFrame(data)
    price=model.predict(xin)
    return price[0]

ui=gr.Interface(
    fn=predict_price,
    inputs=[ 
        gr.inputs.Textbox(placeholder="Mendacium",numeric=True,label="MENDACIUM"),
        gr.inputs.Textbox(placeholder="Depth",numeric=True,label="DEPTH")
    ],

    title="OIL PRICE PREDICTOR",
    outputs="text",
    examples=[[3.681,1958.027],[5.21,951.957],[11.612,2008.463]]
)

if __name__=="__main__":
    ui.launch(share=True)
 