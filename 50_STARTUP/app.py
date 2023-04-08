import gradio as gr
import pandas as pd
import numpy as np
from joblib import load

def predict_profit(
    RandDSpend,Administration,MarketingSpend,State
):
    model=load("startup.jb")

    # Create dict array from parameters
    data={
        "RandDSpend":[RandDSpend],
        "Administration":[Administration],
        "MarketingSpend":[MarketingSpend],
        "State":[State]
    }

    xin=pd.DataFrame(data)
    Profit=model.predict(xin)
    return Profit[0]

ui=gr.Interface(
    fn=predict_profit,
    inputs=[
        gr.inputs.Textbox(placeholder="R&D Amount",numeric=True,label="R&D SPEND"),
        gr.inputs.Textbox(placeholder="Administration Amount",numeric=True,label="ADMINISTRATION"),
        gr.inputs.Textbox(placeholder="Marketing Amount",numeric=True,label="MARKETING SPEND"),
        gr.Dropdown(["New York","California","Florida"],label="STATE"),
    ],

    title="STARTUP PROFIT PREDICTOR",
    outputs="text",
    examples=[[165349.2,136897.8,471784.1,"New York"],
    [67532.53,105751.03,304768.73,"Florida"],
    [64664.71,139553.16,137962.62,"California"]]

)

if __name__=="__main__":
    ui.launch()
