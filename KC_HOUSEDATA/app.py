import importlib
from tkinter import Label
import gradio as gd 
import pandas as pd
import numpy as np
from joblib import load

def predict_price(
    id,date,bedrooms,bathrooms,sqft_living,
    sqft_lot,floors,waterfront,view,condition,
    grade,sqft_above,sqft_basement,yr_built,yr_renovated,
    zipcode,lat,long,sqft_living15,sqft_lot15
):
    model=load("housedata.jb")

    # Create dict array from parameters
    data={
        'id':[id],
        'date':[date],
        'bedrooms':[bedrooms],
        'bathrooms':[bathrooms],
        'sqft_living':[sqft_living],
        'sqft_lot':[sqft_lot],
        'floors':[floors],
        'waterfront':[waterfront],
        'view':[view],
        'condition':[condition],
        'grade':[grade],
        'sqft_above':[sqft_above],
        'sqft_basement':[sqft_basement],
        'yr_built':[yr_built],
        'yr_renovated':[yr_renovated],
        'zipcode':[zipcode],
        'lat':[lat],
        'long':[long],
        'sqft_living15':[sqft_living15],
        'sqft_lot15':[sqft_lot15]
    }

    xin=pd.DataFrame(data)
    Price=model.predict(xin)
    return Price[0]

ui=gd.Interface(
    fn=predict_price,
    inputs=[ 
        gd.inputs.Textbox(placeholder="id",label="ID"),
        gd.inputs.Textbox(placeholder="date",label="DATE"),
        gd.inputs.Textbox(placeholder="bedrooms",numeric=True,label="BEDROOMS"),
        gd.inputs.Textbox(placeholder="bathrooms",numeric=True,label="BATHROOMS"),
        gd.inputs.Textbox(placeholder="sqft_living",numeric=True,label="SQFT_LIVING"),
        gd.inputs.Textbox(placeholder="sqft_lot",numeric=True,label="SQFT_LOT"),
        gd.Dropdown([1. , 2. , 1.5, 3. , 2.5, 3.5],label="FLOORS"),
        gd.Dropdown([0,1],label="WATERFRONT"),
        gd.Dropdown([0, 1, 2, 3, 4],label="VIEW"),
        gd.Dropdown([1,2,3,4,5],label="CONDITION"),
        gd.inputs.Textbox(placeholder="grade",numeric=True,label="GRADE"),
        gd.inputs.Textbox(placeholder="sqft_above",numeric=True,label="SQFT_ABOVE"),
        gd.inputs.Textbox(placeholder="sqft_basement",numeric=True,label="SQFT_BASEMENT"),
        gd.inputs.Textbox(placeholder="yr_built",numeric=True,label="YR_BUILT"),
        gd.inputs.Textbox(placeholder="yr_renovated",numeric=True,label="YR_RENOVATED"),
        gd.inputs.Textbox(placeholder="zipcode",label="ZIPCODE"),
        gd.inputs.Textbox(placeholder="lat",numeric=True,label="LATITUDE"),
        gd.inputs.Textbox(placeholder="long",numeric=True,label="LONGITUDE"),
        gd.inputs.Textbox(placeholder="sqft_living15",numeric=True,label="SQFT_LIVING15"),
        gd.inputs.Textbox(placeholder="sqft_lot15",numeric=True,label="SQFT_LOT15"),
    ],

    title="HOUSE PRICE PREDICTOR",
    outputs="text",
    examples=[["7129300520","20141013T000000",3,1,1180,5650,"1",0,0,3,7,1180,0,1955,0,"98178",47.5112,-122.257,1340,5650],
    ["9297300055","20150124T000000",4,3,2950,5000,"2",0,3,3,9,1980,970,1979,0,"98126",47.5714,-122.375,2140,4000],
    ["0065000400","20141022T000000",4,3,1490,6766,"1.5",0,1,5,7,1490,0,1915,0,"98136",47.5446,-122.382,1990,6526]]
)

if __name__=="__main__":
    ui.launch()