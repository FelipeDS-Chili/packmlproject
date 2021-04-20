from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import json
import pandas as pd
from proyecto.data import get_data, clean_df, DIST_ARGS
import numpy as np





app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)



@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.get("/predict_fare")
def predict(key, pickup_datetime , pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count):

    model = joblib.load('model.joblib')

    X_pred = pd.DataFrame([{
                  "key": '2013-07-06 17:18:00.000000119',
                  "pickup_datetime": pickup_datetime,
                  "pickup_longitude": float(pickup_longitude),
                  "pickup_latitude": float(pickup_latitude),
                  "dropoff_longitude": float(dropoff_longitude),
                  "dropoff_latitude": float(dropoff_latitude),
                  "passenger_count": float(passenger_count)
                }])

#
    return { 'prediction': int(model.predict(X_pred.iloc[[0]])) }


