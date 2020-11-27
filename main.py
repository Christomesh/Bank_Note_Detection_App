from os import name
from fastapi import FastAPI

import pandas as pd
import numpy as np
import pickle

from banknote import BankNote

# create app object
app = FastAPI()

# load my classifier
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)
pickle_in.close()
@app.get('/')
async def index():
    return {"Welcome": "To my BankNotes App"}

@app.get('/{name}')
async def get_name(name: str):
    return {"Welcome": f"{name}"}

@app.post('/predict')
async def predict_banknote(data: BankNote):
    data = data.dict()
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    if (prediction[0] > 0.5):
        prediction = "Fake Note"
    else:
        prediction = "Its a Bank note"
    return {
        "prediction": prediction
    }
