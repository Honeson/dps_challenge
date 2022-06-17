from pyexpat import model
from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import uvicorn
import pandas as pd


app = FastAPI()

class DataInput(BaseModel):
    JAHR: int
    Month: int
    MONATSZAHL: str
    AUSPRAEGUNG: str


def load_model():
    model = joblib.load('models/random_forest.joblib')
    return model

def prepare_new_data(df):
    vars_cat = [col for col in df.columns if df[col].dtypes=='O']
    #vars_cat = ['MONATSZAHL', 'AUSPRAEGUNG']

    for var in vars_cat:
        #df = pd.concat([df,pd.get_dummies(df[var], prefix=var, drop_first=True)], axis=1)
        #df = pd.get_dummies(df[var], prefix=var, drop_first=True)
        #df[var] = pd.get_dummies(df[var])
        df = pd.concat([df,pd.get_dummies(df[var], prefix=var, drop_first=False)], axis=1)
    
    df.drop(labels=vars_cat, axis=1, inplace=True)
    print(df.describe())
    print(df.head())
    return df

@app.post('/predict')
def predict(data: DataInput):
    input_data = []
    JAHR = data.JAHR
    Month = data.Month
    MONATSZAHL = data.MONATSZAHL
    AUSPRAEGUNG = data.AUSPRAEGUNG
    input_data.append([JAHR, Month, MONATSZAHL, AUSPRAEGUNG])
    df = pd.DataFrame(input_data, columns=['JAHR', 'Month', 'MONATSZAHL', 'AUSPRAEGUNG'])
    model = load_model()
    value = model.predict(prepare_new_data(df))
    return {'Value': value}



