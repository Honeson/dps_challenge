from pyexpat import model
from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import uvicorn
import pandas as pd
from sklearn.preprocessing import LabelEncoder


app = FastAPI()

class DataInput(BaseModel):
    MONATSZAHL: str
    AUSPRAEGUNG: str
    JAHR: int
    Month: int


def load_model():
    model = joblib.load('models/random_forest.joblib')
    return model

def prepare_new_data(df):
    vars_cat = [col for col in df.columns if df[col].dtypes=='O']
    #vars_cat = ['MONATSZAHL', 'AUSPRAEGUNG']
    labelencoder = LabelEncoder()
    for var in vars_cat:
    
        df[var] = labelencoder.fit_transform(df[var])
    
    print(df.describe())
    print(df.head())
    return df

@app.post('/predict')
async def predict(data: DataInput):
    input_data = []
    MONATSZAHL = data.MONATSZAHL
    AUSPRAEGUNG = data.AUSPRAEGUNG
    JAHR = data.JAHR
    Month = data.Month
    input_data.append([ MONATSZAHL, AUSPRAEGUNG, JAHR, Month])
    df = pd.DataFrame(input_data, columns=['MONATSZAHL', 'AUSPRAEGUNG', 'JAHR', 'Month'])
    print(df)
    model = load_model()
    value = model.predict(prepare_new_data(df))
    return {'Value': value[0]}



