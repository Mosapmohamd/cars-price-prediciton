from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

# load model once
cat = pickle.load(open("catboost.pkl", "rb"))

app = FastAPI(title="Car Price Prediction API")

# input schema
class CarInput(BaseModel):
    year: int
    make: str
    model: str
    level: str
    transmission: str
    cylinders: int
    disp_liters: float
    odometer: int
    drivetrain: str
    fuel_type: str
    disclosures: bool
    as_is: bool

def price_with_dynamic_range(predicted_price: float):
    if predicted_price <= 20000:
        delta = 500
    elif predicted_price <= 40000:
        delta = 1000
    else:
        delta = 1500

    return {
        "predicted_price": round(predicted_price),
        "price_range": f"{round(predicted_price - delta)} - {round(predicted_price + delta)}",
        "delta": delta
    }

@app.post("/predict")
def predict_price(data: CarInput):
    manual_df = pd.DataFrame([data.dict()])

    manual_df = manual_df.reindex(
        columns=cat.feature_names_,
        fill_value=0
    )

    predicted_price = cat.predict(manual_df)[0]

    return price_with_dynamic_range(predicted_price)
