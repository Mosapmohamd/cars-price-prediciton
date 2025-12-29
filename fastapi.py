from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI(title="Car Price Prediction API")

# load artifacts once
encoders = pickle.load(open("label_encoders.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
cat = pickle.load(open("catboost.pkl", "rb"))


class CarInput(BaseModel):
    year: int
    model: str
    cab: str
    level: str
    extra: str
    transmission: str
    cylinders: int
    disp_liters: float
    odometer: int
    drivetrain: str
    fuel_type: str
    disclosures: bool
    as_is: bool
    make: str


def price_with_dynamic_range(predicted_price: float):
    if predicted_price <= 20000:
        delta = 500
    elif predicted_price <= 40000:
        delta = 1000
    else:
        delta = 1500

    return {
        "predicted_price": round(predicted_price),
        "price_range": f"{round(predicted_price - delta)} : {round(predicted_price + delta)}",
        "delta": delta
    }


@app.post("/predict")
def predict_price(data: CarInput):
    df = pd.DataFrame([data.dict()])

    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    num_cols = scaler.feature_names_in_
    df[num_cols] = scaler.transform(df[num_cols])

    df = df.reindex(columns=cat.feature_names_, fill_value=0)

    prediction = cat.predict(df)[0]

    return price_with_dynamic_range(prediction)
