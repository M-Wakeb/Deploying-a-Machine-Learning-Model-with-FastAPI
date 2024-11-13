from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field
import pickle
import os
from ml.data import process_data
from ml.model import inference
import pandas as pd

# Initialize API object
app = FastAPI()

# Load model and encoders
file_dir = os.path.dirname(__file__)
model_path = os.path.join(file_dir, 'model/rf_model.pkl')
encoder_path = os.path.join(file_dir, 'model/encoder.pkl')
lb_path = os.path.join(file_dir, 'model/lb.pkl')

# Check if the model files exist
print(f"Model path: {model_path}")
print(f"Encoder path: {encoder_path}")
print(f"Label binarizer path: {lb_path}")

# Load the model, encoder, and label binarizer
model = pickle.load(open(model_path, 'rb'))
encoder = pickle.load(open(encoder_path, 'rb'))
lb = pickle.load(open(lb_path, 'rb'))


# Define the input data schema
class InputData(BaseModel):
    age: int = Field(None, example=39)
    workclass: str = Field(None, example='State-gov')
    fnlgt: int = Field(None, example=77516)
    education: str = Field(None, example='Bachelors')
    education_num: int = Field(None, example=13)
    marital_status: str = Field(None, example='Never-married')
    occupation: str = Field(None, example='Adm-clerical')
    relationship: str = Field(None, example='Not-in-family')
    race: str = Field(None, example='White')
    sex: str = Field(None, example='Female')
    capital_gain: int = Field(None, example=2174)
    capital_loss: int = Field(None, example=0)
    hours_per_week: int = Field(None, example=40)
    native_country: str = Field(None, example='United-States')

# Welcome endpoint


@app.get('/')
async def welcome():
    return "Welcome to the model prediction API!"

# Prediction endpoint


@app.post('/predict')
async def predict(data: InputData):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Convert the input data to a DataFrame
    sample = {key.replace('_', '-'): [value]
              for key, value in data.dict().items()}
    input_data = pd.DataFrame.from_dict(sample)

    # Process the input data
    X, _, _, _ = process_data(
        input_data,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Make the prediction
    output = inference(model=model, X=X)[0]
    str_out = '<=50K' if output == 0 else '>50K'

    # Return the prediction result
    return {"pred": str_out}

# Run the application using uvicorn
if __name__ == '__main__':
    uvicorn.run("main:app", reload=True, log_level="info")
