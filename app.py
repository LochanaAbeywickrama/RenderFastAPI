import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from typing import Union


# Load dataset
data = pd.read_csv('phishing.csv')

# Preprocess data 
data['attack_type'] = data['attack_type'].map({"phishing": 1, "legitimate": 0})

X = data.drop('attack_type', axis=1)  # Features
y = data['attack_type']               # Target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'phishing_model.pkl')

# Load the trained model
model = joblib.load('phishing_model.pkl')

# Initialize FastAPI app
app = FastAPI()

# Define the input data structure for a single URL
class URLData(BaseModel):
    urllength: int
    atsymbol: int
    ipaddress: int
    extfavicon: int
    popupwindow: int

# Define the input data structure for a batch of URLs
class URLBatchRequest(BaseModel):
    urls: List[URLData]

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.put("/items/{item_id}")
def update_item(item_id: int, item: URLData):
    return {"item_name": item.urllength, "item_id": item_id}

# Define a prediction endpoint for batch input
@app.post("/predict/")
def predict(batch_request: URLBatchRequest):
    # Prepare the data for batch prediction
    input_data = [[url.urllength, url.atsymbol, url.ipaddress, url.extfavicon, url.popupwindow] for url in batch_request.urls]
    
    # Get predictions for the batch
    predictions = model.predict(input_data)
    
    # Prepare the response
    results = []
    for i, prediction in enumerate(predictions):
        result = {
            "url_index": i,
            "prediction": "phishing" if prediction == 1 else "legitimate"
        }
        results.append(result)
    
    return {"predictions": results}
