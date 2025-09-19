from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Cargar modelo y scaler
scaler = joblib.load("scaler.pkl")
model = joblib.load("modelo_clima.pkl")

app = FastAPI(title="Predicción de Clima (Clustering)")

# Definir datos de entrada
class InputData(BaseModel):
    temperatura_promedio_A: float
    precipitacion_promedio_A: float

@app.get("/")
def home():
    return {"message": "Bienvenido a la API de clustering de clima"}

@app.post("/predict/")
def predict(data: InputData):
    x_new = pd.DataFrame([[data.temperatura_promedio_A, data.precipitacion_promedio_A]],
                         columns=["temperatura_promedio_A", "precipitación_promedio_A"])
    
    # Escalar igual que en el entrenamiento
    x_scaled = scaler.transform(x_new)
    
    # Predecir cluster
    cluster = model.predict(x_scaled)
    return {"cluster": int(cluster[0])}
