from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.model import predict_image
from app.utils import read_image
from app.schemas import PredictionResponse

app = FastAPI(title="Deepfake Detector API")


# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://127.0.0.1:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Deepfake Detector API is running!"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    image = read_image(await file.read())
    pred_class, prob = predict_image(image)
    return {
        "prediction": "FAKE" if pred_class == 0 else "REAL",
        "probability": round(prob, 4)
    }
