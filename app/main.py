# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.model import predict_image
from app.utils import read_image
from app.schemas import PredictionResponse

# -------------------------------
# Initialize FastAPI
# -------------------------------
app = FastAPI(title="Deepfake Detector API")

# -------------------------------
# CORS Middleware
# -------------------------------
origins = [
    "http://127.0.0.1:5500",  # your frontend
    "http://localhost:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Routes
# -------------------------------
@app.get("/")
async def root():
    return {"message": "Deepfake Detector API is running!"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    # Read image bytes and convert to model input
    image = read_image(await file.read())

    # Get prediction
    pred_class, prob = predict_image(image)

    return PredictionResponse(
        prediction="FAKE" if pred_class == 1 else "REAL",
        probability=float(prob)
    )
