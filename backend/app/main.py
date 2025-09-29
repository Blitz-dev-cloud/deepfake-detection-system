from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import PredictionResponse, VideoPredictionResponse
from app.utils import read_image
from app.predict import predict_image, predict_video

app = FastAPI(title="Deepfake Detector API")

# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Deepfake Detector API is running!"}

@app.post("/predict_image", response_model=PredictionResponse)
async def predict_image_endpoint(file: UploadFile = File(...)):
    image = read_image(await file.read())
    pred_class, prob = predict_image(image)
    return {
        "prediction": "FAKE" if pred_class == 0 else "REAL",
        "probability": round(prob, 4)
    }

@app.post("/predict_video", response_model=VideoPredictionResponse)
async def predict_video_endpoint(file: UploadFile = File(...)):
    video_bytes = await file.read()
    pred_class, prob, frames_analyzed = predict_video(video_bytes, frame_skip=30)
    if pred_class is None:
        return {"prediction": "ERROR", "probability": 0.0, "frames_analyzed": 0}

    return {
        "prediction": "FAKE" if pred_class == 0 else "REAL",
        "probability": round(prob, 4),
        "frames_analyzed": frames_analyzed
    }
