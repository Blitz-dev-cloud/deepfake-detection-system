from pydantic import BaseModel

class PredictionResponse(BaseModel):
    prediction: str
    probability: float

class VideoPredictionResponse(BaseModel):
    prediction: str
    probability: float
    frames_analyzed: int
