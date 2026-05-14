from pydantic import BaseModel
from typing import List


class ClassProbability(BaseModel):
    index: int
    class_name: str
    probability: float


class PredictionResponse(BaseModel):
    predicted_index: int
    predicted_class: str
    confidence: float
    probabilities: List[ClassProbability]
