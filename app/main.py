from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas import PredictionResponse


app = FastAPI(
    title="Derm Foundation Classifier API",
    description="Derm Foundation embedding backbone + PyTorch MLP head.",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.cors_origins.split(",")],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.state.predictor = None


def get_predictor():
    if app.state.predictor is None:
        print("Loading TwoStageDermPredictor...", flush=True)

        from app.services.predictor import TwoStageDermPredictor

        app.state.predictor = TwoStageDermPredictor(
            derm_model_id=settings.derm_model_id,
            head_checkpoint_path=str(settings.head_checkpoint_path),
            hf_token=settings.hf_token,
            local_files_only=settings.local_files_only,
            image_size=settings.image_size,
            device_name=settings.device,
        )

        print("TwoStageDermPredictor loaded.", flush=True)

    return app.state.predictor


@app.get("/")
def root():
    return {"message": "Derm Foundation API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type is not None and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        predictor = get_predictor()
        return predictor.predict(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc