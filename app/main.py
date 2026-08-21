import io
import os
import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
import torch
from ultralytics import YOLO

MODEL_PATH = os.getenv("MODEL_PATH", "yolo26n.pt")
DEVICE = os.getenv("DEVICE", "cpu")
executor = ThreadPoolExecutor(max_workers=4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = YOLO(MODEL_PATH)
    yield
    executor.shutdown(wait=True)
    app.state.model = None


app = FastAPI(title="YOLO26n Inference API", version="1.0.0", lifespan=lifespan)


def infer(image: Image.Image):
    """Run blocking Ultralytics inference outside FastAPI's event loop."""
    with torch.inference_mode():
        return app.state.model.predict(source=image, device=DEVICE, verbose=False)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_PATH, "device": DEVICE}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file.")

    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from error

    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(executor, infer, image)
    detections: list[dict[str, Any]] = []

    for result in results:
        names = result.names
        for box in result.boxes:
            class_id = int(box.cls.item())
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": names[class_id],
                    "confidence": float(box.conf.item()),
                    "box": [round(value, 2) for value in box.xyxy[0].tolist()],
                }
            )

    return {"filename": file.filename, "detections": detections}
