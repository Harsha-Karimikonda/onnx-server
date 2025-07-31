import onnxruntime as ort
from PIL import Image
import requests
import json
from io import BytesIO
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Request, Response
import time
from pydantic import BaseModel
import subprocess

ort.set_default_logger_severity(3)

app = FastAPI()

REQUEST_COUNT = Counter("py_http_requests_total", "Total HTTP requests", ["path", "method", "status"])
REQUEST_DURATION = Histogram("py_http_request_duration_seconds", "Request duration in seconds", ["path"])

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response: Response = await call_next(request)
    duration = time.time() - start
    path = request.url.path
    REQUEST_COUNT.labels(path=path, method=request.method, status=str(response.status_code)).inc()
    REQUEST_DURATION.labels(path=path).observe(duration)
    return response

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

try:
    response = requests.get("https://huggingface.co/datasets/huggingface/label-files/raw/main/imagenet-1k-id2label.json")
    response.raise_for_status()
    labels = response.json()
except requests.exceptions.RequestException as e:
    print(f"Error fetching labels: {e}")
    labels = {
        "0": "tench, Tinca tinca",
        "1": "goldfish, Carassius auratus",
        "2": "great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias",
    }

DEFAULT_MODEL_PATH = "efficientnet-lite4-11-int8.onnx"

def load_session(model_path: str):
    """Load an ONNX model and return an inference session."""
    try:
        return ort.InferenceSession(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading the ONNX model: {e}")

session = load_session(DEFAULT_MODEL_PATH)

def preprocess(image):
    """Preprocesses the image for EfficientNet."""
    image = image.resize((256, 256))
    left = (256 - 224) / 2
    top = (256 - 224) / 2
    right = (256 + 224) / 2
    bottom = (256 + 224) / 2
    image = image.crop((left, top, right, bottom))
    image_np = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_np = (image_np - mean) / std
    image_np = np.expand_dims(image_np, axis=0)
    return image_np

def predict(image):
    """Performs inference on a single image."""
    processed_image = preprocess(image)
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: processed_image})[0]
    predicted_class_index = np.argmax(result)
    confidence = float(np.max(result))
    predicted_label = labels[str(predicted_class_index)]
    return predicted_label, confidence

class PredictionRequest(BaseModel):
    image_url: str

@app.post("/train")
async def train_endpoint(request: PredictionRequest):
    try:
        response = requests.get(request.image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        label, confidence = predict(image)
        return {"predicted_label": label, "confidence": confidence}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error downloading the image: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")

@app.post("/predict")
async def predict_endpoint(request: PredictionRequest):
    try:
        response = requests.get(request.image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        label, confidence = predict(image)
        return {"predicted_label": label, "confidence": confidence}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error downloading the image: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")

@app.get("/gpu_util")
async def gpu_util():
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        return {"utilization": result.stdout}
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise HTTPException(status_code=500, detail=f"Error running nvidia-smi: {e}")