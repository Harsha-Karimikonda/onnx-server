import onnxruntime as ort

# Suppress onnxruntime warnings by setting the logger severity level to ERROR
ort.set_default_logger_severity(3)
import numpy as np
from PIL import Image
import requests
import json
from io import BytesIO
import sys


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
        print(f"Error loading the ONNX model at {model_path}: {e}")
        exit(1)

session = load_session(DEFAULT_MODEL_PATH)

def preprocess(image):
    """
    Preprocesses the image for EfficientNet.

    Args:
        image: A PIL Image object.

    Returns:
        A numpy array of the preprocessed image.
    """
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
    """
    Performs inference on a single image.

    Args:
        image: A PIL Image object.

    Returns:
        The predicted label and confidence score.
    """
    processed_image = preprocess(image)

    input_name = session.get_inputs()[0].name

    result = session.run(None, {input_name: processed_image})[0]

    predicted_class_index = np.argmax(result)

    confidence = float(np.max(result))

    predicted_label = labels[str(predicted_class_index)]

    return predicted_label, confidence

if __name__ == "__main__":
    """Usage: python train.py <image_url> [model_path] [labels_path]"""
    if len(sys.argv) < 2:
        print("Usage: python train.py <image_url> [model_path] [labels_path]")
        sys.exit(1)

    image_url = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_MODEL_PATH
    labels_path = sys.argv[3] if len(sys.argv) >= 4 else None

    session = load_session(model_path)

    if labels_path:
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = json.load(f)
        except Exception as e:
            print(f"Error loading labels file {labels_path}: {e}")
            sys.exit(1)

    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")

        label, confidence = predict(image)

        prediction = {
            "predicted_label": label,
            "confidence": confidence
        }
        print(json.dumps(prediction, indent=2))
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the image: {e}")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

