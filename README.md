# Image Prediction API

This project provides a web service for image prediction using a custom ONNX model. It's built with a Go backend and a simple HTML/JavaScript frontend.

## Features

-   **Go Backend**: A robust backend serves the frontend, handles API requests, and manages the prediction model.
-   **Python Prediction Engine**: A Python script leverages `onnxruntime` for efficient inference.
-   **Custom Model Upload**: Users can upload their own ONNX models and corresponding label files.
-   **Web Interface**: A simple UI for predicting from an image URL and uploading models.
-   **Caching**: In-memory caching for repeated prediction requests to improve performance.
-   **Prometheus Metrics**: Exposes metrics for monitoring requests and duration.

## Project Structure

```
.
├── main.go               # Go backend server
├── train.py              # Python prediction script
├── gpu_util.py           # GPU utility script
├── static/
│   ├── index.html        # Frontend HTML
│   ├── script.js         # Frontend JavaScript
│   └── style.css         # Frontend CSS
└── README.md             # This file
```

## Getting Started

### Prerequisites

-   Go
-   Python 3
-   `onnxruntime`, `numpy`, `Pillow`, and `requests` Python packages.

### Installation

1.  Clone the repository.
2.  Install Python dependencies:
    ```bash
    pip install onnxruntime numpy Pillow requests
    ```
3.  Download the default ONNX model (if not present):
    The application uses `efficientnet-lite4-11-int8.onnx` by default. You can download it or use your own.

### Running the Application

1.  Start the backend server:
    ```bash
    go run main.go
    ```
2.  The server will start on port `8080`.
3.  Open your browser and navigate to `http://localhost:8080`.

## How to Use

### Web Interface

-   **Predict**: Enter an image URL in the input box and click "Predict". The predicted label and confidence score will be displayed.
-   **Upload Model**:
    1.  Click "Choose File" to select an `.onnx` model file.
    2.  Optionally, select a JSON file with labels.
    3.  Click "Upload Model". The new model will be used for subsequent predictions.

### API Endpoints

#### `POST /predict`

Performs a prediction on an image.

-   **Request Body**:
    ```json
    {
      "image_url": "https://path/to/your/image.jpg"
    }
    ```
-   **Success Response (200)**:
    ```json
    {
      "predicted_label": "...",
      "confidence": 0.95
    }
    ```

#### `POST /upload`

Uploads a new ONNX model and labels file.

-   **Request Body**: `multipart/form-data` with fields:
    -   `model`: The `.onnx` model file.
    -   `labels` (optional): A JSON file mapping class indices to labels.
-   **Success Response (200)**:
    ```json
    {
      "status": "success",
      "model": "uploaded_model.onnx",
      "labels": "uploaded_labels.json"
    }
    ```

#### `GET /metrics`

Exposes Prometheus metrics for monitoring.

## Monitoring

The application exposes Prometheus metrics at the `/metrics` endpoint. You can use a Prometheus instance to scrape these metrics for monitoring and alerting.

-   `http_requests_total`: Total number of HTTP requests.
-   `http_request_duration_seconds`: Duration of HTTP requests.