# Image Prediction Web Service

This project is a web-based image prediction service. It provides a simple interface to get predictions for an image from a URL using a machine learning model. The application also allows users to upload their own custom ONNX models and label files.

The backend is built with Go, and the machine learning inference is handled by a Python script using ONNX Runtime.

## Features

- **Web Interface:** A simple frontend to enter an image URL and view the prediction result.
- **Custom Model Upload:** Users can upload their own `.onnx` model and an associated `json` file for labels.
- **Go Backend:** A robust web server that handles requests, serves the frontend, and manages the prediction pipeline.
- **Python Inference:** Utilizes a Python script with ONNX Runtime for efficient model inference.
- **Prediction Caching:** Caches results for previously seen image URLs to provide faster responses.
- **Prometheus Metrics:** Exposes a `/metrics` endpoint for monitoring request counts and latency.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

You will need the following installed on your system:

- **Go:** [Installation Guide](https://golang.org/doc/install)
- **Python 3:** [Installation Guide](https://www.python.org/downloads/)
- **Pip** (Python package installer)

### Installation & Running

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Install Python dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install onnxruntime numpy Pillow requests
    ```

3.  **Run the backend server:**
    ```bash
    go run main.go
    ```
    The server will start on port `8080` by default.

4.  **Access the application:**
    Open your web browser and navigate to `http://localhost:8080`.

## Usage

### Via the Web UI

1.  **Predicting with the default model:**
    -   Enter a valid image URL into the input field.
    -   Click the "Predict" button.
    -   The predicted label, confidence score, and a preview of the image will be displayed.

2.  **Uploading a custom model:**
    -   Click the "Choose File" button for the model and select your `.onnx` file.
    -   Click the "Choose File" button for the labels and select your `.json` labels file.
    -   Click "Upload Model".
    -   Once uploaded, all subsequent predictions will use your custom model.

### Via the API

You can also interact with the service programmatically.

-   **POST /predict**
    Sends an image URL for prediction.

    **Request Body:**
    ```json
    {
        "image_url": "https://path.to/your/image.jpg"
    }
    ```

    **Success Response:**
    ```json
    {
        "predicted_label": "your_label",
        "confidence": 0.95
    }
    ```

-   **POST /upload**
    Uploads a new model and labels file using a multipart/form-data request.

    **Form Fields:**
    -   `model`: The `.onnx` model file.
    -   `labels`: The `.json` labels file.

## Built With

- [Go](https://golang.org/) - Backend Language
- [Python](https://www.python.org/) - Machine Learning & Inference
- [ONNX Runtime](https://onnxruntime.ai/) - High-performance inference engine
- [Prometheus](https://prometheus.io/) - Monitoring and alerting