# Image Prediction Platform (Go + Python + Docker + Prometheus/Grafana)

This project provides a web service for image prediction using an ONNX model with:
- A Go service that serves the UI, proxies predictions to the Python service, supports model upload, exposes `/metrics`.
- A Python FastAPI service that performs image inference via ONNX Runtime, serves the UI and exposes `/metrics`.
- Docker Compose to run both services plus Prometheus and Grafana.

## Features

- Go backend (HTTP server, static files, `/upload`, `/train`, `/metrics`).
- Python FastAPI prediction service with endpoints `/predict`, `/train`, `/gpu_util`, `/metrics`, and UI at `/`.
- Static frontend in `static/`.
- Prometheus metrics for both services, scrape-ready via Docker Compose.
- Grafana for visualization.

## Project Structure

```
.
├── main.go                    # Go backend server (serves UI, /train proxy, /upload, /metrics)
├── main.py                    # Python FastAPI app (UI, /predict, /train, /gpu_util, /metrics)
├── train.py                   # (Optional) extra Python script referenced in Dockerfile
├── gpu_util.py                # (Optional) GPU utility
├── static/
│   ├── index.html
│   ├── script.js
│   └── style.css
├── Dockerfile                 # Multi-stage image producing Go binary and Python app artifacts
├── docker-compose.yml         # Orchestrates app_go, app_py, prometheus, grafana
└── prometheus/
    └── prometheus.yml         # Prometheus scrape configuration
```

## Quick Start (Docker Compose)

Requirements:
- Docker and Docker Compose

Build and run the full stack:
```bash
docker compose up --build
```

Services (default ports):
- Go app: http://localhost:8080 (UI at /, metrics at /metrics)
- Python app: http://localhost:8000 (UI at /, predict at /predict, metrics at /metrics)
- Prometheus: http://localhost:19090
- Grafana: http://localhost:3000 (default admin/admin unless changed)

If port 8000 is already in use (e.g., from a locally running uvicorn), stop it first:
```bash
# kill any process on port 8000
lsof -i :8000 -t | xargs -r kill -9
```
Then run `docker compose up --build` again.

## Running Locally Without Docker

Python FastAPI (requires Python 3.10+):
```bash
# using a venv
python -m venv .venv
source .venv/bin/activate
pip install fastapi "uvicorn[standard]" onnxruntime pillow numpy requests pydantic prometheus-client
uvicorn main:app --host 0.0.0.0 --port 8000
```
- UI: http://localhost:8000/
- Predict:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_url":"https://example.com/image.jpg"}'
```
- Metrics: http://localhost:8000/metrics
- GPU util (if `nvidia-smi` available): http://localhost:8000/gpu_util

Go server:
```bash
go mod tidy
go run main.go
```
- UI: http://localhost:8080/
- Train proxy: POST http://localhost:8080/train
- Upload model: POST http://localhost:8080/upload (multipart/form-data)
- Metrics: http://localhost:8080/metrics

## API Summary

Python (FastAPI):
- GET `/` -> serves UI (`static/index.html`)
- POST `/predict` -> body: `{"image_url": "<URL>"}`; returns `{"predicted_label": "...", "confidence": <float>}`
- POST `/train` -> same as `/predict` (legacy alias)
- GET `/gpu_util` -> returns `{"utilization":"<nvidia-smi output>"}` or 500 if not available
- GET `/metrics` -> Prometheus metrics

Go:
- GET `/` -> serves UI
- POST `/train` -> proxies to Python `/train` with request body
- POST `/upload` -> accepts model (required) and labels (optional) files, updates model paths
- GET `/metrics` -> Prometheus metrics

## Prometheus and Grafana

### Prometheus

*   **UI (Graph):** [http://localhost:19090/](http://localhost:19090/)
*   **Configuration:** [`prometheus/prometheus.yml`](prometheus/prometheus.yml)
*   **Scrape Targets:**
    *   `app_go:8080` (Go service)
    *   `app_py:8000` (Python service)
*   **Metrics Exposed:**
    *   Go: `http_requests_total`, `http_request_duration_seconds`
    *   Python: `py_http_requests_total`, `py_http_request_duration_seconds`

#### Querying Prometheus API for JSON

To get raw JSON data from Prometheus, you can use its HTTP API.

*   **Instant Query:**
    ```bash
    curl 'http://localhost:19090/api/v1/query?query=up'
    ```
*   **Range Query (last 5 minutes):**
    ```bash
    curl 'http://localhost:19090/api/v1/query_range?query=py_http_requests_total&start=$(date -u -d "-5 min" +%FT%TZ)&end=$(date -u +%FT%TZ)&step=15s'
    ```
*   **View All Targets:** [http://localhost:19090/api/v1/targets](http://localhost:19090/api/v1/targets)

### Grafana

*   **UI:** [http://localhost:3000](http://localhost:3000) (default login: `admin`/`admin`)
*   **Data Persistence:** Grafana's data is persisted in a Docker named volume (`grafana_data`), which is managed by Docker.

#### Connecting Grafana to Prometheus

When adding Prometheus as a data source in the Grafana UI, you must use the Docker service name, not `localhost`.

1.  Navigate to **Connections** > **Data sources**.
2.  Select **Prometheus**.
3.  For the **Prometheus server URL**, enter: `http://prometheus:9090`
4.  Click **Save & Test**. You should see a "Data source is working" confirmation.

## Model and Labels

- Default ONNX model expected by Python: `efficientnet-lite4-11-int8.onnx` in the working directory.
- Labels: Python tries to fetch ImageNet 1k labels at startup; if offline, it falls back to a tiny map (indices 0–2). For reliability, provide a full local labels file and modify the code to load it.
- Go service can upload a new model and optional labels via `/upload`.
