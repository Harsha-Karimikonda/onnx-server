
# 1) Go 
FROM golang:1.22-alpine AS go_builder
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o /out/server ./main.go

# 2) Python 
FROM python:3.11-slim AS py_builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*
COPY main.py .
COPY efficientnet-lite4-11-int8.onnx .
COPY static ./static
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" onnxruntime pillow numpy requests pydantic prometheus-client

# 3) Final 
FROM debian:stable-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates python3 python3-distutils python3-venv && \
    rm -rf /var/lib/apt/lists/*
COPY --from=go_builder /out/server /app/server
COPY --from=py_builder /app /app

EXPOSE 8080 8000

CMD ["bash", "-lc", "echo 'Built image with Go server and Python app artifacts' && sleep infinity"]
