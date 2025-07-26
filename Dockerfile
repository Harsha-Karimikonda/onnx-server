# Build stage
FROM golang:1.22-alpine AS builder
WORKDIR /src
COPY . .
RUN go mod tidy && go build -o server .

FROM python:3.11-slim
WORKDIR /app

# Install required Python packages
RUN pip install --no-cache-dir onnxruntime pillow numpy requests

# Copy server binary and ML assets
COPY --from=builder /src/server .
COPY train.py .
COPY efficientnet-lite4-11-int8.onnx .
COPY --from=builder /src/static ./static

ENV PORT=8080
EXPOSE 8080
CMD ["./server"]
