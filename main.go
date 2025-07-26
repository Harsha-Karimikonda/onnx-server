package main

import (
    "encoding/json"
    "fmt"
    "io"
    "log"
    "net/http"
    "os"
    "os/exec"
    "sync"
    "time"

    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promhttp"
)
   
   type PredictionResponse struct {
    PredictedLabel string  `json:"predicted_label"`
    Confidence     float64 `json:"confidence"`
   }
   
   type PredictionRequest struct {
    ImageURL string `json:"image_url"`
   }

// simple in-memory cache for prediction results
var (
    cache   = make(map[string][]byte)
    cacheMu sync.RWMutex

    // paths to currently active ONNX model and labels file
    currentModelPath  = "efficientnet-lite4-11-int8.onnx"
    currentLabelsPath = ""

    requestCount = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "http_requests_total",
            Help: "Total number of HTTP requests",
        }, []string{"path", "method", "status"},
    )
    requestDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "http_request_duration_seconds",
            Help:    "Duration of HTTP requests in seconds",
            Buckets: prometheus.DefBuckets,
        }, []string{"path"},
    )
)

func init() {
    prometheus.MustRegister(requestCount, requestDuration)
}
   
    func predictHandler(w http.ResponseWriter, r *http.Request) {
        if r.Method != http.MethodPost {
            http.Error(w, "Only POST method is allowed", http.StatusMethodNotAllowed)
            return
        }

        // Read the body into a byte slice so it can be read multiple times.
        body, err := io.ReadAll(r.Body)
        if err != nil {
            http.Error(w, "Failed to read request body", http.StatusInternalServerError)
            return
        }
        r.Body.Close() // Close the original body

        // Use the buffered body for all subsequent operations.
        var req PredictionRequest
        if err := json.Unmarshal(body, &req); err != nil {
            http.Error(w, "Invalid request body", http.StatusBadRequest)
            return
        }

        if req.ImageURL == "" {
            http.Error(w, "image_url is required", http.StatusBadRequest)
            return
        }

        // Check cache first
        cacheMu.RLock()
        if data, ok := cache[req.ImageURL]; ok {
            cacheMu.RUnlock()
            w.Header().Set("Content-Type", "application/json")
            w.Write(data)
            return
        }
        cacheMu.RUnlock()

                // Build command args dynamically based on uploaded model/labels
        args := []string{"train.py", req.ImageURL, currentModelPath}
        if currentLabelsPath != "" {
            args = append(args, currentLabelsPath)
        }
        cmd := exec.Command("python3", args...)
        output, err := cmd.CombinedOutput()
        if err != nil {
            log.Printf("Error executing command: %v", err)
            log.Printf("Output: %s", string(output))
            http.Error(w, fmt.Sprintf("Error executing script: %s", output), http.StatusInternalServerError)
            return
        }

        // cache the response
        cacheMu.Lock()
        cache[req.ImageURL] = output
        cacheMu.Unlock()

        w.Header().Set("Content-Type", "application/json")
        w.Write(output)
}

// logMiddleware logs incoming HTTP requests with method, path, status code and duration, and records Prometheus metrics.
func logMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        lrw := &loggingResponseWriter{ResponseWriter: w, statusCode: http.StatusOK}
        next.ServeHTTP(lrw, r)
        duration := time.Since(start)
        log.Printf("%s %s %d %s", r.Method, r.URL.Path, lrw.statusCode, duration)
        requestCount.WithLabelValues(r.URL.Path, r.Method, fmt.Sprintf("%d", lrw.statusCode)).Inc()
        requestDuration.WithLabelValues(r.URL.Path).Observe(duration.Seconds())
    })
}

type loggingResponseWriter struct {
    http.ResponseWriter
    statusCode int
}

func (lrw *loggingResponseWriter) WriteHeader(code int) {
    lrw.statusCode = code
    lrw.ResponseWriter.WriteHeader(code)
}

func corsMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    	w.Header().Set("Access-Control-Allow-Origin", "*")
    	w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
    	w.Header().Set("Access-Control-Allow-Headers", "Accept, Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization")
    	if r.Method == "OPTIONS" {
    		return
    	}
    	next.ServeHTTP(w, r)
    })
   }
   
   func rootHandler(w http.ResponseWriter, r *http.Request) {
    if r.URL.Path != "/" {
    	http.NotFound(w, r)
    	return
    }
    http.ServeFile(w, r, "static/index.html")
   }
   
   func uploadHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "Only POST method is allowed", http.StatusMethodNotAllowed)
        return
    }

    // Parse multipart form with a reasonable default limit (e.g., 100MB)
    if err := r.ParseMultipartForm(100 << 20); err != nil {
        http.Error(w, "Failed to parse form: "+err.Error(), http.StatusBadRequest)
        return
    }

    // Model file
    modelFile, modelHeader, err := r.FormFile("model")
    if err != nil {
        http.Error(w, "model file is required: "+err.Error(), http.StatusBadRequest)
        return
    }
    defer modelFile.Close()
    modelPath := "uploaded_" + modelHeader.Filename
    out, err := os.Create(modelPath)
    if err != nil {
        http.Error(w, "Failed to create file: "+err.Error(), http.StatusInternalServerError)
        return
    }
    if _, err := io.Copy(out, modelFile); err != nil {
        out.Close()
        http.Error(w, "Failed to save model: "+err.Error(), http.StatusInternalServerError)
        return
    }
    out.Close()

    // Optional labels
    labelsFile, labelsHeader, err := r.FormFile("labels")
    labelsPath := ""
    if err == nil {
        defer labelsFile.Close()
        labelsPath = "uploaded_" + labelsHeader.Filename
        lout, err := os.Create(labelsPath)
        if err != nil {
            http.Error(w, "Failed to create labels file: "+err.Error(), http.StatusInternalServerError)
            return
        }
        if _, err := io.Copy(lout, labelsFile); err != nil {
            lout.Close()
            http.Error(w, "Failed to save labels: "+err.Error(), http.StatusInternalServerError)
            return
        }
        lout.Close()
    }

    // Update global paths atomically
    cacheMu.Lock()
    currentModelPath = modelPath
    currentLabelsPath = labelsPath
    // Clear cache as predictions may differ with new model
    cache = make(map[string][]byte)
    cacheMu.Unlock()

    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(http.StatusOK)
    fmt.Fprintf(w, `{"status":"success","model":"%s","labels":"%s"}`+"\n", modelPath, labelsPath)
}

func main() {
    mux := http.NewServeMux()
    mux.Handle("/", http.HandlerFunc(rootHandler))
    mux.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("static"))))
    mux.Handle("/metrics", promhttp.Handler())
        mux.HandleFunc("/upload", uploadHandler)
    mux.HandleFunc("/predict", predictHandler)
   
    port := os.Getenv("PORT")
    if port == "" {
        port = "8080"
    }
    log.Printf("Server starting on port %s...", port)
        handlerChain := corsMiddleware(logMiddleware(mux))
    if err := http.ListenAndServe(":"+port, handlerChain); err != nil {
        log.Fatalf("Could not start server: %s\n", err)
    }
   }
