// Upload model handler
const uploadBtn = document.getElementById('uploadBtn');
if (uploadBtn) {
    uploadBtn.addEventListener('click', () => {
        const modelInput = document.getElementById('modelFile');
        const labelsInput = document.getElementById('labelsFile');
        const uploadResult = document.getElementById('uploadResult');

        if (!modelInput.files.length) {
            uploadResult.textContent = 'Please choose an ONNX model file.';
            return;
        }

        const formData = new FormData();
        formData.append('model', modelInput.files[0]);
        if (labelsInput && labelsInput.files.length) {
            formData.append('labels', labelsInput.files[0]);
        }

        uploadResult.textContent = 'Uploading...';
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(async resp => {
            if (resp.ok) {
                const data = await resp.json();
                uploadResult.textContent = `Upload successful: ${data.model}`;
            } else {
                const txt = await resp.text();
                uploadResult.textContent = `Upload failed: ${txt}`;
            }
        })
        .catch(err => {
            uploadResult.textContent = `Error: ${err}`;
        });
    });
}

document.getElementById('predictBtn').addEventListener('click', () => {
    const imageUrl = document.getElementById('imageUrl').value;
    const resultDiv = document.getElementById('result');
    const imagePreview = document.getElementById('imagePreview');

    // Show image immediately
    imagePreview.src = imageUrl;
    // Reset previous result and show loading state
    resultDiv.textContent = 'Predicting...';

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image_url: imageUrl })
    })
    .then(async response => {
        if (response.ok) {
            // Successful JSON response
            const data = await response.json();
            resultDiv.innerHTML = `Predicted: ${data.predicted_label} <br> Confidence: ${data.confidence}`;
            imagePreview.src = imageUrl;
        } else {
            // Error response (likely plain text)
            const errText = await response.text();
            resultDiv.innerHTML = `Error: ${errText}`;
        }
    })
    .catch(error => {
        resultDiv.innerHTML = `Fetch error: ${error}`;
    });
});