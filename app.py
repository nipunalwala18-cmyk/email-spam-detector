from flask import Flask, request, jsonify, render_template
import joblib
import cv2
import pytesseract
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('model1')
vectorizer = joblib.load('model2')

# Preprocessing Functions
def preprocess_text(text):
    return text.lower().strip()

def text_to_vector(text):
    tokens = text.split()
    vectors = [vectorizer.wv[t] for t in tokens if t in vectorizer.wv]
    if vectors:
        return np.mean(vectors, axis=0).reshape(1, -1)  # fixed axis
    else:
        return np.zeros((1, vectorizer.vector_size))

def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # ensure RGB
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh)
        return text
    except Exception as e:
        return f"ERROR: {str(e)}"


# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_text", methods=["POST"])
def predict_text():
    data = request.json
    text = preprocess_text(data["text"])
    vec = text_to_vector(text)
    prediction = model.predict(vec)[0]
    confidence = np.max(model.predict_proba(vec)) * 100

    return jsonify({
        "input": text,
        "prediction": "SPAM" if prediction == 1 else "NOT SPAM",
        "confidence": f"{confidence:.2f}%"
    })

@app.route("/predict_image", methods=['POST'])
def predict_image():
    if "file" not in request.files:
        return jsonify({"error": "No File Uploaded"}), 400
    
    file = request.files["file"]
    extracted_text = preprocess_image(file.read())
    vec = text_to_vector(extracted_text)
    prediction = model.predict(vec)[0]
    confidence = np.max(model.predict_proba(vec)) * 100

    return jsonify({
        "input": extracted_text,
        "prediction": "SPAM" if prediction == 1 else "NOT SPAM",
        "confidence": f"{confidence:.2f}%"
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
