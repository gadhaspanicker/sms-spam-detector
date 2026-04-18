from flask import Flask, render_template, request
import pickle
import re
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize app
app = Flask(__name__)

# Load trained model
model = load_model("model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Max length (same as training)
max_len = 100


# 🔹 Text Cleaning Function (same as training)
def clean(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 🔹 Home Route
@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        message = request.form.get("message")

        if message:
            # Clean text
            text = clean(message)

            # Convert to sequence
            seq = tokenizer.texts_to_sequences([text])

            # Padding
            padded = pad_sequences(seq, maxlen=max_len)

            # Prediction
            pred = model.predict(padded)[0][0]

            # Result with confidence
            if pred > 0.5:
                result = f"Spam 🚫 ({pred*100:.2f}% confidence)"
            else:
                result = f"Not Spam ✅ ({(1-pred)*100:.2f}% confidence)"

    return render_template("index.html", result=result)


# 🔹 Run App
if __name__ == "__main__":
    app.run(debug=True)