# AI-Powered SMS Spam Detection using Bidirectional LSTM

## Overview

This project implements a Natural Language Processing (NLP) system to classify SMS messages as spam or not spam using a Bidirectional Long Short-Term Memory (BiLSTM) model. It includes a complete pipeline from text preprocessing to model inference through a Flask-based web application.

---

## Features

* Text preprocessing and normalization
* Tokenization and sequence padding
* Bidirectional LSTM-based deep learning model
* Binary classification (Spam / Not Spam)
* Web interface built with Flask

---

## Technology Stack

* Python
* TensorFlow / Keras
* Scikit-learn
* Flask
* HTML and CSS

---

## Methodology

1. Clean and preprocess raw text data
2. Convert text into sequences using a tokenizer
3. Apply padding to ensure uniform input length
4. Train a Bidirectional LSTM model on processed data
5. Use the trained model for real-time classification

---

## Model Details

* Architecture: Bidirectional LSTM
* Embedding layer for word representation
* Dropout for regularization
* Sigmoid activation for binary classification

---

## Application Interface

![App Screenshot](https://raw.githubusercontent.com/gadhaspanicker/sms-spam-detector/main/screenshot.png)

---

## How to Run

1. Clone the repository:

```id="runc01"
git clone https://github.com/gadhaspanicker/sms-spam-detector.git
```

2. Navigate to the project directory:

```id="runc02"
cd sms-spam-detector
```

3. Install dependencies:

```id="runc03"
pip install -r requirements.txt
```

4. Run the application:

```id="runc04"
python app.py
```

5. Open in browser:

```id="runc05"
http://127.0.0.1:5000/
```

---

## Project Structure

```id="str01"
sms-spam-detector/
│── app.py
│── model.h5
│── tokenizer.pkl
│── requirements.txt
│── templates/
│     └── index.html
│── screenshot.png
```

---

## Future Enhancements

* Improve model performance with pretrained embeddings
* Deploy as a REST API
* Enhance UI/UX design
* Extend to multi-language spam detection

---

## Author

Gadha S Panicker

---

## Summary

This project demonstrates the application of NLP and deep learning techniques using Bidirectional LSTM to build an effective SMS spam classification system with a functional web interface.
