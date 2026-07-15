# Email Spam Detector using XGBoost

An Email Spam Detection system built using **Machine Learning** and **XGBoost** with Natural Language Processing (NLP). The application classifies email messages as **Spam** or **Not Spam (Ham)** through a simple Flask web interface.

---

## Features

- Detects spam emails using XGBoost
- Text preprocessing with NLP techniques
- Word2Vec-based feature extraction
- Interactive Flask web application
- Fast and accurate spam prediction
- Trained on SMS Spam Collection and Enron Spam datasets

---

## Tech Stack

- Python
- XGBoost
- Flask
- Scikit-learn
- Pandas
- NumPy
- NLTK
- Gensim (Word2Vec)
- HTML/CSS

---

## Project Structure

```
Email-Spam-Detector/
│
├── static/                  # CSS, JavaScript, images
├── templates/               # HTML templates
├── app.py                   # Flask application
├── train.ipynb              # Model training notebook
├── build_data_file.py       # Data preprocessing script
├── spam.csv                 # SMS Spam dataset
├── enron_spam.csv           # Enron email dataset
├── enron_spam_data.csv      # Processed dataset
├── model1                   # Saved ML model
├── model2                   # Saved ML model
├── wv.model                 # Trained Word2Vec model
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

This project uses two publicly available datasets:

- SMS Spam Collection Dataset
- Enron Email Spam Dataset

The datasets contain labeled spam and non-spam messages used for training and evaluation.

---

## Workflow

1. Load the datasets
2. Clean and preprocess email text
3. Tokenize and lemmatize text
4. Remove stop words
5. Generate Word2Vec embeddings
6. Train the XGBoost classifier
7. Save the trained model
8. Predict spam using the Flask web application

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/email-spam-detector.git
```

Move into the project directory:

```bash
cd email-spam-detector
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## Example

**Input**

```
Congratulations! You have won a free iPhone. Click here to claim your prize.
```

**Prediction**

```
Spam
```

**Input**

```
Hi, are we meeting tomorrow at 10 AM?
```

**Prediction**

```
Not Spam
```

---

## Future Improvements

- Add email attachment analysis
- Improve accuracy with BERT/Transformer models
- Support multilingual spam detection

---

## Requirements

```
Python 3.9+
Flask
xgboost
scikit-learn
gensim
nltk
pandas
numpy
joblib
```

Install manually:

```bash
pip install flask xgboost scikit-learn gensim nltk pandas numpy joblib
```

## Author

Nipun Alwala

GitHub: https://github.com/nipunalwala18-cmyk
