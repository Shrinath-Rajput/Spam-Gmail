Email Spam Detection System (End-to-End ML Project)
* Overview

This project implements a complete End-to-End Machine Learning system for Email Spam Detection.
It goes beyond model training and demonstrates a full ML workflow including:

Data preprocessing

Model training

Experiment tracking with MLflow

Backend API integration

Database storage of predictions

Frontend user interface

Version control & experiment reproducibility

The goal was to simulate a production-oriented ML system, not just a notebook experiment.

* Problem Statement

Automatically classify incoming email text as:

* Not Spam

* Spam

This helps reduce unwanted emails and demonstrates text classification using machine learning.

* Project Architecture

User Input (Frontend)
⬇
Flask Backend API
⬇
TF-IDF Vectorizer
⬇
SGDClassifier Model
⬇
Prediction Result
⬇
MongoDB (Store Prediction History)
⬇
MLflow (Track Experiment & Model)

# Model & Methodology
🔹 Feature Extraction

TF-IDF Vectorization

Stopword removal

Limited max_features for efficiency

🔹 Machine Learning Model

SGDClassifier

loss="log_loss" (Logistic Regression behavior)

Suitable for sparse text data

Efficient and scalable

🔹 Why SGDClassifier?

Works well with high-dimensional sparse data

Faster than traditional LogisticRegression for large datasets

Supports incremental learning

* Experiment Tracking

Implemented MLflow to track:

Model parameters

Accuracy metrics

Training runs

Model artifacts

MLflow integrated with DagsHub for Git-based experiment tracking.

* Database Integration

Used MongoDB (Local) to store:

Input email

Prediction result

Confidence score

This simulates real-world logging of ML inference results.

* Backend & Frontend
Backend

Flask API

Model loading

Real-time prediction

MongoDB insertion

Frontend

Simple UI for email input

Displays spam / not spam result

# Tech Stack

Python

Scikit-Learn

MLflow

Flask

MongoDB

DagsHub

Git & GitHub

# Project Structure
email-detection/
│
├── artifacts/
│   └── model.pkl
│
├── src/
│   ├── data_transformation.py
│   ├── model_trainer.py
│
├── templates/
│   └── index.html
│
├── pushdata.py
├── app.py
├── main.py
├── requirements.txt
└── README.md

▶* How to Run
1️ Install Dependencies
pip install -r requirements.txt

2️ Train Model
python main.py

3️ Run Flask App
python app.py

4️ Start MLflow UI
mlflow ui


Open:

http://localhost:5000