import sys
import os
import pickle
import numpy as np

import mlflow
import mlflow.sklearn

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging


class ModelTrainer:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Model training started")

            X_train, y_train = train_arr
            X_test, y_test = test_arr

            y_train = y_train.astype(int)
            y_test = y_test.astype(int)

            classes = np.unique(y_train)

            # 🔥 SAFETY CHECK
            if len(classes) < 2:
                raise ValueError(
                    f"Training data has only one class: {classes}. "
                    f"Model training requires at least 2 classes (spam & ham)."
                )

            # =========================
            # MLflow setup
            # =========================
            mlflow.set_experiment("Email_Spam_Detection")

            with mlflow.start_run():

                model = SGDClassifier(
                    loss="log_loss",
                    max_iter=1,
                    tol=None,
                    random_state=42
                )

                print(">>> PARTIAL FIT STARTED")

                for epoch in range(3):
                    model.partial_fit(X_train, y_train, classes=classes)
                    print(f">>> epoch {epoch + 1} completed")

                print(">>> TRAINING COMPLETED")

                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                logging.info(f"Model accuracy: {acc}")

                # =========================
                # MLflow logging
                # =========================
                mlflow.log_param("model", "SGDClassifier")
                mlflow.log_param("loss", "log_loss")
                mlflow.log_param("epochs", 3)
                mlflow.log_metric("accuracy", acc)

                mlflow.sklearn.log_model(model, "spam_model")

                # =========================
                # Save model locally
                # =========================
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                with open(self.model_path, "wb") as f:
                    pickle.dump(model, f)

                logging.info("Model saved successfully")

                print(f"✅ Training complete | Accuracy: {acc}")

                return acc

        except Exception as e:
            raise CustomException(e, sys)
