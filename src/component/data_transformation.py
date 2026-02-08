import sys
import os
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from src.exception import CustomException
from src.logger import logging
from src.utlis import save_object


class DataTransformationConfig:
    preprocessor_obj = os.path.join("artifacts", "preprocessing.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def initiate_data_transformation(self, train_data, test_data):
        try:
            train_df = pd.read_csv(train_data)
            test_df = pd.read_csv(test_data)

            logging.info(f"Train columns: {train_df.columns}")

            TEXT_COLUMN = "message"

    
            
            possible_label_cols = ["label", "class", "category", "v1", "spam"]
            LABEL_COLUMN = None

            for col in possible_label_cols:
                if col in train_df.columns:
                    LABEL_COLUMN = col
                    break

            if LABEL_COLUMN is None:
                raise ValueError(
                    "No label column found. Dataset does not contain spam/ham labels."
                )

            train_df = train_df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()
            test_df = test_df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

            # Normalize labels
            train_df[LABEL_COLUMN] = (
                train_df[LABEL_COLUMN]
                .astype(str)
                .str.lower()
                .str.strip()
                .map({"spam": 1, "ham": 0})
            )

            test_df[LABEL_COLUMN] = (
                test_df[LABEL_COLUMN]
                .astype(str)
                .str.lower()
                .str.strip()
                .map({"spam": 1, "ham": 0})
            )

            train_df = train_df.dropna()
            test_df = test_df.dropna()

            y_train = train_df[LABEL_COLUMN].astype(int)
            y_test = test_df[LABEL_COLUMN].astype(int)

            # -----------------------------
            # 🔥 FINAL SAFETY CHECK
            # -----------------------------
            if y_train.nunique() < 2:
                raise ValueError(
                    f"Only one class found in dataset: {y_train.unique()}. "
                    f"Spam classifier requires both spam and ham samples."
                )

            X_train_text = train_df[TEXT_COLUMN].astype(str)
            X_test_text = test_df[TEXT_COLUMN].astype(str)

            tfidf = TfidfVectorizer(
                stop_words="english",
                max_features=500
            )

            X_train_vec = tfidf.fit_transform(X_train_text)
            X_test_vec = tfidf.transform(X_test_text)

            save_object(
                file_path=self.config.preprocessor_obj,
                obj=tfidf
            )

            logging.info(
                f"Data Transformation completed | classes={y_train.unique()}"
            )

            return (
                (X_train_vec, y_train.to_numpy()),
                (X_test_vec, y_test.to_numpy()),
                self.config.preprocessor_obj
            )

        except Exception as e:
            raise CustomException(e, sys)
