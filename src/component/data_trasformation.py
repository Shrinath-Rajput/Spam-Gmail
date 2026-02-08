import sys
import os

import numpy as np
import pandas as pd


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder , StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utlis import save_object

class DataTrasformationConfig:
    preprocessor_obj=os.path.join('artifacts','preprocessing.pkl')


class DataTransformation:
    def __init__(self):
        self.data_trasformation_config=DataTrasformationConfig()
        logging.info("succes")


    def get_data_transformer_object(self):
        
        try:
            numerical_columns = []  #fix
            categorical_columns=["gender"]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False))  # ✅ FIX
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder())
                ]
            )

            logging.info("Numerical & categorical pipelines created")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_trasformation(self, train_data, test_data):
        try:
            train_df = pd.read_csv(train_data)
            test_df = pd.read_csv(test_data)

            logging.info("Read train and test data completed")

        # 🔥 EMAIL DATASET FIX
            target_column_name = train_df.columns[-1]

            X_train = train_df.iloc[:, 0].astype(str)
            y_train = train_df[target_column_name]

            X_test = test_df.iloc[:, 0].astype(str)
            y_test = test_df[target_column_name]

            from sklearn.feature_extraction.text import TfidfVectorizer

            tfidf = TfidfVectorizer(stop_words="english", max_features=3000)

            X_train_vec = tfidf.fit_transform(X_train)
            X_test_vec = tfidf.transform(X_test)

            #train_arr = np.c_[X_train_vec.toarray(), y_train.to_numpy()]
            #test_arr = np.c_[X_test_vec.toarray(), y_test.to_numpy()]

            train_arr = (X_train_vec, y_train.to_numpy())
            test_arr = (X_test_vec, y_test.to_numpy())

            save_object(
                file_path=self.data_trasformation_config.preprocessor_obj,
                obj=tfidf
                 )

            logging.info("Data transformation completed")

            return train_arr, test_arr, self.data_trasformation_config.preprocessor_obj

        except Exception as e:
            raise CustomException(e, sys)

        