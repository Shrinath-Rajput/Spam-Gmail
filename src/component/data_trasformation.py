import sys
import os

import numpy as np
import pandas as pd
df=pd.read_csv('archivedata/emails.csv')

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


    def get_data_trasformer_obaject(self):
        
        try:
            categorical_columns=["gender"]

            um_pipeline = Pipeline(
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

                preprocessor_obj = self.get_data_transformer_object()

                target_column_name = "math_score"

                input_feature_train_df = train_df.drop(columns=[target_column_name])
                target_feature_train_df = train_df[target_column_name]

                input_feature_test_df = test_df.drop(columns=[target_column_name])
                target_feature_test_df = test_df[target_column_name]

                logging.info("Applying preprocessing on training and test data")

                input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)  # ✅ FIX

                train_arr = np.c_[
                    input_feature_train_arr, np.array(target_feature_train_df)
                ]

                test_arr = np.c_[
                    input_feature_test_arr, np.array(target_feature_test_df)
                ]

                save_object(
                    file_path=self.data_trasformation_config.preprocessor_obj_file_path,
                    obj=preprocessor_obj
                )

                logging.info("Preprocessor saved successfully")

                return (
                    train_arr,
                    test_arr,
                    self.data_trasformation_config.preprocessor_obj_file_path
                )
            

            except Exception as e:
                raise CustomException(e, sys)


        

  
        
        