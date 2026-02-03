import os
import sys 

from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
pd.read_csv('archivedata/emails.csv')


from src.component.data_trasformation import DataTransformation
from src.component.data_trasformation import DataTrasformationConfig
from src.component.model_trainer import ModelTrainer


class DataIngestionConfig:
    train_data=str=os.path.join('artifacts','train.csv')
    test_data=str=os.path.join('artifacts','test.csv')
    row_data=str=os.path.join('artifacts','row.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        
        try:
            df=pd.read_csv('archivedata\emails.csv')
            logging.info("Read the dataset")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data),exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.row_data),exist_ok=True)
            logging.info("file and folder be created")

            #train and test data
            train_set_dataset,test_set_dataset=train_test_split(df,test_size=0.2,random_state=42)
            train_set_dataset.to_csv(self.ingestion_config.train_data,index=False,header=True)
            test_set_dataset.to_csv(self.ingestion_config.test_data,index=False,header=True)

            return(

                self.ingestion_config.train_data,
                self.ingestion_config.test_data,

            )
        except Exception as e:
            raise CustomException(e,sys)
        
        