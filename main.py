import sys
from src.exception import CustomException
from src.logger import logging

from src.component.data_ingestion import DataIngestion
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import ModelTrainer


def main():
    try:
        logging.info("========== TRAINING PIPELINE STARTED ==========")

        # 1️⃣ Data Ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        # 2️⃣ Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_data_path,
            test_data_path
        )

        # 3️⃣ Model Training
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info("========== TRAINING PIPELINE COMPLETED ==========")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
