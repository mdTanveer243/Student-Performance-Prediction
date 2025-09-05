import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
from mlproject.logger import logging
from mlproject.exception import CustomException
from mlproject.components.data_ingestion import DataIngestion
from mlproject.components.data_transformation import DataTransformation
from mlproject.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    logging.info("Execution started for the ML pipeline.")

    try:
        # Data Ingestion
        logging.info("Starting data ingestion...")
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data ingestion completed. Train Path: {train_path}, Test Path: {test_path}")

        # Data Transformation
        logging.info("Starting data transformation...")
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)
        logging.info("Data transformation completed successfully.")

        # Model Trainer 
        logging.info("Starting model training...")
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr, test_arr))

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise CustomException(e, sys)
