from dataclasses import dataclass
import os 
import sys 
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
from mlproject.exception import CustomException
from mlproject.logger import logging
import pandas as pd 
from mlproject.utils import read_sql_data
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifact', 'train.csv')
    test_data_path: str = os.path.join('artifact', 'test.csv')
    raw_data_path: str = os.path.join('artifact', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            # Replace hardcoded path with relative path
            file_path = os.path.join(os.getcwd(), "notebook", "data", "raw.csv")
            df = pd.read_csv(file_path)


            logging.info("Reading completed from raw CSV file")

            required_columns = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course", "math_score", "writing_score", "reading_score"]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise CustomException(f"Missing columns in dataset: {missing_columns}", sys)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Saving raw data to: {self.ingestion_config.raw_data_path}")

            # Split data
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info(f"Saving train data to: {self.ingestion_config.train_data_path}")
            logging.info(f"Saving test data to: {self.ingestion_config.test_data_path}")

            logging.info("Data Ingestion Completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
