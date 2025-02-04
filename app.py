import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from mlproject.logger import logging
from mlproject.logger import logging 


from mlproject.exception import CustomException
from mlproject.components import data_ingestion
from mlproject.components.data_ingestion import DataIngestion, DataIngestionConfig


if __name__ == "__main__":
    logging.info("The Execution has started")

    try:
        data_ingestion_config=DataIngestionConfig()
        data_ingestion= DataIngestion()
        data_ingestion.initiate_data_ingestion()
        
    
    except Exception as e :
        logging.info("Custom Exception")
        raise CustomException(e,sys)
