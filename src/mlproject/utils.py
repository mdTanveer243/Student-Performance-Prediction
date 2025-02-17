import os 
import sys 
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from dotenv import load_dotenv
import pymysql
import pickle
import numpy as np 

# Load environment variables for database connection
load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")

def read_sql_data():
    """Reads data from the SQL database and returns a pandas DataFrame."""
    logging.info("Reading SQL database started")
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info(f"Connection Established to database: {db}")

        df = pd.read_sql_query("SELECT * FROM student", mydb)
        logging.info("Data successfully fetched from SQL database")
        
        return df

    except Exception as ex:
        raise CustomException(ex, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """Evaluates multiple models using GridSearchCV and returns the model performance report."""
    try:
        report = {} 
        
        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")

            para = params.get(model_name, None)  # Extract parameter grid, if available
            
            if para:  # Apply GridSearchCV only if parameters exist
                gs = GridSearchCV(model, param_grid=para, cv=3)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                best_model = model
                best_model.fit(X_train, y_train)

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            logging.info(f"{model_name} | Train Score: {train_model_score} | Test Score: {test_model_score}")

            report[model_name] = test_model_score  # Store the test score

        return report

    except Exception as e:
        raise CustomException(e, sys)

def saved_object(file_path, obj):
    """Saves an object (model) to a specified file path using pickle."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)
