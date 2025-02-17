# END to END Data Science Project  


## Project Overview

This is a Machine Learning Project designed to predict students' math scores based on various factors such as gender, parental education, lunch type, and test preparation. The project follows an end-to-end ML pipeline, including data ingestion, preprocessing, model training, evaluation, and deployment.The project follows a modular structure, ensuring scalability, maintainability, and ease of use.


## Project Structure

MLProject/                
│── src/                    
│   ├── mlproject/          
│   │   ├── __init__.py     
│   │   ├── logger.py       # Logging utility
│   │   ├── exception.py    # Exception handling
│   │   ├── components/     
│   │   │   ├── __init__.py    
│   │   │   ├── data_ingestion.py
│   │   │   ├── data_monitoring.py
│   │   │   ├── data_trainer.py
│   │   │   ├── data_transformation.py
│   │   ├── pipelines/         
│   │   │   ├── __init__.py
│   │   │   ├── training_pipeline.py
│   │   │   ├── prediction_pipeline.py
│── notebook/                    
│   ├── data/          
│   │   ├── EDA STUDENT PERFORMANCE.ipynb     
│   │   ├── MODEL TRAINING.ipynb
├── utils.py
│── app.py                    
│── Dockerfile                
│── requirements.txt          
│── README.md                 
│── .gitignore                
│── setup.py                  
│── template.py



## Dataset Overview

The dataset contains 1000 student records with the following features:
1. gender : sex of student -> (Male/Female)
2. race/ethnicity : ethnicity of students -> (Group A,B,C,D,E)
3. Parental level of education : parent's final education -> (bachelor's degree,some college,maaster's degree, associate's degree, high school)
4. Lunch : having lunch before test (standard or free/reduced)
5. test preparation course : completed or not completed before test 
6. math score : Target variable - Math score of the student
7. reading score : Student's reading score
8. writing score : Student's writing score



## Features 

✅ Modular ML Pipeline – Includes Data Ingestion, Transformation, Model Training & Evaluation.
✅ Custom Exception Handling – Centralized error handling for better debugging.
✅ Logging – Integrated logging to track pipeline execution.
✅ Hyperparameter Tuning – GridSearchCV for optimizing models.
✅ Model Persistence – Trained models are stored for future use.
✅ MLflow Tracking – Logs training runs for performance monitoring.
✅ SQL Database Support – Reads data directly from an SQL database.



## Contributing

Feel free to fork this repo, make improvements, and submit a PR!

