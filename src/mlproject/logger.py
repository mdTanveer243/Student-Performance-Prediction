import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
import logging
from datetime import datetime

# Define log file path
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_DIR = os.path.join(os.getcwd(), "logs") 
os.makedirs(LOG_DIR, exist_ok=True)  

LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Set up logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logging.info("Logger initialized successfully!")  
