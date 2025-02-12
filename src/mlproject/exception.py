from mlproject.logger import logging  
import sys 

def error_message_detail(error, error_details: sys):
    _, _, exc_tb = error_details.exc_info()
    
    if exc_tb is not None:  # Check if traceback exists
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "Unknown"
        line_number = "Unknown"
    
    error_message = f"Error in script: [{file_name}] at line [{line_number}] - {str(error)}"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message , error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_details)

    
    def __str__(self):
        return self.error_message
 