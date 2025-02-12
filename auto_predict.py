import requests
import time
from datetime import datetime
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def auto_predict():
    BASE_URL = "https://fottyy.streamlit.app"
    USERNAME = "matchday_wizard"  # Hardcoded username
    PASSWORD = "GoalMaster"  # Hardcoded password
    
    try:
        # Start a session to maintain cookies
        session = requests.Session()
        
        # Step 1: Login
        logger.info("Attempting to login...")
        login_data = {
            "username": USERNAME,
            "password": PASSWORD
        }
        response = session.post(f"{BASE_URL}/login", json=login_data)
        if not response.ok:
            logger.error("Login failed")
            return False
            
        # Step 2: Navigate to prediction page and wait for predictions
        logger.info("Getting predictions...")
        params = {
            "page": "main",
            "auto_predict": "true"  # We'll add this parameter to trigger automatic predictions
        }
        response = session.get(BASE_URL, params=params)
        if not response.ok:
            logger.error("Failed to get predictions")
            return False
            
        logger.info("Auto-prediction completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during auto-prediction: {str(e)}")
        return False

if __name__ == "__main__":
    auto_predict()
