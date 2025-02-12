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
    
    # Configure session with longer timeout
    session = requests.Session()
    session.timeout = 30
    
    try:
        # Step 1: First visit the page to get session cookie
        logger.info("Getting initial session...")
        response = session.get(BASE_URL)
        if not response.ok:
            logger.error("Failed to get initial session")
            return False
            
        # Extract any auth redirect URL if present
        if "auth/app" in response.url:
            logger.info("Handling Streamlit authentication...")
            # Follow the auth redirect
            auth_response = session.get(response.url)
            if not auth_response.ok:
                logger.error("Failed to complete Streamlit authentication")
                return False

        # Step 2: Login
        logger.info("Attempting to login...")
        params = {
            "page": "login"
        }
        data = {
            "username": USERNAME,
            "password": PASSWORD
        }
        response = session.post(f"{BASE_URL}", params=params, data=data)
        if not response.ok:
            logger.error("Login failed")
            return False
            
        # Verify we're properly logged in by checking for redirect
        if "auth/app" in response.url:
            logger.info("Following post-login authentication...")
            auth_response = session.get(response.url)
            if not auth_response.ok:
                logger.error("Failed to complete post-login authentication")
                return False
            
        # Step 3: Navigate to prediction page and wait for predictions
        logger.info("Getting predictions...")
        params = {
            "page": "main",
            "auto_predict": "true"  # Trigger automatic predictions
        }
        response = session.get(BASE_URL, params=params)
        if not response.ok:
            logger.error("Failed to get predictions")
            return False
            
        logger.info("Auto-prediction completed successfully")
        return True
            
        logger.info("Auto-prediction completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during auto-prediction: {str(e)}")
        return False

if __name__ == "__main__":
    auto_predict()
