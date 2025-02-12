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
    
    # Configure session with longer timeout and headers
    session = requests.Session()
    session.timeout = 30
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    try:
        # Step 1: Initial page load to get cookies
        logger.info("Getting initial session...")
        response = session.get(BASE_URL, allow_redirects=True)
        if not response.ok:
            logger.error(f"Failed to get initial session: {response.status_code}")
            return False

        # Step 2: Handle Streamlit authentication flow
        if "_stcore/authenticate" in response.url:
            logger.info("Detected Streamlit authentication page...")
            
            # Extract CSRF token if present
            csrf_token = None
            if 'csrf_token' in response.cookies:
                csrf_token = response.cookies['csrf_token']
                session.headers.update({'X-CSRFToken': csrf_token})
            
            # Prepare login data
            login_data = {
                'username': USERNAME,
                'password': PASSWORD
            }
            if csrf_token:
                login_data['csrf_token'] = csrf_token
            
            # Attempt login
            logger.info("Attempting login...")
            login_url = f"{BASE_URL}/_stcore/authenticate"
            login_response = session.post(login_url, data=login_data, allow_redirects=True)
            
            if not login_response.ok:
                logger.error(f"Login failed with status code: {login_response.status_code}")
                return False
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
