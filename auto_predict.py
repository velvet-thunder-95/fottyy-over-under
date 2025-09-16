import requests
import time
import logging
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def keep_alive(wait_time=600):  # Default to 10 minutes
    """Login to Streamlit app and keep it alive by waiting"""
    BASE_URL = "https://fottyy-over-under.streamlit.app/?page=login"
    USERNAME = st.secrets["auth"]["username_1"]
    PASSWORD = st.secrets["auth"]["password_1"]
    
    try:
        # Create a session to maintain cookies
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        # Get login page
        logger.info("Accessing login page...")
        response = session.get(BASE_URL)
        logger.info(f"Login page status: {response.status_code}")
        
        # Get CSRF token if present
        csrf_token = response.cookies.get('csrf_token')
        logger.info(f"CSRF Token found: {csrf_token is not None}")
        
        if csrf_token:
            session.headers.update({'X-CSRFToken': csrf_token})
        
        # Prepare login data
        login_data = {
            'username': USERNAME,
            'password': PASSWORD
        }
        if csrf_token:
            login_data['csrf_token'] = csrf_token
        
        # Submit login
        login_url = f"https://fottyy-over-under.streamlit.app/_stcore/authenticate"  # Use consistent URL
        logger.info("Submitting login...")
        login_response = session.post(login_url, data=login_data, allow_redirects=True)
        logger.info(f"Login response status: {login_response.status_code}")
        
        if not login_response.ok:
            logger.error(f"Login failed: {login_response.status_code}")
            return False
            
        logger.info("Login successful")
        
        # Wait for specified time
        logger.info(f"Waiting for {wait_time} seconds to keep app alive...")
        time.sleep(wait_time)
        
        logger.info("Keep-alive completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during keep-alive: {str(e)}")
        return False

if __name__ == "__main__":
    # Use 10 minutes (600 seconds) for production
    keep_alive(600)
