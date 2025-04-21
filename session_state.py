import streamlit as st
from supabase_db import SupabaseDB
import logging

# Configure logging
logger = logging.getLogger(__name__)

def init_session_state():
    """Initialize session state variables"""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'db' not in st.session_state:
        st.session_state.db = SupabaseDB()

def check_login_state():
    """Check if user is logged in"""
    return st.session_state.logged_in

def login_user(username, password):
    """Log in a user"""
    try:
        db = st.session_state.db
        user = db.authenticate_user(username, password)
        
        if user:
            st.session_state.user_id = user['id']
            st.session_state.username = user['username']
            st.session_state.logged_in = True
            logger.info(f"User {username} logged in successfully")
            return True
        else:
            logger.warning(f"Failed login attempt for user {username}")
            return False
            
    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        return False

def logout_user():
    """Log out the current user"""
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.logged_in = False
    logger.info("User logged out")

def register_user(username, password):
    """Register a new user"""
    try:
        db = st.session_state.db
        success = db.create_user(username, password)
        
        if success:
            logger.info(f"User {username} registered successfully")
            # Automatically log in the user after registration
            return login_user(username, password)
        else:
            logger.warning(f"Failed to register user {username}")
            return False
            
    except Exception as e:
        logger.error(f"Error during registration: {str(e)}")
        return False
