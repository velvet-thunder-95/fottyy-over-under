# app.py



import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from football_api import get_matches, LEAGUE_IDS, get_match_result, get_match_by_teams
from sklearn.impute import SimpleImputer
import joblib
from sklearn.preprocessing import StandardScaler
import logging
import xgboost as xgb
from history import show_history_page, PredictionHistory
from session_state import init_session_state, check_login_state
import json
from scipy.stats import poisson
import pytz
from zoneinfo import ZoneInfo
import time
import base64
from unidecode import unidecode as unidecode_text
from sklearn.impute import SimpleImputer
from transfermarkt_api import TransfermarktAPI
from odds_generator import OddsGenerator
from odds_fetcher import OddsFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Get the absolute path of the current file (app.py)
current_file_path = os.path.abspath(__file__)

# Get the directory containing app.py
current_dir = os.path.dirname(current_file_path)

# Get the parent directory (project root)
project_root = current_dir

# Add the src directory to the Python path
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)


# Set page title and favicon
st.set_page_config(
    page_title="Fottyy - Football Prediction",
    page_icon="assets/favicon.png"
)

# Custom CSS
st.markdown("""
<style>
    /* Base styles */
    .stApp {
        background-color: #f0f2f6;
    }

    /* Login Form Styling */
    .login-container {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        max-width: 400px;
        margin: 2rem auto;
    }

    /* Form Elements */
    .stTextInput > div > div {
        background-color: white !important;
    }

    .stTextInput input {
        color: #1a1a1a !important;
        background-color: white !important;
        font-size: 1rem !important;
    }

    .stTextInput > label {
        color: #1a1a1a !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    /* Button Styling - Global */
    .stButton > button {
        width: 100% !important;
        height: auto !important;
        padding: 0.75rem 1.5rem !important;
        background-color: #2c5282 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        margin: 0.5rem 0 !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        background-color: #1a365d !important;
        transform: translateY(-1px) !important;
    }

    /* Login Form Submit Button */
    .stForm button[type="submit"] {
        width: 100% !important;
        padding: 0.75rem 1.5rem !important;
        background-color: #2c5282 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        margin-top: 1rem !important;
        cursor: pointer !important;
    }

    .stForm button[type="submit"]:hover {
        background-color: #1a365d !important;
    }

    /* Prediction Elements */
    .winner-prediction {
        background-color: #48bb78 !important;
        color: white !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        margin: 1rem 0 !important;
        text-align: center !important;
        font-weight: bold !important;
        font-size: 1.2rem !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }

    .probability-container {
        background-color: white !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        margin: 0.5rem !important;
        text-align: center !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }

    .probability-label {
        color: #1a1a1a !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
        font-size: 1rem !important;
    }

    .probability-value {
        color: #2c5282 !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
    }

    /* Progress Bar Container */
    .progress-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .progress-label {
        color: #1a1a1a;
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .progress-bar {
        width: 100%;
        height: 8px;
        background-color: #e2e8f0;
        border-radius: 4px;
        overflow: hidden;
        margin-top: 0.25rem;
    }

    .progress-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }

    .progress-fill-home {
        background-color: #48bb78;  /* Green */
    }

    .progress-fill-draw {
        background-color: #ed8936;  /* Orange */
    }

    .progress-fill-away {
        background-color: #3182ce;  /* Blue */
    }

    /* Progress Bar Styling */
    .stProgress > div > div > div {
        height: 8px;
        background-color: #e2e8f0;
    }
    
    /* Home Team Progress */
    .stProgress:nth-of-type(1) > div > div > div > div {
        background-color: #48bb78 !important;
    }
    
    /* Draw Progress */
    .stProgress:nth-of-type(2) > div > div > div > div {
        background-color: #ed8936 !important;
    }
    
    /* Away Team Progress */
    .stProgress:nth-of-type(3) > div > div > div > div {
        background-color: #3182ce !important;
    }
    
    /* Adjust spacing */
    .stProgress {
        margin-bottom: 0.5rem;
    }
    
    /* Team names and percentages */
    .element-container p {
        margin-bottom: 1rem;
        font-weight: 500;
    }

    /* Headers and Text */
    h1, h2, h3, h4, h5, h6 {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }

    /* Prediction Text */
    .prediction-text {
        color: #1a1a1a !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        margin: 0.5rem 0 !important;
    }

    /* Confidence Levels */
    .prediction-high, .prediction-medium, .prediction-low {
        color: white !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        margin: 1rem 0 !important;
        text-align: center !important;
        font-weight: bold !important;
        font-size: 1.2rem !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }

    .prediction-high {
        background-color: #48bb78 !important;  /* Green */
    }

    .prediction-medium {
        background-color: #ed8936 !important;  /* Orange */
    }

    .prediction-low {
        background-color: #e53e3e !important;  /* Red */
    }

    /* Section Headers */
    .section-header {
        color: #1a1a1a !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        margin: 1.5rem 0 1rem 0 !important;
    }

    /* Match card styles */
    .match-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
    }

    .league-name {
        color: #4a5568;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 1rem;
    }

    .teams-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
    }

    .team-name {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2d3748;
    }

    .vs-badge {
        background: #edf2f7;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: #4a5568;
        font-weight: 500;
    }

    .prediction-wrapper {
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }

    .prediction-high {
        background-color: #c6f6d5;
        border: 1px solid #9ae6b4;
    }

    .prediction-medium {
        background-color: #feebc8;
        border: 1px solid #fbd38d;
    }

    .prediction-low {
        background-color: #fed7d7;
        border: 1px solid #feb2b2;
    }

    .prediction-text {
        color: #2d3748 !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
    }

    .confidence-text {
        color: #4a5568;
        font-size: 0.9rem;
        font-weight: 500;
    }

    .stats-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin: 0 0 5px 0;
        padding: 0;
        width: 100%;
        max-width: 800px;
    }

    .stat-box {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 8px;
        text-align: center;
    }

    .stat-label {
        color: #4a5568;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 4px;
    }

    .stat-value {
        color: #2d3748;
        font-size: 1.1rem;
        font-weight: 600;
    }

    /* Progress bar styles */
    .progress-container {
        margin-top: 1.5rem;
    }

    .progress-bar {
        height: 8px;
        border-radius: 4px;
        background: #e2e8f0;
        overflow: hidden;
        margin-bottom: 0.5rem;
    }

    .progress-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }

    .legend-item {
        display: inline-flex;
        align-items: center;
        margin-right: 1rem;
        font-size: 0.8rem;
        color: #4a5568;
    }

    .legend-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 4px;
    }
    
    /* Team logo styling */
    .team-info {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.5rem;
    }
    
    .team-logo {
        width: 48px;
        height: 48px;
        object-fit: contain;
        border-radius: 50%;
        background: white;
        padding: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .teams-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem;
        gap: 1rem;
    }
    
    .vs-badge {
        background: #2c5282;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .team-name {
        font-weight: 600;
        font-size: 1rem;
        color: #2d3748;
        text-align: center;
    }
    
    /* Team Logos and Match Card */
    .match-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }

    .league-name {
        font-size: 0.9rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 1rem;
        text-align: center;
    }

    .teams-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 1rem 0;
    }

    .team-info {
        display: flex;
        flex-direction: column;
        align-items: center;
        flex: 1;
    }

    .team-logo {
        width: 60px;
        height: 60px;
        object-fit: contain;
        margin-bottom: 0.5rem;
    }

    .team-name {
        font-size: 1rem;
        font-weight: 600;
        color: #2d3748;
        text-align: center;
    }

    .vs-badge {
        background: #edf2f7;
        color: #4a5568;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0 1rem;
    }
    
    /* Add more styles after this */
    .match-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 10px 0;
    }
    .team-info {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        min-width: 100px;
    }
    .team-logo, .team-logo-placeholder {
        width: 40px;
        height: 40px;
        object-fit: contain;
        border-radius: 50%;
        background: white;
        padding: 2px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .team-logo-placeholder {
        background: #f0f0f0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: #666;
        font-size: 14px;
    }
    .team-name {
        font-size: 0.9rem;
        font-weight: 500;
        color: #2d3748;
        margin-top: 2px;
        line-height: 1.2;
    }
    .vs-badge {
        margin: 0 10px;
        font-size: 1rem;
        font-weight: 500;
        color: #4a5568;
    }
    .league-name {
        font-size: 0.8rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 4px 0;
        line-height: 1.2;
    }
    .kickoff-time {
        font-size: 0.8rem;
        color: #718096;
        margin: 2px 0;
        line-height: 1.2;
    }
    
    /* Back to Top Button */
    .back-to-top {
        position: fixed;
        bottom: 60px;
        right: 20px;
        display: flex;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-color: white;
        color: #000000;
        text-align: center;
        font-size: 20px;
        text-decoration: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        z-index: 1000;
        justify-content: center;
        align-items: center;
        transition: all 0.3s ease;
    }

    .back-to-top:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        color: #000000;
    }

    .back-to-top i {
        line-height: 40px;
        display: inline-block;
        color: #000000;
    }

    /* Slider Styling */
    .stSlider > div > div > div > div {
        height: 24px !important;
    }

    .stSlider > div > div > div > div > div {
        height: 24px !important;
        background-color: #2c5282 !important;
        border-radius: 12px !important;
    }

    /* Slider Thumb */
    .stSlider > div > div > div > div > div:nth-child(2) {
        width: 32px !important;
        height: 32px !important;
        background-color: white !important;
        border: 3px solid #2c5282 !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2) !important;
        top: -4px !important;
        border-radius: 50% !important;
        cursor: pointer !important;
    }
    
</style>
""", unsafe_allow_html=True)

# Hardcoded credentials
VALID_USERNAME = "matchday_wizard"
VALID_PASSWORD = "GoalMaster"

# Additional user
VALID_USERNAME_2 = "fottyy_pro"
VALID_PASSWORD_2 = "BetMaster123"

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'login_submitted' not in st.session_state:
    st.session_state.login_submitted = False

def login(username, password):
    """Validate login credentials"""
    if (username == VALID_USERNAME and password == VALID_PASSWORD) or \
       (username == VALID_USERNAME_2 and password == VALID_PASSWORD_2):
        st.session_state.logged_in = True
        return True
    return False

def logout():
    st.session_state.logged_in = False

def show_login_page():
    """Display the login page"""
    with st.container():
        st.markdown("""<div class="login-container">
            <h2 style="color: #1a1a1a; text-align: center; margin-bottom: 2rem;">Welcome Back!</h2>
        </div>""", unsafe_allow_html=True)
        
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            submitted = st.form_submit_button("Login", use_container_width=True)
            
            if submitted:
                if login(username, password):
                    st.session_state.logged_in = True
                    st.query_params["page"] = "main"
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        

def get_league_name(match):
    """Get league name from match data using LEAGUE_IDS mapping"""
    competition_id = match.get('competition_id')
    if not competition_id:
        return "Unknown League"
        
    # Find the league name by competition ID
    for league_name, league_id in LEAGUE_IDS.items():
        if league_id == competition_id:
            return league_name
    
    return f"League {competition_id}"

# Load the saved model
@st.cache_resource
def load_model():
    try:
        # Update the path to point to the models directory
        model_path = os.path.join(project_root, 'models', 'football_prediction_model.joblib')
        model = joblib.load(model_path)
        
        # If it's a scikit-learn XGBoost model, get the underlying booster
        if hasattr(model, '_Booster'):
            return model._Booster
        # If it's already a booster, return it directly
        elif isinstance(model, xgb.Booster):
            return model
        # Otherwise, return the model as is
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        print(f"Detailed error: {e}")
        return None


# Load the predictor model
predictor = load_model()
transfermarkt_api = TransfermarktAPI()
odds_generator = OddsGenerator(transfermarkt_api)  # Pass the transfermarkt_api instance
odds_fetcher = OddsFetcher()  # Initialize the odds fetcher for Supabase football_odds table

def create_match_features_from_api(match_data):
    """Create features DataFrame from match data with error handling"""
    try:
        features = {}
        logger = logging.getLogger(__name__)

        # First add features in the exact order they were used during training
        feature_order = [
            'win_rate_difference', 'possession_difference', 'xg_difference',
            'shot_difference', 'momentum_difference', 'implied_prob_sum',
            'form_difference', 'odds_ratio', 'total_momentum',
            'home_win_rate', 'away_win_rate', 'home_possession', 'away_possession',
            'home_xg', 'away_xg', 'home_shots', 'away_shots', 'home_momentum',
            'away_momentum', 'implied_home_prob', 'implied_draw_prob', 'implied_away_prob',
            'home_form_points', 'away_form_points', 'odds_home_win', 'odds_draw',
            'odds_away_win', 'season', 'competition_id', 'home_total_matches',
            'away_total_matches', 'home_shots_on_target', 'away_shots_on_target',
            'home_corners', 'away_corners', 'home_fouls', 'away_fouls',
            'shot_accuracy_home', 'shot_accuracy_away', 'home_win_rate_ratio'
        ]

        # Basic features with safe defaults
        try:
            features['season'] = int(match_data.get('season', '2024/2025').split('/')[0])
        except (ValueError, AttributeError):
            features['season'] = 2024
            
        features['competition_id'] = int(match_data.get('competition_id', 0))
        
        # Match counts and form using completed matches - ensure minimum of 1
        features['home_total_matches'] = max(1, float(match_data.get('matches_completed_minimum', 1)))
        features['away_total_matches'] = max(1, float(match_data.get('matches_completed_minimum', 1)))
        
        # Win rates - use pre-match PPG with default values
        try:
            home_ppg = max(0.1, float(match_data.get('pre_match_teamA_overall_ppg', 1.0)))
            away_ppg = max(0.1, float(match_data.get('pre_match_teamB_overall_ppg', 1.0)))
        except (ValueError, TypeError):
            logger.warning("Invalid PPG values, using defaults")
            home_ppg = 1.0
            away_ppg = 1.0
            
        features['home_win_rate'] = home_ppg / 3.0
        features['away_win_rate'] = away_ppg / 3.0
        
        # Adjust win rates based on home/away specific PPG with safe division
        try:
            home_ppg_specific = max(0.1, float(match_data.get('pre_match_home_ppg', home_ppg)))
            away_ppg_specific = max(0.1, float(match_data.get('pre_match_away_ppg', away_ppg)))
            home_ppg_ratio = home_ppg_specific / home_ppg if home_ppg > 0 else 1.0
            away_ppg_ratio = away_ppg_specific / away_ppg if away_ppg > 0 else 1.0
        except (ValueError, TypeError):
            logger.warning("Invalid specific PPG values, using defaults")
            home_ppg_specific = home_ppg
            away_ppg_specific = away_ppg
            home_ppg_ratio = 1.0
            away_ppg_ratio = 1.0
        
        features['home_win_rate'] *= max(0.5, min(1.5, home_ppg_ratio))
        features['away_win_rate'] *= max(0.5, min(1.5, away_ppg_ratio))
        
        # Form points with home/away adjustment
        features['home_form_points'] = home_ppg_specific
        features['away_form_points'] = away_ppg_specific
        
        # Use potential metrics with safe defaults
        try:
            goal_potential = max(0.1, float(match_data.get('o25_potential', 50))) / 100.0
            corner_potential = max(1.0, float(match_data.get('corners_potential', 10))) / 10.0
        except (ValueError, TypeError):
            logger.warning("Invalid potential values, using defaults")
            goal_potential = 0.5
            corner_potential = 1.0
        
        # Shot statistics based on xG and potential metrics
        avg_shots_ratio = 12  # Average shots per expected goal
        try:
            home_xg = max(0.1, float(match_data.get('team_a_xg_prematch', 1.0)))
            away_xg = max(0.1, float(match_data.get('team_b_xg_prematch', 1.0)))
        except (ValueError, TypeError):
            logger.warning("Invalid xG values, using defaults")
            home_xg = 1.0
            away_xg = 1.0
        
        features['home_shots'] = max(1, int(home_xg * avg_shots_ratio * goal_potential))
        features['away_shots'] = max(1, int(away_xg * avg_shots_ratio * goal_potential))
        features['home_shots_on_target'] = max(1, int(features['home_shots'] * 0.4))
        features['away_shots_on_target'] = max(1, int(features['away_shots'] * 0.4))
        
        # Corner predictions using corner odds and potential
        try:
            corner_odds_home = max(0.1, float(match_data.get('odds_corners_1', 2.0)))
            corner_odds_away = max(0.1, float(match_data.get('odds_corners_2', 2.0)))
        except (ValueError, TypeError):
            logger.warning("Invalid corner odds, using defaults")
            corner_odds_home = 2.0
            corner_odds_away = 2.0
        
        # Default corner predictions with safe values
        if corner_odds_home <= 0.1 or corner_odds_away <= 0.1:
            features['home_corners'] = max(1, int(corner_potential * 5))
            features['away_corners'] = max(1, int(corner_potential * 5))
        else:
            corner_odds_ratio = corner_odds_home / corner_odds_away
            features['home_corners'] = max(1, int(corner_potential * (5 + corner_odds_ratio)))
            features['away_corners'] = max(1, int(corner_potential * (5 + 1/corner_odds_ratio)))
        
        # Fouls based on cards potential with safe defaults
        try:
            cards_potential = max(1.0, float(match_data.get('cards_potential', 3.75)))
        except (ValueError, TypeError):
            logger.warning("Invalid cards potential, using default")
            cards_potential = 3.75
            
        features['home_fouls'] = max(1, int(10 * cards_potential / 3.75))
        features['away_fouls'] = max(1, int(10 * cards_potential / 3.75))
        
        # Possession based on team strength and BTTS potential
        try:
            btts_potential = max(0.1, float(match_data.get('btts_potential', 50))) / 100.0
        except (ValueError, TypeError):
            logger.warning("Invalid BTTS potential, using default")
            btts_potential = 0.5
            
        strength_ratio = features['home_win_rate'] / max(0.1, features['away_win_rate'])
        base_possession = 50.0
        possession_adjustment = min(15, max(-15, 10 * (strength_ratio - 1) * btts_potential))
        features['home_possession'] = max(35, min(65, base_possession + possession_adjustment))
        features['away_possession'] = 100 - features['home_possession']
        
        # Expected goals - use prematch xG with safe defaults
        features['home_xg'] = home_xg
        features['away_xg'] = away_xg
        
        # Goals - use prematch goals with safe defaults
        try:
            features['home_goals'] = max(0, float(match_data.get('team_a_goals_prematch', 0)))
            features['away_goals'] = max(0, float(match_data.get('team_b_goals_prematch', 0)))
        except (ValueError, TypeError):
            logger.warning("Invalid goals values, using defaults")
            features['home_goals'] = 0
            features['away_goals'] = 0
        
        # Shot accuracy with safe division
        features['shot_accuracy_home'] = features['home_shots_on_target'] / max(1, features['home_shots'])
        features['shot_accuracy_away'] = features['away_shots_on_target'] / max(1, features['away_shots'])
        
        # Team strength ratios with safe division
        features['home_win_rate_ratio'] = features['home_win_rate'] / max(0.1, features['away_win_rate'])
        
        # Momentum based on recent form
        features['home_momentum'] = features['home_form_points'] / 3.0
        features['away_momentum'] = features['away_form_points'] / 3.0
        
        # Odds and implied probabilities with safe defaults
        try:
            # First try to get odds from FootyStats
            odds_home = match_data.get('odds_ft_1')
            odds_draw = match_data.get('odds_ft_x')
            odds_away = match_data.get('odds_ft_2')
            
            # If FootyStats odds are not available, use odds from odds generator
            if odds_home is None or odds_draw is None or odds_away is None:
                logger.info("Using odds from odds generator")
                odds_home = float(match_data.get('generated_odds_1', 2.0))
                odds_draw = float(match_data.get('generated_odds_x', 3.0))
                odds_away = float(match_data.get('generated_odds_2', 2.0))
                
            # Ensure odds are valid positive numbers
            odds_home = max(1.1, float(odds_home))
            odds_draw = max(1.1, float(odds_draw))
            odds_away = max(1.1, float(odds_away))
            
            # Get over/under odds
            odds_over25 = max(1.1, float(match_data.get('odds_ft_over25', match_data.get('generated_odds_over25', 2.0))))
            odds_under25 = max(1.1, float(match_data.get('odds_ft_under25', match_data.get('generated_odds_under25', 2.0))))
            
            features['odds_home_win'] = odds_home
            features['odds_draw'] = odds_draw
            features['odds_away_win'] = odds_away
            features['odds_over25'] = odds_over25
            features['odds_under25'] = odds_under25
            
            # Calculate implied probabilities and EV for match outcomes
            if all([odds_home, odds_draw, odds_away]):
                # Convert odds to probabilities
                total_prob = (1/odds_home + 1/odds_draw + 1/odds_away)
                home_implied = (1/odds_home) / total_prob
                draw_implied = (1/odds_draw) / total_prob
                away_implied = (1/odds_away) / total_prob
                
                # Debug prints
                print(f"Home - Pred: {home_implied*100:.2f}%, Odds: {odds_home:.2f}")
                print(f"Draw - Pred: {draw_implied*100:.2f}%, Odds: {odds_draw:.2f}")
                print(f"Away - Pred: {away_implied*100:.2f}%, Odds: {odds_away:.2f}")
                
                home_ev = calculate_ev(home_implied*100, odds_home)
                draw_ev = calculate_ev(draw_implied*100, odds_draw)
                away_ev = calculate_ev(away_implied*100, odds_away)
                
                # Debug prints
                print(f"Home EV: {home_ev:.2f}%")
                print(f"Draw EV: {draw_ev:.2f}%")
                print(f"Away EV: {away_ev:.2f}%")
            else:
                home_implied = draw_implied = away_implied = 0
                home_ev = draw_ev = away_ev = 0
            
            # Extract existing odds from match data
            footystats_odds = {
                'home_odds': float(match_data.get('odds_ft_1', 0)),
                'draw_odds': float(match_data.get('odds_ft_x', 0)),
                'away_odds': float(match_data.get('odds_ft_2', 0)),
                'over25_odds': float(match_data.get('odds_ft_over25', 0)),
                'under25_odds': float(match_data.get('odds_ft_under25', 0))
            }
            
            # Use odds generator if any FootyStats odds are missing
            if not all(footystats_odds.values()):
                generated_odds = odds_generator.get_odds(match_data, footystats_odds)
                match_data['odds_ft_1'] = generated_odds['home_odds']
                match_data['odds_ft_x'] = generated_odds['draw_odds']
                match_data['odds_ft_2'] = generated_odds['away_odds']
                match_data['odds_ft_over25'] = generated_odds['over25_odds']
                match_data['odds_ft_under25'] = generated_odds['under25_odds']
            
            # Calculate implied probabilities with margin adjustment
            if all([match_data['odds_ft_1'], match_data['odds_ft_x'], match_data['odds_ft_2']]):
                # Convert odds to probabilities
                total_prob = (1/match_data['odds_ft_1'] + 1/match_data['odds_ft_x'] + 1/match_data['odds_ft_2'])
                features['implied_home_prob'] = (1/match_data['odds_ft_1']) / total_prob
                features['implied_draw_prob'] = (1/match_data['odds_ft_x']) / total_prob
                features['implied_away_prob'] = (1/match_data['odds_ft_2']) / total_prob
            else:
                logger.warning("Invalid total probability, using default distribution")
                features['implied_home_prob'] = 0.4
                features['implied_draw_prob'] = 0.25
                features['implied_away_prob'] = 0.35
            
            # Add all required derived features
            features['win_rate_difference'] = features['home_win_rate'] - features['away_win_rate']
            features['possession_difference'] = features['home_possession'] - features['away_possession']
            features['xg_difference'] = features['home_xg'] - features['away_xg']
            features['shot_difference'] = features['home_shots'] - features['away_shots']
            features['momentum_difference'] = features['home_momentum'] - features['away_momentum']
            features['implied_prob_sum'] = features['implied_home_prob'] + features['implied_draw_prob'] + features['implied_away_prob']
            features['form_difference'] = features['home_form_points'] - features['away_form_points']
            features['odds_ratio'] = features['odds_home_win'] / max(0.1, features['odds_away_win'])
            features['total_momentum'] = features['home_momentum'] + features['away_momentum']
            
            # All features needed by the model
            training_features = [
                'season', 'competition_id', 'home_total_matches', 'away_total_matches',
                'home_win_rate', 'away_win_rate', 'home_form_points', 'away_form_points',
                'home_shots', 'away_shots', 'home_shots_on_target', 'away_shots_on_target',
                'home_corners', 'away_corners', 'home_fouls', 'away_fouls',
                'home_possession', 'away_possession', 'home_xg', 'away_xg',
                'shot_accuracy_home', 'shot_accuracy_away', 'home_win_rate_ratio',
                'home_momentum', 'away_momentum', 'odds_home_win', 'odds_draw',
                'odds_away_win', 'implied_home_prob', 'implied_draw_prob',
                'implied_away_prob', 'form_difference', 'win_rate_difference',
                'shot_difference', 'possession_difference', 'xg_difference',
                'total_momentum', 'momentum_difference', 'odds_ratio', 'implied_prob_sum'
            ]

            # Add any missing training features with default values
            for feature in training_features:
                if feature not in features:
                    if feature in ['season', 'competition_id', 'home_total_matches', 'away_total_matches']:
                        features[feature] = 0
                    elif feature in ['shot_accuracy_home', 'shot_accuracy_away', 'home_win_rate_ratio']:
                        features[feature] = 0.5
                    else:
                        features[feature] = 0

            # Convert to DataFrame with proper feature order
            df = pd.DataFrame([features])
            
            # Ensure columns are in the correct order
            df = df[training_features]
            
            return df
        
        except Exception as e:
            logger.error(f"Error in create_match_features_from_api: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Error creating match features: {str(e)}")
        return None

def adjust_probabilities(home_prob, draw_prob, away_prob, match_data):
    """Adjust probabilities based on odds and team strengths. Input and output are decimals (0-1)"""
    try:
        # Get odds with safe defaults
        home_odds = float(match_data.get('odds_ft_1', 2.0))
        away_odds = float(match_data.get('odds_ft_2', 2.0))
        draw_odds = float(match_data.get('odds_ft_x', 3.0))
        
        # Ensure odds are valid positive numbers
        home_odds = max(1.1, home_odds)
        away_odds = max(1.1, away_odds)
        draw_odds = max(1.1, draw_odds)
        
        # Convert odds to probabilities
        odds_home_prob = 1 / home_odds
        odds_away_prob = 1 / away_odds
        odds_draw_prob = 1 / draw_odds
        
        # Normalize odds probabilities
        total_odds_prob = odds_home_prob + odds_away_prob + odds_draw_prob
        if total_odds_prob > 0:
            odds_home_prob /= total_odds_prob
            odds_away_prob /= total_odds_prob
            odds_draw_prob /= total_odds_prob
        else:
            odds_home_prob = 0.4
            odds_draw_prob = 0.25
            odds_away_prob = 0.35
        
        # Get team strengths
        home_ppg = float(match_data.get('home_ppg', 1.5))
        away_ppg = float(match_data.get('away_ppg', 1.5))
        home_overall_ppg = float(match_data.get('pre_match_teamA_overall_ppg', 1.5))
        away_overall_ppg = float(match_data.get('pre_match_teamB_overall_ppg', 1.5))
        
        # Calculate form-based probabilities
        total_ppg = home_overall_ppg + away_overall_ppg
        if total_ppg > 0:
            form_home_prob = home_overall_ppg / total_ppg
            form_away_prob = away_overall_ppg / total_ppg
            form_draw_prob = 0.25  # Base draw probability
            
            # Normalize form probabilities
            total_form = form_home_prob + form_away_prob + form_draw_prob
            form_home_prob /= total_form
            form_away_prob /= total_form
            form_draw_prob /= total_form
        else:
            form_home_prob = 0.4
            form_draw_prob = 0.25
            form_away_prob = 0.35
        
        # Weights for different factors
        model_weight = 0.5  # Increased weight for model predictions
        odds_weight = 0.3  # Reduced weight for odds since they may be missing
        form_weight = 0.2
        
        # Calculate final probabilities
        final_home_prob = (home_prob * model_weight + 
                        odds_home_prob * odds_weight + 
                        form_home_prob * form_weight)
        
        final_away_prob = (away_prob * model_weight + 
                        odds_away_prob * odds_weight + 
                        form_away_prob * form_weight)
        
        final_draw_prob = (draw_prob * model_weight + 
                        odds_draw_prob * odds_weight + 
                        form_draw_prob * form_weight)
        
        # Normalize final probabilities
        total = final_home_prob + final_draw_prob + final_away_prob
        if total > 0:
            final_home_prob = float(final_home_prob / total)
            final_away_prob = float(final_away_prob / total)
            final_draw_prob = float(final_draw_prob / total)
        else:
            final_home_prob = 0.4
            final_draw_prob = 0.25
            final_away_prob = 0.35
        
        # Apply minimum probability thresholds
        min_prob = 0.1
        if final_home_prob < min_prob:
            final_home_prob = min_prob
            remaining = 1 - min_prob
            final_draw_prob = remaining * 0.4
            final_away_prob = remaining * 0.6
        elif final_away_prob < min_prob:
            final_away_prob = min_prob
            remaining = 1 - min_prob
            final_home_prob = remaining * 0.6
            final_draw_prob = remaining * 0.4
        elif final_draw_prob < min_prob:
            final_draw_prob = min_prob
            remaining = 1 - min_prob
            final_home_prob = remaining * 0.5
            final_away_prob = remaining * 0.5
            
        # Ensure probabilities sum to 1
        total = final_home_prob + final_draw_prob + final_away_prob
        final_home_prob = float(final_home_prob / total)
        final_draw_prob = float(final_draw_prob / total)
        final_away_prob = float(final_away_prob / total)
        
        return final_home_prob, final_draw_prob, final_away_prob
        
    except Exception as e:
        logger.error(f"Error adjusting probabilities: {str(e)}")
        return 0.4, 0.25, 0.35  # Return reasonable defaults

def calculate_form(recent_matches, team):
    if recent_matches.empty:
        return 0
    
    form = 0
    max_form = len(recent_matches) * 3
    for _, match in recent_matches.iterrows():
        if match['Team 1'] == team:
            result = match['FT'].split('-')
            form += 3 if int(result[0]) > int(result[1]) else (1 if result[0] == result[1] else 0)
        else:
            result = match['FT'].split('-')
            form += 3 if int(result[1]) > int(result[0]) else (1 if result[1] == result[0] else 0)
    return form / max_form

def calculate_goals(recent_matches, team):
    if recent_matches.empty:
        return {'diff': 0, 'total': 0}
    
    goals_for = 0
    goals_against = 0
    for _, match in recent_matches.iterrows():
        result = match['FT'].split('-')
        if match['Team 1'] == team:
            goals_for += int(result[0])
            goals_against += int(result[1])
        else:
            goals_for += int(result[1])
            goals_against += int(result[0])
    
    num_matches = len(recent_matches)
    return {
        'diff': (goals_for - goals_against) / num_matches,
        'total': (goals_for + goals_against) / num_matches
    }

def get_matches_for_days(start_date, end_date):
    all_matches = []
    current_date = start_date
    while current_date <= end_date:
        matches = get_matches(current_date.strftime('%Y-%m-%d'))
        all_matches.extend(matches)
        current_date += timedelta(days=1)
    return all_matches

def get_matches_for_date(date):
    """Get matches for a specific date."""
    start_date = date
    end_date = date
    return get_matches_for_days(start_date, end_date)

def extract_league_name(match_url):
    """Extract league name from match URL."""
    if not match_url:
        return "Unknown League"
    # Example URL: /saudi-arabia/al-fateh-sc-vs-al-wahda-fc-mecca-h2h-stats
    parts = match_url.split('/')
    if len(parts) > 1:
        return parts[1]  # returns 'saudi-arabia'
    return "Unknown League"

def calculate_match_prediction(match):
    """Calculate match prediction using multiple factors"""
    
    # Get basic odds
    home_odds = float(match.get('odds_ft_1', 0))
    draw_odds = float(match.get('odds_ft_x', 0))
    away_odds = float(match.get('odds_ft_2', 0))
    
    # Get team performance metrics
    home_ppg = float(match.get('home_ppg', 0))
    away_ppg = float(match.get('away_ppg', 0))
    home_overall_ppg = float(match.get('pre_match_teamA_overall_ppg', 0))
    away_overall_ppg = float(match.get('pre_match_teamB_overall_ppg', 0))
    
    # Get expected goals (xG)
    home_xg = float(match.get('team_a_xg_prematch', 0))
    away_xg = float(match.get('team_b_xg_prematch', 0))
    
    # Calculate probabilities from odds (if available)
    if all([home_odds, draw_odds, away_odds]):
        # Convert odds to probabilities
        total_prob = (1/home_odds + 1/draw_odds + 1/away_odds)
        home_prob_odds = (1/home_odds) / total_prob
        draw_prob_odds = (1/draw_odds) / total_prob
        away_prob_odds = (1/away_odds) / total_prob
        
        # Debug prints
        print(f"Home - Pred: {home_prob_odds*100:.2f}%, Odds: {home_odds:.2f}")
        print(f"Draw - Pred: {draw_prob_odds*100:.2f}%, Odds: {draw_odds:.2f}")
        print(f"Away - Pred: {away_prob_odds*100:.2f}%, Odds: {away_odds:.2f}")
        
        home_ev = calculate_ev(home_prob_odds*100, home_odds)
        draw_ev = calculate_ev(draw_prob_odds*100, draw_odds)
        away_ev = calculate_ev(away_prob_odds*100, away_odds)
        
        # Debug prints
        print(f"Home EV: {home_ev:.2f}%")
        print(f"Draw EV: {draw_ev:.2f}%")
        print(f"Away EV: {away_ev:.2f}%")
    else:
        home_prob_odds = draw_prob_odds = away_prob_odds = 0
        home_ev = draw_ev = away_ev = 0
    
    # Calculate form-based probabilities
    total_ppg = home_overall_ppg + away_overall_ppg
    if total_ppg > 0:
        home_prob_form = home_overall_ppg / total_ppg
        away_prob_form = away_overall_ppg / total_ppg
    else:
        home_prob_form = 0.4
        away_prob_form = 0.4
    form_draw_prob = 1 - (home_prob_form + away_prob_form)
    
    # Calculate xG-based probabilities
    total_xg = home_xg + away_xg
    if total_xg > 0:
        home_prob_xg = home_xg / total_xg
        away_prob_xg = away_xg / total_xg
    else:
        home_prob_xg = 0.5
        away_prob_xg = 0.5
    
    # Weighted combination of all factors
    # Odds are given highest weight as they incorporate market knowledge
    home_final = (home_prob_odds * 0.5) + (home_prob_form * 0.25) + (home_prob_xg * 0.25)
    away_final = (away_prob_odds * 0.5) + (away_prob_form * 0.25) + (away_prob_xg * 0.25)
    draw_final = 1 - (home_final + away_final)  # Ensure probabilities sum to 1
    
    return home_final, draw_final, away_final

def display_prediction(prediction, confidence):
    """Display prediction with appropriate color based on confidence level"""
    if confidence >= 70:  
        sentiment_class = "prediction-high"
        confidence_text = "High Confidence"
    elif confidence >= 50:  
        sentiment_class = "prediction-medium"
        confidence_text = "Medium Confidence"
    else:
        sentiment_class = "prediction-low"
        confidence_text = "Low Confidence"
    
    st.markdown(f"""
        <div class="prediction-wrapper {sentiment_class}">
            <div class="prediction-text">
                Prediction: {prediction}
            </div>
            <div class="confidence-text">
                {confidence_text} ({confidence:.1f}%)
            </div>
        </div>
    """, unsafe_allow_html=True)

def display_probability_bars(home_prob, draw_prob, away_prob, home_team, away_team):
    """Display probability bars for match outcomes"""
    st.markdown("### Match Outcome Probabilities")
    
    # Create three columns for the probabilities
    col1, col2, col3 = st.columns(3)
    
    # Home team probability
    with col1:
        st.markdown(f"""
            <div style="text-align: center;">
                <div style="margin-bottom: 0.5rem;">
                    <span style="color: #48bb78; font-weight: 600; font-size: 1.2rem;">{home_prob*100:.1f}%</span>
                </div>
                <div style="color: #1a1a1a; font-weight: 500;">{home_team}</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Draw probability
    with col2:
        st.markdown(f"""
            <div style="text-align: center;">
                <div style="margin-bottom: 0.5rem;">
                    <span style="color: #ed8936; font-weight: 600; font-size: 1.2rem;">{draw_prob*100:.1f}%</span>
                </div>
                <div style="color: #1a1a1a; font-weight: 500;">Draw</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Away team probability
    with col3:
        st.markdown(f"""
            <div style="text-align: center;">
                <div style="margin-bottom: 0.5rem;">
                    <span style="color: #3182ce; font-weight: 600; font-size: 1.2rem;">{away_prob*100:.1f}%</span>
                </div>
                <div style="color: #1a1a1a; font-weight: 500;">{away_team}</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Add the combined progress bar
    st.markdown(f"""
        <div style="margin-top: 1rem; padding: 0 1rem;">
            <div style="width: 100%; height: 20px; background: #e2e8f0; border-radius: 10px; overflow: hidden; display: flex;">
                <div style="width: {home_prob*100}%; height: 100%; background-color: #48bb78;"></div>
                <div style="width: {draw_prob*100}%; height: 100%; background-color: #ed8936;"></div>
                <div style="width: {away_prob*100}%; height: 100%; background-color: #3182ce;"></div>
            </div>
        </div>
        
        <div style="display: flex; justify-content: space-between; margin-top: 1rem; padding: 0 1rem;">
            <div style="display: flex; align-items: center;">
                <div style="width: 10px; height: 10px; background: #48bb78; border-radius: 2px; margin-right: 5px;"></div>
                <span style="font-size: 0.8rem;">Home Win</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 10px; height: 10px; background: #ed8936; border-radius: 2px; margin-right: 5px;"></div>
                <span style="font-size: 0.8rem;">Draw</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 10px; height: 10px; background: #3182ce; border-radius: 2px; margin-right: 5px;"></div>
                <span style="font-size: 0.8rem;">Away Win</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

def display_market_values(home_team, away_team):
    """Display a button to load market values for both teams"""
    button_key = f"load_market_values_{home_team}_{away_team}".replace(" ", "_")
    
    # Use session state to track if market values should be shown
    if button_key not in st.session_state:
        st.session_state[button_key] = False
    
    # Show the load button if values aren't loaded yet
    if not st.session_state[button_key]:
        if st.button(f"💰 Show Market Values", key=f"btn_{button_key}"):
            st.session_state[button_key] = True
            st.rerun()
        return
    
    # If button was clicked, show loading and fetch values
    with st.spinner("Loading market values..."):
        try:
            logger.info(f"Fetching market values for {home_team} vs {away_team}")
            home_value, away_value = get_market_values(home_team, away_team)
            
            # Format values for display
            def format_value(value):
                if value is None:  
                    return 'N/A'
                # Handle integer values (in euros)
                try:
                    market_value = value.get('market_value', 'N/A') if isinstance(value, dict) else value
                    if market_value == 'N/A':
                        return market_value
                    if isinstance(market_value, (int, float)):
                        return f"€{market_value/1000000:.1f}M"
                    # Handle string values like '10.5m €'
                    num_value = float(market_value.replace('m €', '').strip())
                    return f"€{num_value:.1f}M"
                except (AttributeError, ValueError, TypeError):
                    return market_value if market_value != 'N/A' else 'N/A'
                    
            formatted_home_value = format_value(home_value)
            formatted_away_value = format_value(away_value)
            
            st.markdown(f"""
                <div style="background: #ffffff; 
                            padding: 1rem; 
                            border-radius: 8px; 
                            margin: 1rem 0;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="text-align: center; flex: 1;">
                            <div style="font-weight: 600; color: #4a5568; margin-bottom: 0.5rem; font-size: 0.9rem;">
                                {home_team}
                            </div>
                            <span style="font-weight: 700; 
                                       color: #2c5282;
                                       font-size: 1.1rem;">
                                {formatted_home_value}
                            </span>
                        </div>
                        <div style="font-weight: 600; color: #4a5568;">vs</div>
                        <div style="text-align: center; flex: 1;">
                            <div style="font-weight: 600; color: #4a5568; margin-bottom: 0.5rem; font-size: 0.9rem;">
                                {away_team}
                            </div>
                            <span style="font-weight: 700; 
                                       color: #2c5282;
                                       font-size: 1.1rem;">
                                {formatted_away_value}
                            </span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Add a button to hide the values
            if st.button("❌ Hide Market Values", key=f"hide_{button_key}"):
                st.session_state[button_key] = False
                st.rerun()
                
            logger.info("Successfully displayed market values")
            
        except Exception as e:
            st.error("❌ Failed to load market values. Please try again.")
            logger.error(f"Error displaying market values: {str(e)}")
            logger.exception("Full traceback:")
            
            # Reset the button state on error
            st.session_state[button_key] = False
            st.rerun()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_market_values(home_team, away_team):
    """Get market values for both teams with caching"""
    logger.info(f"Fetching market values for {home_team} vs {away_team}")
    try:
        api = TransfermarktAPI()
        # Get market values using the new method
        market_values = api.get_both_teams_market_value(home_team, away_team)
        
        # Extract values from the dictionary
        home_value = market_values.get('home_market_value', 0)
        away_value = market_values.get('away_market_value', 0)
        
        # Format values to millions with 1 decimal place
        def format_value(value):
            if value is None or value == 0:
                return 'N/A'
            return f"€{value/1000000:.1f}M"
        
        formatted_home = format_value(home_value)
        formatted_away = format_value(away_value)
        
        logger.info(f"Retrieved market values - Home: {formatted_home}, Away: {formatted_away}")
        return formatted_home, formatted_away
    except Exception as e:
        logger.error(f"Error getting market values: {str(e)}")
        logger.exception("Full traceback:")
        return 'N/A', 'N/A'

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_multiple_market_values(teams):
    """Get market values for multiple teams with caching"""
    api = TransfermarktAPI(max_workers=40)  # Increased to 20 worker threads for better parallelization
    return api.get_multiple_teams_market_value(teams)

def display_match_odds(match_data):
    """Display FootyStats match odds in an organized box."""
    # Check if odds are from Supabase (Swisslos)
    odds_source = match_data.get('odds_source', '')
    
    # Display header with source information if from Swisslos
    if odds_source == 'supabase':
        st.markdown("""
            <h3 style="text-align: center; color: #1f2937; margin: 20px 0; font-size: 1.5rem; background-color: #e6f7ff; padding: 10px; border-radius: 5px;">Odds from Swisslos</h3>
        """, unsafe_allow_html=True)
        logger.info(f"Displaying odds from Swisslos for {match_data.get('home_name', '')} vs {match_data.get('away_name', '')}")
    else:
        st.markdown("""
            <h3 style="text-align: center; color: #1f2937; margin: 20px 0; font-size: 1.5rem;">Odds from fotty</h3>
        """, unsafe_allow_html=True)
        logger.info(f"Displaying odds from system for {match_data.get('home_name', '')} vs {match_data.get('away_name', '')}")
    
    # Get predicted probabilities for all markets
    home_pred = float(match_data.get('home_prob', 0)) 
    draw_pred = float(match_data.get('draw_prob', 0)) 
    away_pred = float(match_data.get('away_prob', 0)) 
    
    # Match outcome odds
    home_odds = float(match_data.get('odds_ft_1', 0))
    draw_odds = float(match_data.get('odds_ft_x', 0))
    away_odds = float(match_data.get('odds_ft_2', 0))
    
    # Calculate implied probabilities and EV for match outcomes
    if all([home_odds, draw_odds, away_odds]):
        # Convert odds to probabilities
        total_prob = (1/home_odds + 1/draw_odds + 1/away_odds)
        home_implied = (1/home_odds) / total_prob
        draw_implied = (1/draw_odds) / total_prob
        away_implied = (1/away_odds) / total_prob
        
        # Debug prints
        print(f"Home - Pred: {home_pred*100:.2f}%, Odds: {home_odds:.2f}")
        print(f"Draw - Pred: {draw_pred*100:.2f}%, Odds: {draw_odds:.2f}")
        print(f"Away - Pred: {away_pred*100:.2f}%, Odds: {away_odds:.2f}")
        
        home_ev = calculate_ev(home_pred*100, home_odds)
        draw_ev = calculate_ev(draw_pred*100, draw_odds)
        away_ev = calculate_ev(away_pred*100, away_odds)
        
        # Debug prints
        print(f"Home EV: {home_ev:.2f}%")
        print(f"Draw EV: {draw_ev:.2f}%")
        print(f"Away EV: {away_ev:.2f}%")
    else:
        home_implied = draw_implied = away_implied = 0
        home_ev = draw_ev = away_ev = 0
    
    # Display match information
    st.markdown("""
        <style>
            .odds-container {
                display: flex;
                justify-content: space-between;
                margin-bottom: 20px;
            }
            .odds-box {
                flex: 1;
                margin: 0 10px;
                padding: 0.5rem;
                border-radius: 6px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
        </style>
    """, unsafe_allow_html=True)

    # Display match odds
    col1, col2, col3 = st.columns(3)
    with col1:
        display_odds_box("Home Win", home_odds, home_implied, home_ev)
    with col2:
        display_odds_box("Draw", draw_odds, draw_implied, draw_ev)
    with col3:
        display_odds_box("Away Win", away_odds, away_implied, away_ev)

    # Calculate and display Over/Under odds
    try:
        over = float(match_data.get('odds_ft_over25', 0))
        under = float(match_data.get('odds_ft_under25', 0))
        if over > 0 and under > 0:
            # Get our predicted probabilities
            over_pred = match_data.get('over25_prob', 0) * 100  # Convert to percentage
            under_pred = match_data.get('under25_prob', 0) * 100
            
            # Calculate implied probabilities
            over_raw_prob = 1 / over
            under_raw_prob = 1 / under
            total_prob = over_raw_prob + under_raw_prob
            over_implied = (over_raw_prob / total_prob) * 100
            under_implied = (under_raw_prob / total_prob) * 100
            
            # Calculate EV using the same method as match outcomes
            over_ev = calculate_ev(over_pred, over)
            under_ev = calculate_ev(under_pred, under)
            
            # Display Over/Under odds
            col6, col7 = st.columns(2)
            with col6:
                display_odds_box("Over 2.5", over, over_implied, over_ev)
            with col7:
                display_odds_box("Under 2.5", under, under_implied, under_ev)
    except Exception as e:
        print(f"Error processing Over/Under: {str(e)}")

    # Calculate and display BTTS odds
    try:
        btts_yes = float(match_data.get('odds_btts_yes', 0))
        btts_no = float(match_data.get('odds_btts_no', 0))
        if btts_yes > 0 and btts_no > 0:
            # Get our predicted probabilities
            btts_pred = match_data.get('btts_prob', 0) * 100  # Convert to percentage
            btts_no_pred = match_data.get('btts_no_prob', 0) * 100
            
            # Calculate implied probabilities
            btts_raw_prob = 1 / btts_yes
            btts_no_raw_prob = 1 / btts_no
            total_btts_prob = btts_raw_prob + btts_no_raw_prob
            btts_implied = (btts_raw_prob / total_btts_prob) * 100
            btts_no_implied = (btts_no_raw_prob / total_btts_prob) * 100
            
            # Calculate EV using the same method as match outcomes
            btts_ev = calculate_ev(btts_pred, btts_yes)
            btts_no_ev = calculate_ev(btts_no_pred, btts_no)
            
            # Display BTTS odds
            col8, col9 = st.columns(2)
            with col8:
                display_odds_box("BTTS Yes", btts_yes, btts_implied, btts_ev)
            with col9:
                display_odds_box("BTTS No", btts_no, btts_no_implied, btts_no_ev)
    except Exception as e:
        print(f"Error processing BTTS: {str(e)}")

def get_match_prediction(match_data):
    """Calculate match prediction using the loaded model"""
    try:
        logger.info("Starting prediction calculation")
        logger.info(f"Match data: {match_data}")
        
        # First check if we already have probabilities from the odds generator
        home_prob = match_data.get('home_prob')
        draw_prob = match_data.get('draw_prob')
        away_prob = match_data.get('away_prob')
        
        # Check if we have valid probabilities already
        valid_probs = (home_prob is not None and draw_prob is not None and away_prob is not None and
            all(isinstance(p, (int, float)) for p in [home_prob, draw_prob, away_prob]) and
            all(0 <= p <= 1 for p in [home_prob, draw_prob, away_prob]))
        
        # If we don't have valid probabilities, try to get them from Supabase football_odds table
        if not valid_probs:
            try:
                # Get team names and league name
                home_team = match_data.get('home_name', '')
                away_team = match_data.get('away_name', '')
                league_name = get_league_name(match_data)
                
                # Try to get odds from Supabase
                logger.info(f"Trying to get odds from Supabase for {home_team} vs {away_team} in {league_name}")
                
                # The team name and league mapping is now handled in odds_fetcher.py
                odds_data = odds_fetcher.get_odds_from_db(home_team, away_team, league_name)
                
                if odds_data:
                    logger.info(f"Found odds in Supabase: {odds_data}")
                    # Convert odds to probabilities
                    probs = odds_fetcher.convert_odds_to_probabilities(odds_data)
                    
                    if probs:
                        logger.info(f"Using probabilities from Supabase: {probs}")
                        home_prob = probs['home_prob']
                        draw_prob = probs['draw_prob']
                        away_prob = probs['away_prob']
                        valid_probs = True
                        
                        # Store the odds in match_data for later use
                        match_data['odds_ft_1'] = odds_data['home_odds']
                        match_data['odds_ft_x'] = odds_data['draw_odds']
                        match_data['odds_ft_2'] = odds_data['away_odds']
                        
                        # Set the odds source to 'supabase' to indicate these are from Swisslos
                        match_data['odds_source'] = 'supabase'
                        logger.info(f"Set odds_source to 'supabase' for {home_team} vs {away_team}")
                        
                        # Also set additional odds data
                        if 'over25_odds' in odds_data:
                            match_data['odds_over'] = odds_data['over25_odds']
                        if 'under25_odds' in odds_data:
                            match_data['odds_under'] = odds_data['under25_odds']
                        if 'btts_yes_odds' in odds_data:
                            match_data['odds_btts_yes'] = odds_data['btts_yes_odds']
                        if 'btts_no_odds' in odds_data:
                            match_data['odds_btts_no'] = odds_data['btts_no_odds']
                else:
                    logger.info(f"No odds found in Supabase for {home_team} vs {away_team}")
                
                # Print debug info about odds_data and valid_probs
                logger.info(f"odds_data: {odds_data if odds_data else 'None'}")
                logger.info(f"valid_probs: {valid_probs}")
                if odds_data:
                    logger.info(f"'source' in odds_data: {'source' in odds_data}")
                    if 'source' in odds_data:
                        logger.info(f"odds_data['source'] == 'supabase': {odds_data['source'] == 'supabase'}")
                
                # Additional debug info for match data
                logger.info(f"match_data['odds_source']: {match_data.get('odds_source', 'not set')}")
                logger.info(f"Final odds for {home_team} vs {away_team}: Home={match_data.get('odds_ft_1', 'N/A')}, Draw={match_data.get('odds_ft_x', 'N/A')}, Away={match_data.get('odds_ft_2', 'N/A')}")
                
                # We've already set the odds_source in the code above, so we don't need to do it again here
            except Exception as e:
                logger.error(f"Error getting odds from Supabase: {str(e)}")
                # Continue with existing odds system if there's an error
        
        # If we have valid probabilities (either from match_data or from Supabase), use those
        if valid_probs:
            
            # Convert to float and normalize
            home_prob = float(home_prob)
            draw_prob = float(draw_prob)
            away_prob = float(away_prob)
            
            total = home_prob + draw_prob + away_prob
            if total > 0:
                home_prob /= total
                draw_prob /= total
                away_prob /= total
                
                logger.info(f"Using probabilities from odds generator: {home_prob}, {draw_prob}, {away_prob}")
                return home_prob, draw_prob, away_prob
        
        # If we don't have valid probabilities, use the model
        if predictor is None:
            logger.error("Model not loaded properly")
            return 0.4, 0.25, 0.35  # Return reasonable defaults
            
        # Create features DataFrame
        try:
            features_df = create_match_features_from_api(match_data)
        except Exception as e:
            logger.error(f"Error creating features: {str(e)}")
            return 0.4, 0.25, 0.35  # Return reasonable defaults if feature creation fails
        
        # Convert DataFrame to DMatrix
        try:
            dmatrix = xgb.DMatrix(features_df)
        except Exception as e:
            logger.error(f"Error creating DMatrix: {str(e)}")
            return 0.4, 0.25, 0.35

        # Make prediction using DMatrix
        try:
            probabilities = predictor.predict(dmatrix)
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return 0.4, 0.25, 0.35
        
        # If the model outputs a single probability, convert it to three probabilities
        if len(probabilities.shape) == 1:
            home_prob = float(probabilities[0])
            draw_prob = 0.25  # Default draw probability
            away_prob = float(1 - home_prob - draw_prob)  # Remaining probability
        else:
            home_prob = float(probabilities[0][0])
            draw_prob = float(probabilities[0][1])
            away_prob = float(probabilities[0][2])
        
        # Ensure probabilities are valid
        if not all(0 <= p <= 1 for p in [home_prob, draw_prob, away_prob]):
            logger.warning("Invalid probability values from model, using defaults")
            home_prob = 0.4
            draw_prob = 0.25
            away_prob = 0.35
            
        # Normalize probabilities to sum to 1
        total = home_prob + draw_prob + away_prob
        if total > 0:
            home_prob = float(home_prob / total)
            draw_prob = float(draw_prob / total)
            away_prob = float(away_prob / total)
        else:
            logger.warning("Zero total probability, using defaults")
            home_prob = 0.4
            draw_prob = 0.25
            away_prob = 0.35
        
        # Adjust probabilities based on odds and team strengths
        try:
            home_prob, draw_prob, away_prob = adjust_probabilities(
                home_prob, draw_prob, away_prob, match_data
            )
        except Exception as e:
            logger.error(f"Error adjusting probabilities: {str(e)}")
            # If adjustment fails, keep the normalized probabilities
        
        return home_prob, draw_prob, away_prob
        
    except Exception as e:
        logger.error(f"Error in get_match_prediction: {str(e)}")
        return 0.4, 0.25, 0.35  # Return reasonable defaults instead of None

def normalize_probabilities(probs):
    """Normalize probabilities to sum to 100"""
    total = sum(probs)
    if total == 0:
        return [0] * len(probs)
    return [p * 100 / total for p in probs]

def update_match_results():
    """Update completed match results and calculate profits/losses"""
    history = PredictionHistory()
    
    # Get pending predictions from Supabase
    pending_result = history.db.supabase.table('predictions').select('*').eq('status', 'Pending').execute()
    pending_predictions = pending_result.data
    
    for pred in pending_predictions:
        try:
            # Find the match
            match = get_match_by_teams(
                pred['home_team'],
                pred['away_team'],
                pred['date']
            )
            
            if not match:
                logger.warning(f"Match not found for {pred['home_team']} vs {pred['away_team']} on {pred['date']}")
                continue
                
            # Get the result
            match_data = get_match_result(match['id'])
            
            if not match_data:
                logger.warning(f"No result data for match ID {match['id']}")
                continue
                
            if match_data['status'] == 'FINISHED':
                # Determine winner
                try:
                    home_score = int(match_data['home_score'])
                    away_score = int(match_data['away_score'])
                    
                    if home_score > away_score:
                        actual_outcome = "HOME"
                    elif away_score > home_score:
                        actual_outcome = "AWAY"
                    else:
                        actual_outcome = "DRAW"
                    
                    # Calculate profit/loss with fixed bet amount
                    if actual_outcome == pred['predicted_outcome']:
                        if actual_outcome == "HOME":
                            profit = float(pred['home_odds']) - 1
                        elif actual_outcome == "AWAY":
                            profit = float(pred['away_odds']) - 1
                        else:
                            profit = float(pred['draw_odds']) - 1
                    else:
                        profit = -1
                    
                    # Update prediction result in Supabase
                    history.db.update_prediction_result(
                        pred['id'],
                        actual_outcome,
                        profit,
                        home_score,
                        away_score
                    )
                except (ValueError, TypeError) as e:
                    logger.error(f"Error processing scores for match {match['id']}: {str(e)}")
                    continue
            else:
                logger.info(f"Match not complete, status: {match_data['status']}")
                
        except Exception as e:
            logger.error(f"Error updating match result: {str(e)}")
            continue

def process_match_prediction(match):
    """Process and save prediction for a match. Returns (prediction_data, confidence) tuple if successful, None otherwise"""
    try:
        # Convert unix timestamp to datetime
        match_date = datetime.fromtimestamp(match['date_unix']).date()
        today = datetime.now().date()
        
        # Get predictions (these are in decimal form 0-1)
        home_prob, draw_prob, away_prob = get_match_prediction(match)
        
        # Handle None probabilities
        if any(prob is None for prob in [home_prob, draw_prob, away_prob]):
            logger.error("Invalid probabilities returned from get_match_prediction")
            return None, None
        
        # Calculate Over 2.5 and BTTS probabilities
        over25_prob = calculate_over25_probability(match.get('team_a_xg_prematch', 0), match.get('team_b_xg_prematch', 0))
        btts_prob = calculate_btts_probability(match.get('team_a_xg_prematch', 0), match.get('team_b_xg_prematch', 0))
        
        # Store all probabilities in match data
        match['home_prob'] = home_prob
        match['draw_prob'] = draw_prob
        match['away_prob'] = away_prob
        match['over25_prob'] = over25_prob if over25_prob is not None else 0.5
        match['under25_prob'] = 1 - (over25_prob if over25_prob is not None else 0.5)
        match['btts_prob'] = btts_prob if btts_prob is not None else 0.5
        match['btts_no_prob'] = 1 - (btts_prob if btts_prob is not None else 0.5)

        # Determine predicted outcome and confidence based on probability margins
        probs = [home_prob, draw_prob, away_prob]
        max_prob = max(probs)
        sorted_probs = sorted(probs, reverse=True)
        margin = sorted_probs[0] - sorted_probs[1]  # Margin between highest and second highest
        
        # Calculate confidence based on both margin and absolute probability
        # 1. Base confidence from absolute probability (max 60%)
        base_confidence = max_prob * 60  # e.g. if highest prob is 0.60, base is 36%
        
        # 2. Additional confidence from margin (max 40%)
        # Margin of 0.20 (20%) or more gets full 40%
        margin_confidence = min(margin * 200, 40)  # margin * 200 means 0.20 margin = 40%
        
        # 3. Total confidence (max 100%)
        confidence = base_confidence + margin_confidence
        
        # Determine outcome
        if max_prob == home_prob:
            predicted_outcome = "HOME"
        elif max_prob == away_prob:
            predicted_outcome = "AWAY"
        else:
            predicted_outcome = "DRAW"
        
        # Get league name with fallback
        league_name = get_league_name(match)
        if not league_name:
            # Extract from URL as fallback
            league_name = extract_league_name(match.get('match_url', ''))
        
        # Get match ID
        match_id = match.get('id')
        if not match_id:
            logger.error(f"No match ID found for {match['home_name']} vs {match['away_name']}")
            return None, None

        # Create prediction data
        prediction_data = {
            'date': match_date.strftime('%Y-%m-%d'),
            'league': league_name,
            'home_team': match['home_name'],
            'away_team': match['away_name'],
            'predicted_outcome': predicted_outcome,
            'confidence': float(confidence),  # Ensure confidence is stored as float
            'home_odds': match.get('odds_ft_1', 0),
            'draw_odds': match.get('odds_ft_x', 0),
            'away_odds': match.get('odds_ft_2', 0),
            'bet_amount': 10.0,  # Default bet amount
            'status': 'Pending',
            'match_id': str(match_id)  # Convert to string but don't generate fallback
        }
        
        # Only store in database if it's today's match
        if match_date == today:
            # Check for existing prediction
            history = PredictionHistory()
            existing_predictions = history.db.supabase.table('predictions').select('*').eq('date', prediction_data['date']).eq('home_team', prediction_data['home_team']).eq('away_team', prediction_data['away_team']).execute()
            
            # Check if prediction already exists for this match
            match_exists = len(existing_predictions.data) > 0
            
            # Only add if prediction doesn't exist
            if not match_exists:
                history.db.add_prediction(prediction_data)
        
        return prediction_data, confidence
        
    except Exception as e:
        logger.error(f"Error in process_match_prediction: {str(e)}")
        return None, None

def display_match_details(match, prediction_data, confidence):
    """Display match details, prediction, and odds"""
    try:
        # Handle case where prediction_data is None
        if prediction_data is None:
            logger.warning("No prediction data available")
            st.warning("Unable to display match details - prediction data not available")
            return
            
        # Get league name with fallback
        league_name = prediction_data.get('league')
        if not league_name:
            league_name = get_league_name(match)
        if not league_name:
            league_name = extract_league_name(match.get('match_url', ''))
        if not league_name:
            league_name = "Unknown League"
            
        # Get team names
        home_team = match.get('home_name', 'Home Team')
        away_team = match.get('away_name', 'Away Team')
        
        # Get team logos
        home_logo = get_team_logo_path(home_team)
        away_logo = get_team_logo_path(away_team)
        
        # Convert logo paths to base64 if they exist
        home_logo_html = ''
        away_logo_html = ''
        
        try:
            if home_logo and os.path.exists(home_logo):
                with open(home_logo, "rb") as f:
                    home_encoded = base64.b64encode(f.read()).decode()
                    home_logo_html = f'<img src="data:image/png;base64,{home_encoded}" class="team-logo" alt="{home_team} logo">'
            else:
                home_logo_html = f'<div class="team-logo-placeholder">{home_team[:3].upper()}</div>'
                
            if away_logo and os.path.exists(away_logo):
                with open(away_logo, "rb") as f:
                    away_encoded = base64.b64encode(f.read()).decode()
                    away_logo_html = f'<img src="data:image/png;base64,{away_encoded}" class="team-logo" alt="{away_team} logo">'
            else:
                away_logo_html = f'<div class="team-logo-placeholder">{away_team[:3].upper()}</div>'
        except Exception as e:
            logger.error(f"Error loading team logos: {str(e)}")
            home_logo_html = f'<div class="team-logo-placeholder">{home_team[:3].upper()}</div>'
            away_logo_html = f'<div class="team-logo-placeholder">{away_team[:3].upper()}</div>'
        
        # Add CSS for team display
        st.markdown("""
            <style>
                .match-container {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 10px 0;
                }
                .team-info {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    text-align: center;
                    min-width: 100px;
                }
                .team-logo, .team-logo-placeholder {
                    width: 40px;
                    height: 40px;
                    object-fit: contain;
                    border-radius: 50%;
                    background: white;
                    padding: 2px;
                    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                }
                .team-logo-placeholder {
                    background: #f0f0f0;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: bold;
                    color: #666;
                    font-size: 14px;
                }
                .team-name {
                    font-size: 0.9rem;
                    font-weight: 500;
                    color: #2d3748;
                    margin-top: 2px;
                    line-height: 1.2;
                }
                .vs-badge {
                    margin: 0 10px;
                    font-size: 1rem;
                    font-weight: 500;
                    color: #4a5568;
                }
            </style>
        """, unsafe_allow_html=True)
        
        # Display teams with logos or placeholders
        st.markdown(f"""
            <div class="match-container">
                <div class="team-info">
                    {home_logo_html}
                    <span class="team-name">{home_team}</span>
                </div>
                <span class="vs-badge">VS</span>
                <div class="team-info">
                    {away_logo_html}
                    <span class="team-name">{away_team}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Display prediction
        prediction = f"{match.get('home_name', 'Home Team')} vs {match.get('away_name', 'Away Team')} - {prediction_data.get('predicted_outcome', 'No Prediction')}"
        display_prediction(prediction, confidence or 0)
        
        # Display market values with lazy loading
        try:
            # Display market values with the new lazy loading button
            display_market_values(home_team, away_team)
            
            # Get probabilities for the match outcome
            home_prob = float(match.get('home_prob', 0))
            draw_prob = float(match.get('draw_prob', 0))
            away_prob = float(match.get('away_prob', 0))
            
            # Normalize probabilities
            total = home_prob + draw_prob + away_prob
            if total > 0:
                home_prob = (home_prob / total) * 100
                draw_prob = (draw_prob / total) * 100
                away_prob = (away_prob / total) * 100
            
            # Display match outcome probabilities
            st.markdown(f'''
                <div style="margin: 10px 0 20px 0;">
                    <div style="font-weight: 600; color: #4a5568; margin-bottom: 8px; text-align: center;">Match Outcome Probabilities</div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0; background: white; border-radius: 8px; overflow: hidden; border: 1px solid #e2e8f0;">
                        <div style="padding: 8px; text-align: center; border-right: 1px solid #e2e8f0;">
                            <div style="color: #2d3748; font-weight: 600; font-size: 0.8rem; margin-bottom: 4px;">Home</div>
                            <div style="color: #48bb78; font-size: 1rem; font-weight: 700;">{home_prob:.1f}%</div>
                        </div>
                        <div style="padding: 8px; text-align: center; border-right: 1px solid #e2e8f0;">
                            <div style="color: #2d3748; font-weight: 600; font-size: 0.8rem; margin-bottom: 4px;">Draw</div>
                            <div style="color: #ed8936; font-size: 1rem; font-weight: 700;">{draw_prob:.1f}%</div>
                        </div>
                        <div style="padding: 8px; text-align: center;">
                            <div style="color: #2d3748; font-weight: 600; font-size: 0.8rem; margin-bottom: 4px;">Away</div>
                            <div style="color: #3182ce; font-size: 1rem; font-weight: 700;">{away_prob:.1f}%</div>
                        </div>
                    </div>
                    <div style="width: 100%; height: 4px; background: #edf2f7; border-radius: 2px; overflow: hidden; display: flex; margin-top: 8px;">
                        <div style="width: {home_prob}%; height: 100%; background-color: #48bb78;"></div>
                        <div style="width: {draw_prob}%; height: 100%; background-color: #ed8936;"></div>
                        <div style="width: {away_prob}%; height: 100%; background-color: #3182ce;"></div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)
            
        except Exception as e:
            logger.error(f"Error displaying match probabilities: {str(e)}")
            
        # Display kickoff time
        try:
            kickoff = match.get('kickoff', '')
            if kickoff:
                # Convert to German time
                german_time = convert_to_cet(kickoff)
                
                # Format the date nicely
                st.markdown(f"""
                    <div style="text-align: center;">
                        <div class="league-name">{league_name}</div>
                        <div class="kickoff-time">{german_time} (German Time)</div>
                    </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            logging.error(f"Error displaying kickoff time: {str(e)}")
            st.warning("Error displaying kickoff time")
            
        # Display odds if available
        try:
            display_match_odds(match)
        except Exception as e:
            logger.error(f"Error displaying odds: {str(e)}")
            
        # Close match card container
        st.markdown("</div>", unsafe_allow_html=True)
            
    except Exception as e:
        logger.error(f"Error displaying match details: {str(e)}")
        st.error("An error occurred while displaying match details")

def display_kickoff_time(match_data):
    """Display kickoff time in German timezone"""
    try:
        # Get unix timestamp from match data
        unix_timestamp = match_data.get('date_unix')
        if not unix_timestamp:
            st.warning("Kickoff time not available")
            return

        # Convert unix timestamp to datetime in German timezone
        german_tz = pytz.timezone('Europe/Berlin')
        utc_dt = datetime.fromtimestamp(unix_timestamp, pytz.UTC)
        german_dt = utc_dt.astimezone(german_tz)

        # Format the time
        formatted_time = german_dt.strftime('%A, %d %B %Y %H:%M')
        
        # Display kickoff time with styled box
        st.markdown(f"""
            <div style="width: 100%; max-width: 800px; margin: 5px auto; background-color: white; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; text-align: center;">
                <div style="display: flex; align-items: center; justify-content: center; gap: 8px;">
                    <span style="color: #2d3748; font-weight: 500; font-size: 1.1rem;">
                        {formatted_time} (German Time)
                    </span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        logging.error(f"Error displaying kickoff time: {str(e)}")
        st.warning("Error displaying kickoff time")

def calculate_over25_probability(home_xg, away_xg):
    """Calculate probability of over 2.5 goals using Poisson distribution"""
    try:
        from scipy.stats import poisson
        import numpy as np
        
        print(f"Calculating Over 2.5 probability with xG: home={home_xg}, away={away_xg}")
        
        # Convert xG to float and handle None/invalid values
        try:
            home_xg = float(home_xg) if home_xg not in (None, 0) else 0.5
            away_xg = float(away_xg) if away_xg not in (None, 0) else 0.5
        except (ValueError, TypeError):
            print(f"Error converting xG values: {str(e)}")
            home_xg = 0.5
            away_xg = 0.5
            
        if home_xg <= 0 or away_xg <= 0:
            print("Invalid xG values (<=0)")
            return 0.5
        
        # Calculate probability matrix for total goals
        max_goals = 10
        total_prob = 0
        
        # Calculate probability of 3 or more goals
        for i in range(max_goals):
            for j in range(max_goals):
                if i + j > 2:  # Over 2.5 goals
                    p1 = poisson.pmf(i, home_xg)
                    p2 = poisson.pmf(j, away_xg)
                    total_prob += p1 * p2
        
        print(f"Calculated Over 2.5 probability: {total_prob:.4f}")
        return total_prob
    except Exception as e:
        print(f"Error in Over 2.5 calculation: {str(e)}")
        return 0.5  # Return default probability on error

def calculate_btts_probability(home_xg, away_xg):
    """Calculate probability of both teams scoring using Poisson distribution"""
    try:
        from scipy.stats import poisson
        import numpy as np
        
        print(f"Calculating BTTS probability with xG: home={home_xg}, away={away_xg}")
        
        # Convert xG to float and handle None/invalid values
        try:
            home_xg = float(home_xg) if home_xg not in (None, 0) else 0.5
            away_xg = float(away_xg) if away_xg not in (None, 0) else 0.5
        except (ValueError, TypeError):
            print(f"Error converting xG values: {str(e)}")
            home_xg = 0.5
            away_xg = 0.5
        
        if home_xg <= 0 or away_xg <= 0:
            print("Invalid xG values (<=0)")
            return 0.5
        
        # Probability of home team scoring at least 1
        home_scoring_prob = 1 - poisson.pmf(0, home_xg)
        print(f"Home team scoring prob: {home_scoring_prob:.4f}")
        
        # Probability of away team scoring at least 1
        away_scoring_prob = 1 - poisson.pmf(0, away_xg)
        print(f"Away team scoring prob: {away_scoring_prob:.4f}")
        
        # Probability of both teams scoring = P(Home scores) * P(Away scores)
        btts_prob = home_scoring_prob * away_scoring_prob
        print(f"Calculated BTTS probability: {btts_prob:.4f}")
        
        return btts_prob
    except Exception as e:
        print(f"Error in BTTS calculation: {str(e)}")
        return 0.5  # Return default probability on error

def display_odds_box(title, odds, implied_prob, ev):
    """Helper function to display odds box with consistent styling"""
    if ev is None:
        ev = 0
        
    ev_color = get_ev_color(ev)
    
    st.markdown(f"""
        <div style="background-color: {ev_color}; padding: 0.5rem; border-radius: 6px; margin: 0.25rem 0; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);">
            <h4 style="margin: 0; color: #1a1a1a; font-size: 0.9rem; font-weight: 600;">{title}</h4>
            <div style="display: flex; justify-content: space-between; margin-top: 0.25rem;">
                <div>
                    <p style="margin: 0; color: #4a5568; font-size: 0.8rem; line-height: 1.2;">Odds</p>
                    <p style="margin: 0; color: #1a1a1a; font-weight: 600; font-size: 0.9rem; line-height: 1.2;">{odds:.2f}</p>
                </div>
                <div>
                    <p style="margin: 0; color: #4a5568; font-size: 0.8rem; line-height: 1.2;">EV</p>
                    <p style="margin: 0; color: #1a1a1a; font-weight: 600; font-size: 0.9rem; line-height: 1.2;">{ev:+.1f}%</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def get_ev_color(ev_percentage):
    """
    Get background color based on EV percentage
    
    Args:
        ev_percentage (float): Expected Value percentage
        
    Returns:
        str: Hex color code
    """
    try:
        if ev_percentage > 25:
            return "#00AF50"  # Dark Green - PLUS EV over 25%
        elif ev_percentage > 15:
            return "#A9D08F"  # Light Green - PLUS EV 15-25%
        elif ev_percentage > 5:
            return "#E2EFDB"  # Very Light Green - PLUS EV 5-15%
        elif ev_percentage >= -5:
            return "#FFFF00"  # Yellow - Breakeven -5% to 5%
        elif ev_percentage >= -15:
            return "#F4AF84"  # Light Red - MINUS EV -5% to -15%
        else:
            return "#FE0000"  # Dark Red - MINUS EV below -15%
    except Exception as e:
        logger.error(f"Error getting EV color: {str(e)}")
        return "#FFFFFF"  # White as fallback

def calculate_ev(predicted_prob, odds):
    """
    Calculate Expected Value (EV) for a bet using the formula: (Bookie Odds/Breakeven Odds - 1) * 100
    where Breakeven Odds = 1/Probability
    
    Args:
        predicted_prob (float): Our predicted probability (0-100)
        odds (float): Decimal odds from bookmaker
        
    Returns:
        float: Expected Value percentage
    """
    try:
        # Convert probability to decimal (0-1)
        prob_decimal = predicted_prob / 100
        
        # Calculate breakeven odds
        breakeven_odds = 1 / prob_decimal
        
        # Calculate EV percentage
        ev_percentage = (odds / breakeven_odds - 1) * 100
        
        return round(ev_percentage, 2)
        
    except ZeroDivisionError:
        logger.error("Division by zero in EV calculation - probability cannot be 0")
        return 0.0
    except Exception as e:
        logger.error(f"Error calculating EV: {str(e)}")
        return 0.0

def get_team_logo_path(team_name):
    """Get the logo path for a team from teams_data.json"""
    try:
        # Use relative path for teams_data.json
        teams_data_path = 'teams_data.json'
        if not os.path.exists(teams_data_path):
            logger.error(f"teams_data.json not found at {teams_data_path}")
            return None
            
        with open(teams_data_path, 'r') as f:
            teams_data = json.load(f)
        
        original_name = team_name
        
        # Handle common variations in team names
        variations = []
        
        # Original name
        variations.append(team_name)
        
        # Remove common suffixes
        team_name = team_name.replace(' FC', '').replace(' CF', '')
        team_name = team_name.replace(' SAD', '').replace(' CD', '')
        team_name = team_name.replace(' SC', '').replace(' SK', '')
        team_name = team_name.replace(' FK', '').replace(' JK', '')
        team_name = team_name.strip()
        variations.append(team_name)
        
        # Try without accents
        unaccented = unidecode_text(team_name)
        if unaccented != team_name:
            variations.append(unaccented)
            
        # Try with/without UD prefix
        if team_name.startswith('UD '):
            variations.append(team_name[3:])
        else:
            variations.append('UD ' + team_name)
            
        # Handle Turkish team variations
        turkish_teams = {
            'Gaziantep': [
                'Gaziantep',
                'Gaziantep FK',
                'Gazişehir Gaziantep',
                'Gazisehir Gaziantep',
                'Gaziantep Football Club'
            ],
            'Galatasaray': [
                'Galatasaray',
                'Galatasaray SK',
                'Galatasaray Istanbul'
            ],
            'Fenerbahce': [
                'Fenerbahce',
                'Fenerbahçe',
                'Fenerbahce SK',
                'Fenerbahce Istanbul'
            ],
            'Besiktas': [
                'Besiktas',
                'Beşiktaş',
                'Besiktas JK',
                'Besiktas Istanbul'
            ],
            'Trabzonspor': [
                'Trabzonspor',
                'Trabzon'
            ]
        }
        
        # Check if team name contains any Turkish team variation
        for base_team, team_variations in turkish_teams.items():
            if any(variant.lower() in team_name.lower() for variant in team_variations):
                variations.extend(team_variations)
                break
            
        # Try each variation
        for variant in variations:
            if variant in teams_data:
                # Convert absolute path to relative path
                logo_path = teams_data[variant]['logo_path']
                relative_path = os.path.join('team_logos', os.path.basename(logo_path))
                
                if os.path.exists(relative_path):
                    logger.info(f"Found logo for {original_name} using variant: {variant}")
                    return relative_path
                else:
                    logger.warning(f"Logo file not found at {relative_path} for {variant}")
                    
        logger.error(f"No logo found for {original_name}. Tried variations: {variations}")
        return None
            
    except Exception as e:
        logger.error(f"Error getting logo path for {team_name}: {str(e)}")
        return None

def show_main_app():
    # Update results automatically
    update_match_results()
    
    # Add navigation buttons and back to top button
    add_navigation_buttons()
    add_back_to_top_button()
    
    st.markdown('<div id="top"></div>', unsafe_allow_html=True)
    
    germany_tz = ZoneInfo("Europe/Berlin")

    # Get the current time in Germany's timezone
    now = datetime.now(germany_tz)

    # Create two columns for date inputs
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=now.date(),
            min_value=now.date(),
            max_value=now.date() + timedelta(days=21),
            help="Select start date to view matches (up to 3 weeks ahead)"
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=now.date(),
            min_value=now.date(),
            max_value=now.date() + timedelta(days=21),
            help="Select end date to view matches (up to 3 weeks ahead)"
        )
    
    if start_date and end_date:
        if start_date > end_date:
            st.error("Start date must be before or equal to end date")
            return
            
        # Get matches for the date range
        matches = []
        current_date = start_date
        while current_date <= end_date:
            daily_matches = get_matches_for_date(current_date)
            matches.extend(daily_matches)
            current_date += timedelta(days=1)

        # Filter out matches that have already been played in German time
        current_timestamp = now.timestamp()
        matches = [match for match in matches if match.get('date_unix', 0) > current_timestamp]
            
        # Sort matches by kickoff time
        matches.sort(key=lambda x: x.get('date_unix', 0))
        
        # Get unique leagues from matches
        available_leagues = {"All Matches": None}
        for match in matches:
            league_name = get_league_name(match)
            if league_name and league_name != "Unknown League":
                available_leagues[league_name] = match.get('competition_id')
        
        # League filter
        selected_leagues = st.multiselect(
            "Select Competitions",
            options=list(available_leagues.keys()),
            default=["All Matches"],
            key="selected_leagues",
            help="Filter matches by competitions (select multiple)"
        )
        
        # Confidence filter
        confidence_levels = st.multiselect(
            "Filter by Confidence Levels",
            options=["All", "High", "Medium", "Low"],
            default=["All"],
            key="confidence_levels",
            help="Filter predictions by confidence levels (High: ≥70%, Medium: 50-69%, Low: <50%)"
        )
        
        # Savable Filters UI
        import importlib
        filter_storage = importlib.import_module('filter_storage')

        if 'saved_filters' not in st.session_state:
            st.session_state.saved_filters = filter_storage.load_saved_filters()

        st.markdown('<div class="filter-preset-section"><h4>Save & Load Filter Presets</h4></div>', unsafe_allow_html=True)
        filter_name = st.text_input("Name your filter preset", key="main_filter_name", placeholder="e.g. Weekend Favs")
        save_col, apply_col, delete_col = st.columns([2, 1, 1])
        save_btn = save_col.button("Save Filter", key="save_main_filter", help="Save the current filter selections as a preset")
        if save_btn:
            if filter_name:
                st.session_state.saved_filters = filter_storage.save_filter(
                    filter_name,
                    selected_leagues,
                    confidence_levels
                )
                st.success(f"Saved filter preset '{filter_name}'!")
            else:
                st.error("Please enter a filter name.")
        if st.session_state.saved_filters:
            st.markdown("<div class='saved-filters-list'><b>Presets:</b></div>", unsafe_allow_html=True)
            for idx, sf in enumerate(st.session_state.saved_filters):
                st.markdown(f"<div class='filter-preset'><b>{sf['name']}</b> | Leagues: <span class='filter-leagues'>{', '.join(sf['leagues'])}</span> | Confidence: <span class='filter-confidence'>{', '.join(sf['confidence'])}</span></div>", unsafe_allow_html=True)
                preset_cols = st.columns([1, 1])
                apply_btn = preset_cols[0].button("Apply", key=f"apply_main_filter_{idx}")
                delete_btn = preset_cols[1].button("Delete", key=f"delete_main_filter_{idx}")
                if apply_btn:
                    st.session_state.pending_filter = sf
                    st.rerun()
                if delete_btn:
                    st.session_state.saved_filters = filter_storage.delete_filter(sf['id'])
                    st.rerun()
        # Minimal CSS for filter preset UI and smaller buttons
        st.markdown('''
        <style>
        .filter-preset-section h4 {
            margin-bottom: 0.3rem;
            color: #2c5282;
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: 0.3px;
        }
        .saved-filters-list {
            margin-top: 0.3rem;
            margin-bottom: 0.1rem;
            color: #2c5282;
            font-size: 0.97rem;
        }
        .filter-preset {
            background: #f7fafc;
            border-radius: 7px;
            padding: 0.4rem 0.7rem;
            margin-bottom: 0.18rem;
            font-size: 0.95rem;
            box-shadow: 0 1px 2px rgba(44,82,130,0.03);
        }
        .filter-leagues, .filter-confidence {
            color: #3182ce;
            font-weight: 600;
        }
        /* Make Streamlit buttons smaller in height */
        .stButton > button {
            padding-top: 0.42rem !important;
            padding-bottom: 0.42rem !important;
            font-size: 0.97rem !important;
            min-height: 2.1rem !important;
            height: 2.2rem !important;
            line-height: 2.2rem !important;  /* Match line-height to height for perfect vertical centering */
        }
        </style>
        ''', unsafe_allow_html=True)

        # After building available_leagues, but BEFORE rendering widgets:
        if "pending_filter" in st.session_state:
            sf = st.session_state.pending_filter
            valid_leagues = [l for l in sf['leagues'] if l in available_leagues.keys()]
            if not valid_leagues:
                valid_leagues = ["All Matches"]
            # Clear previous session state for these keys
            st.session_state.pop("selected_leagues", None)
            st.session_state.pop("confidence_levels", None)
            st.session_state.selected_leagues = valid_leagues
            st.session_state.confidence_levels = sf['confidence']
            del st.session_state.pending_filter
            st.rerun()
        # ---------------------------------------------------------------
        
        # Filter matches by selected leagues
        if "All Matches" not in selected_leagues:
            matches = [m for m in matches if get_league_name(m) in selected_leagues]
        
        if not matches:
            st.info(f"No matches found for selected competitions between {start_date} and {end_date}.")
            return
        
        # Group matches by league for better organization
        matches_by_league = {}
        for match in matches:
            league_name = get_league_name(match)
            if league_name not in matches_by_league:
                matches_by_league[league_name] = []
            matches_by_league[league_name].append(match)
        
        # Display matches grouped by league
        for league_name, league_matches in matches_by_league.items():
            matches_to_display = []
            
            # First pass: process all matches and store those that meet confidence criteria
            for match in league_matches:
                result = process_match_prediction(match)
                if result:
                    prediction_data, confidence = result
                    show_match = False
                    
                    if "All" in confidence_levels:
                        show_match = True
                    else:
                        # Check if match confidence matches any selected confidence level
                        if "High" in confidence_levels and confidence >= 70:  
                            show_match = True
                        elif "Medium" in confidence_levels and 50 <= confidence < 70:  
                            show_match = True
                        elif "Low" in confidence_levels and confidence < 50:  
                            show_match = True
                    
                    if show_match:
                        matches_to_display.append((match, prediction_data, confidence))
            
            # Only show league header if there are matches to display
            if matches_to_display:
                st.markdown(f"## {league_name}")
                
                # Second pass: display filtered matches
                for match, prediction_data, confidence in matches_to_display:
                    display_kickoff_time(match)
                    display_match_details(match, prediction_data, confidence)
            else:
                st.info(f"No matches with selected confidence levels found in {league_name}.")
                
    # Add Navigation JavaScript
    st.markdown("""
<script>
    function handleLogout() {
        // Clear session state
        localStorage.clear();
        sessionStorage.clear();
        
        // Redirect to home page
        window.location.href = '/';
    }

    function navigateToHome() {
        window.location.href = '/';
    }

    function navigateToHistory() {
        window.location.href = '/?page=history';
    }
</script>
""", unsafe_allow_html=True)

# Add Navigation Buttons
def add_navigation_buttons():
    col1, col2, col3 = st.columns([2,2,2])
    
    with col1:
        if st.button("Trends History", key="graph"):
            st.query_params["page"] = "graph"
            st.rerun()
            
    with col2:
        if st.button("History", key="history"):
            st.query_params["page"] = "history"
            st.rerun()
            
    with col3:
        if st.button("Logout", key="logout"):
            st.session_state.logged_in = False
            st.query_params.clear()
            st.rerun()

def add_back_to_top_button():
    """Add a back to top button with arrow icon"""
    st.markdown("""
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <a href="#top" class="back-to-top" title="Back to Top">
            <i class="fas fa-arrow-up"></i>
        </a>
        <script>
            // Show/hide back to top button based on scroll position
            window.onscroll = function() {
                var button = document.querySelector('.back-to-top');
                if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
                    button.style.display = 'flex';
                } else {
                    button.style.display = 'none';
                }
            };
        </script>
    """, unsafe_allow_html=True)

def convert_to_cet(kickoff):
    """Convert kickoff time to CET format"""
    try:
        # Parse the kickoff time (assuming it's in HH:MM format)
        time_obj = datetime.strptime(kickoff, '%H:%M').time()
        return time_obj.strftime('%H:%M CET')
    except:
        return kickoff  # Return original string if conversion fails

def auto_predict_matches():
    """Automatically predict matches without UI interaction"""
    try:
        matches = get_matches_for_date(datetime.now())
        for match in matches:
            process_match_prediction(match)
        return True
    except Exception as e:
        logger.error(f"Error in auto prediction: {str(e)}")
        return False

from graph_page import render_graph_page

def main():
    # Auto-predict endpoint for GitHub Actions
    params = dict(st.query_params)
    if 'auto_predict' in params and params['auto_predict'][0] == 'true':
        success = auto_predict_matches()
        st.json({
            'status': 'success' if success else 'error',
            'timestamp': datetime.now().isoformat(),
            'message': 'Auto-prediction completed' if success else 'Auto-prediction failed'
        })
        return

    # Health check endpoint
    if 'health-check' in st.query_params:
        st.json({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'app': 'fottyy'
        })
        return

    # Initialize session state
    init_session_state()
    
    # Get current page from query parameters
    page = st.query_params.get("page", "main")
    
    if not st.session_state.logged_in and page != "login":
        st.query_params["page"] = "login"
        st.rerun()
    
    # Route to appropriate page
    if page == "login":
        show_login_page()
    elif page == "history":
        show_history_page()
    elif page == "graph":
        render_graph_page()
    else:
        show_main_app()

if __name__ == "__main__":
    main()
