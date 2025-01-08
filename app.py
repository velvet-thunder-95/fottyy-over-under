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
</style>
""", unsafe_allow_html=True)

# Hardcoded credentials
VALID_USERNAME = "matchday_wizard"
VALID_PASSWORD = "GoalMaster"

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'login_submitted' not in st.session_state:
    st.session_state.login_submitted = False

def login(username, password):
    if username == VALID_USERNAME and password == VALID_PASSWORD:
        st.session_state.logged_in = True
        return True
    return False

def logout():
    st.session_state.logged_in = False

def show_login_page():
    st.markdown('<h1 class="app-title">⚽ Football Match Predictor</h1>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
            <div class="login-container">
                <h2 style="color: #1a1a1a; text-align: center; margin-bottom: 2rem;">Welcome Back!</h2>
            </div>
        """, unsafe_allow_html=True)
        
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

def create_match_features_from_api(match_data):
    """Create features DataFrame from match data with error handling"""
    try:
        features = {}

        # First add features in the exact order they were used during training
        feature_order = [
            'season', 'competition_id',
            'home_total_matches', 'away_total_matches',
            'home_win_rate', 'away_win_rate',
            'home_form_points', 'away_form_points',
            'home_shots', 'away_shots',
            'home_shots_on_target', 'away_shots_on_target',
            'home_corners', 'away_corners',
            'home_fouls', 'away_fouls',
            'home_possession', 'away_possession',
            'home_xg', 'away_xg',
            'shot_accuracy_home', 'shot_accuracy_away',
            'home_win_rate_ratio', 'home_momentum', 'away_momentum',
            'odds_home_win', 'odds_draw', 'odds_away_win',
            'implied_home_prob', 'implied_draw_prob', 'implied_away_prob'
        ]

        # Basic features
        features['season'] = int(match_data['season'].split('/')[0])
        features['competition_id'] = match_data['competition_id']
        
        # Match counts and form using completed matches
        features['home_total_matches'] = max(1, match_data.get('matches_completed_minimum', 1))
        features['away_total_matches'] = max(1, match_data.get('matches_completed_minimum', 1))
        
        # Win rates - use pre-match PPG and adjust based on home/away performance
        features['home_win_rate'] = match_data.get('pre_match_teamA_overall_ppg', 0) / 3.0
        features['away_win_rate'] = match_data.get('pre_match_teamB_overall_ppg', 0) / 3.0
        
        # Adjust win rates based on home/away specific PPG
        home_ppg_ratio = match_data.get('pre_match_home_ppg', 0) / max(0.1, match_data.get('pre_match_teamA_overall_ppg', 1))
        away_ppg_ratio = match_data.get('pre_match_away_ppg', 0) / max(0.1, match_data.get('pre_match_teamB_overall_ppg', 1))
        features['home_win_rate'] *= max(0.5, min(1.5, home_ppg_ratio))
        features['away_win_rate'] *= max(0.5, min(1.5, away_ppg_ratio))
        
        # Form points with home/away adjustment
        features['home_form_points'] = match_data.get('pre_match_home_ppg', features['home_win_rate'] * 3)
        features['away_form_points'] = match_data.get('pre_match_away_ppg', features['away_win_rate'] * 3)
        
        # Use potential metrics to estimate match stats
        goal_potential = match_data.get('o25_potential', 50) / 100.0
        corner_potential = match_data.get('corners_potential', 10) / 10.0
        
        # Shot statistics based on xG and potential metrics
        avg_shots_ratio = 12  # Average shots per expected goal
        home_xg = match_data.get('team_a_xg_prematch', 1.5)
        away_xg = match_data.get('team_b_xg_prematch', 1.5)
        
        features['home_shots'] = int(home_xg * avg_shots_ratio * goal_potential)
        features['away_shots'] = int(away_xg * avg_shots_ratio * goal_potential)
        features['home_shots_on_target'] = int(features['home_shots'] * 0.4)  # Typical shot accuracy
        features['away_shots_on_target'] = int(features['away_shots'] * 0.4)
        
        # Corner predictions using corner odds and potential
        corner_odds_home = float(match_data.get('odds_corners_1', 0))
        corner_odds_away = float(match_data.get('odds_corners_2', 0))
        
        # Default corner predictions if odds not available
        if corner_odds_home == 0 or corner_odds_away == 0:
            features['home_corners'] = int(corner_potential * 5)  # Default distribution
            features['away_corners'] = int(corner_potential * 5)
        else:
            corner_odds_ratio = corner_odds_home / corner_odds_away
            features['home_corners'] = int(corner_potential * (5 + corner_odds_ratio))
            features['away_corners'] = int(corner_potential * (5 + 1/corner_odds_ratio))
        
        # Fouls based on cards potential
        cards_potential = match_data.get('cards_potential', 3.75)
        features['home_fouls'] = int(10 * cards_potential / 3.75)
        features['away_fouls'] = int(10 * cards_potential / 3.75)
        
        # Possession based on team strength and BTTS potential
        btts_potential = match_data.get('btts_potential', 50) / 100.0
        strength_ratio = features['home_win_rate'] / max(0.1, features['away_win_rate'])
        features['home_possession'] = 50 * (1 + 0.2 * (strength_ratio - 1) * btts_potential)
        features['away_possession'] = 100 - features['home_possession']
        
        # Expected goals - use prematch xG and adjust by potential
        features['home_xg'] = match_data['team_a_xg_prematch'] * goal_potential
        features['away_xg'] = match_data['team_b_xg_prematch'] * goal_potential
        
        # Shot accuracy with better handling of edge cases
        features['shot_accuracy_home'] = features['home_shots_on_target'] / max(1, features['home_shots'])
        features['shot_accuracy_away'] = features['away_shots_on_target'] / max(1, features['away_shots'])
        
        # Advanced metrics using potential data
        o15_potential = match_data.get('o15_potential', 70) / 100.0
        o35_potential = match_data.get('o35_potential', 40) / 100.0
        
        # Calculate momentum features with potential adjustments
        features['home_win_rate_ratio'] = features['home_win_rate'] / max(0.001, features['away_win_rate'])
        features['home_momentum'] = features['home_form_points'] * features['home_win_rate'] * o15_potential
        features['away_momentum'] = features['away_form_points'] * features['away_win_rate'] * o15_potential
        
        # Odds and probabilities with margin adjustment
        features['odds_home_win'] = match_data.get('odds_ft_1', 3.0)
        features['odds_draw'] = match_data.get('odds_ft_x', 3.0)
        features['odds_away_win'] = match_data.get('odds_ft_2', 3.0)
        
        # Calculate implied probabilities with dynamic margin based on league level
        margin = 1.07 - (0.02 * (match_data.get('competition_id', 0) % 3))  # Adjust margin by competition level
        raw_home_prob = 1 / features['odds_home_win']
        raw_draw_prob = 1 / features['odds_draw']
        raw_away_prob = 1 / features['odds_away_win']
        total_prob = raw_home_prob + raw_draw_prob + raw_away_prob
        
        # Adjust probabilities using BTTS and over/under potential
        btts_factor = btts_potential * o35_potential
        features['implied_home_prob'] = (raw_home_prob / (total_prob * margin)) * (1 + 0.1 * btts_factor)
        features['implied_draw_prob'] = (raw_draw_prob / (total_prob * margin)) * (1 - 0.1 * btts_factor)
        features['implied_away_prob'] = (raw_away_prob / (total_prob * margin)) * (1 + 0.1 * btts_factor)
        
        # Normalize probabilities
        prob_sum = features['implied_home_prob'] + features['implied_draw_prob'] + features['implied_away_prob']
        features['implied_home_prob'] /= prob_sum
        features['implied_draw_prob'] /= prob_sum
        features['implied_away_prob'] /= prob_sum

        # Add derived features in the same order as training
        features['form_difference'] = features['home_form_points'] - features['away_form_points']
        features['win_rate_difference'] = features['home_win_rate'] - features['away_win_rate']
        features['shot_difference'] = features['home_shots'] - features['away_shots']
        features['possession_difference'] = features['home_possession'] - features['away_possession']
        features['xg_difference'] = features['home_xg'] - features['away_xg']
        features['total_momentum'] = features['home_momentum'] + features['away_momentum']
        features['momentum_difference'] = features['home_momentum'] - features['away_momentum']
        features['odds_ratio'] = features['odds_home_win'] / features['odds_away_win']
        features['implied_prob_sum'] = prob_sum

        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Ensure all values are numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill any NaN values with 0
        df = df.fillna(0)
        
        return df
        
    except Exception as e:
        logger.error(f"Error creating features: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to create features: {str(e)}")

def adjust_probabilities(home_prob, draw_prob, away_prob, match_data):
    """Adjust probabilities based on odds and team strengths"""
    # Get odds
    home_odds = float(match_data.get('odds_ft_1', 2.0))
    away_odds = float(match_data.get('odds_ft_2', 2.0))
    draw_odds = float(match_data.get('odds_ft_x', 3.0))
    
    # Convert odds to probabilities
    odds_home_prob = 1 / home_odds
    odds_away_prob = 1 / away_odds
    odds_draw_prob = 1 / draw_odds
    
    # Normalize odds probabilities
    total_odds_prob = odds_home_prob + odds_away_prob + odds_draw_prob
    odds_home_prob /= total_odds_prob
    odds_away_prob /= total_odds_prob
    odds_draw_prob /= total_odds_prob
    
    # Get team strengths
    home_ppg = float(match_data.get('home_ppg', 0))
    away_ppg = float(match_data.get('away_ppg', 0))
    home_overall_ppg = float(match_data.get('pre_match_teamA_overall_ppg', 0))
    away_overall_ppg = float(match_data.get('pre_match_teamB_overall_ppg', 0))
    
    # Calculate form-based probabilities
    total_ppg = home_overall_ppg + away_overall_ppg
    if total_ppg > 0:
        form_home_prob = home_overall_ppg / total_ppg
        form_away_prob = away_overall_ppg / total_ppg
    else:
        form_home_prob = 0.4
        form_away_prob = 0.4
    form_draw_prob = 1 - (form_home_prob + form_away_prob)
    
    # Weights for different factors
    model_weight = 0.4
    odds_weight = 0.4
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
    total = final_home_prob + final_away_prob + final_draw_prob
    final_home_prob /= total
    final_away_prob /= total
    final_draw_prob /= total
    
    # Apply minimum probability thresholds
    min_prob = 0.1
    if final_away_prob < min_prob:
        final_away_prob = min_prob
        excess = (1 - min_prob) / 2
        final_home_prob = excess
        final_draw_prob = excess
    
    return final_home_prob, final_draw_prob, final_away_prob

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
    else:
        home_prob_odds = 0.33
        draw_prob_odds = 0.33
        away_prob_odds = 0.33
    
    # Calculate form-based probabilities
    total_ppg = home_ppg + away_ppg
    if total_ppg > 0:
        home_prob_form = home_ppg / total_ppg
        away_prob_form = away_ppg / total_ppg
    else:
        home_prob_form = 0.5
        away_prob_form = 0.5
    
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
    
    # Calculate draw probability (based on odds and typical draw frequency)
    draw_final = draw_prob_odds * 0.6  # Reduce draw probability slightly as they're less common
    
    # Normalize probabilities to sum to 1
    total = home_final + away_final + draw_final
    home_final /= total
    away_final /= total
    draw_final /= total
    
    return home_final, draw_final, away_final

def display_prediction(prediction, confidence):
    """Display prediction with appropriate color based on confidence level"""
    if confidence >= 0.6:
        sentiment_class = "prediction-high"
        confidence_text = "High Confidence"
    elif confidence >= 0.4:
        sentiment_class = "prediction-medium"
        confidence_text = "Medium Confidence"
    else:
        sentiment_class = "prediction-low"
        confidence_text = "Low Confidence"
        
    st.markdown(f"""
        <div class="{sentiment_class}">
            {prediction}
            <div style="font-size: 0.9rem; margin-top: 0.5rem;">
                {confidence_text}
            </div>
        </div>
    """, unsafe_allow_html=True)

def display_probability_bars(home_prob, draw_prob, away_prob, home_team, away_team):
    """Display probability bars for match outcomes"""
    st.markdown("### Match Outcome Probabilities")
    
    # Create a container with white background
    with st.container():
        st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); margin: 1rem 0;">
            </div>
        """, unsafe_allow_html=True)
        
        # Create three columns for the probabilities
        col1, col2, col3 = st.columns(3)
        
        # Home team probability
        with col1:
            st.markdown(f"""
                <div style="text-align: center;">
                    <div style="margin-bottom: 0.5rem;">
                        <span style="color: #48bb78; font-weight: 600; font-size: 1.2rem;">{home_prob:.1%}</span>
                    </div>
                    <div style="color: #1a1a1a; font-weight: 500;">{home_team}</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Draw probability
        with col2:
            st.markdown(f"""
                <div style="text-align: center;">
                    <div style="margin-bottom: 0.5rem;">
                        <span style="color: #ed8936; font-weight: 600; font-size: 1.2rem;">{draw_prob:.1%}</span>
                    </div>
                    <div style="color: #1a1a1a; font-weight: 500;">Draw</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Away team probability
        with col3:
            st.markdown(f"""
                <div style="text-align: center;">
                    <div style="margin-bottom: 0.5rem;">
                        <span style="color: #3182ce; font-weight: 600; font-size: 1.2rem;">{away_prob:.1%}</span>
                    </div>
                    <div style="color: #1a1a1a; font-weight: 500;">{away_team}</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Add the combined progress bar
        st.markdown(f"""
            <div style="margin-top: 1rem; padding: 0 1rem;">
                <div style="width: 100%; height: 20px; background: #e2e8f0; border-radius: 10px; overflow: hidden; display: flex;">
                    <div style="width: {home_prob * 100}%; height: 100%; background-color: #48bb78;"></div>
                    <div style="width: {draw_prob * 100}%; height: 100%; background-color: #ed8936;"></div>
                    <div style="width: {away_prob * 100}%; height: 100%; background-color: #3182ce;"></div>
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

def display_match_odds(match_data):
    """Display FootyStats match odds in an organized box."""
    # Display match stats in columns using Streamlit's column layout
    st.markdown("""
        <h3 style="text-align: center; color: #1f2937; margin: 20px 0; font-size: 1.5rem;">Match Stats & Odds</h3>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h4 style="color: #1f2937; font-size: 1.2rem; font-weight: 600; margin-bottom: 15px; text-align: center;">Match Result</h4>
            </div>
        """, unsafe_allow_html=True)
        
        # Match Result odds
        home_win = match_data.get('odds_ft_1', 'N/A')
        draw = match_data.get('odds_ft_x', 'N/A')
        away_win = match_data.get('odds_ft_2', 'N/A')
        
        st.markdown(f"""
            <div style="display: flex; flex-direction: column; gap: 10px; margin-top: 10px;">
                <div style="background-color: #f8fafc; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <div style="color: #1f2937; font-weight: 500;">Home Win</div>
                    <div style="color: #2563eb; font-size: 1.2rem; font-weight: 600;">{home_win}</div>
                </div>
                <div style="background-color: #f8fafc; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <div style="color: #1f2937; font-weight: 500;">Draw</div>
                    <div style="color: #2563eb; font-size: 1.2rem; font-weight: 600;">{draw}</div>
                </div>
                <div style="background-color: #f8fafc; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <div style="color: #1f2937; font-weight: 500;">Away Win</div>
                    <div style="color: #2563eb; font-size: 1.2rem; font-weight: 600;">{away_win}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h4 style="color: #1f2937; font-size: 1.2rem; font-weight: 600; margin-bottom: 15px; text-align: center;">Goals Markets</h4>
            </div>
        """, unsafe_allow_html=True)
        
        # Goals Markets odds
        over = match_data.get('odds_ft_over25', 'N/A')
        under = match_data.get('odds_ft_under25', 'N/A')
        btts = match_data.get('odds_btts_yes', 'N/A')
        
        st.markdown(f"""
            <div style="display: flex; flex-direction: column; gap: 10px; margin-top: 10px;">
                <div style="background-color: #f8fafc; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <div style="color: #1f2937; font-weight: 500;">Over 2.5</div>
                    <div style="color: #2563eb; font-size: 1.2rem; font-weight: 600;">{over}</div>
                </div>
                <div style="background-color: #f8fafc; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <div style="color: #1f2937; font-weight: 500;">Under 2.5</div>
                    <div style="color: #2563eb; font-size: 1.2rem; font-weight: 600;">{under}</div>
                </div>
                <div style="background-color: #f8fafc; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <div style="color: #1f2937; font-weight: 500;">BTTS</div>
                    <div style="color: #2563eb; font-size: 1.2rem; font-weight: 600;">{btts}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

def get_match_prediction(match_data):
    """Calculate match prediction using the loaded model"""
    try:
        logger.info("Starting prediction calculation")
        logger.info(f"Match data: {match_data}")
        
        if predictor is None:
            raise ValueError("Model not loaded properly")
            
        # Create features DataFrame
        features_df = create_match_features_from_api(match_data)
        
        # Convert DataFrame to DMatrix
        dmatrix = xgb.DMatrix(features_df)
        
        # Make prediction using DMatrix
        probabilities = predictor.predict(dmatrix)
        
        # If the model outputs a single probability, convert it to three probabilities
        if len(probabilities.shape) == 1:
            home_prob = probabilities[0]
            odds_total = (1/match_data.get('odds_ft_1', 3.0) + 
                         1/match_data.get('odds_ft_x', 3.0) + 
                         1/match_data.get('odds_ft_2', 3.0))
            draw_prob = (1/match_data.get('odds_ft_x', 3.0)) / odds_total
            away_prob = (1/match_data.get('odds_ft_2', 3.0)) / odds_total
        else:
            home_prob = probabilities[0][0]
            draw_prob = probabilities[0][1]
            away_prob = probabilities[0][2]
        
        # Adjust probabilities based on odds and team strengths
        home_prob, draw_prob, away_prob = adjust_probabilities(
            home_prob, draw_prob, away_prob, match_data
        )
        
        return home_prob, draw_prob, away_prob
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        st.error(f"Prediction error: {str(e)}")
        return None, None, None

def normalize_probabilities(probs):
    """Normalize probabilities to sum to 1"""
    total = sum(probs)
    if total == 0:
        return [1/3, 1/3, 1/3]  # Equal probabilities if sum is 0
    return [p/total for p in probs]

def update_match_results():
    """Update completed match results and calculate profits/losses"""
    history = PredictionHistory()
    
    # Get pending predictions
    pending_predictions = history.get_predictions(status='Pending')
    
    for _, pred in pending_predictions.iterrows():
        try:
            # Find the match
            match = get_match_by_teams(
                pred['home_team'],
                pred['away_team'],
                pred['date']
            )
            
            if match:
                # Get the result
                match_data = get_match_result(match['id'])
                
                if match_data and match_data['status'] == 'FINISHED':
                    # Determine winner
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
                            profit = pred['home_odds'] - 1
                        elif actual_outcome == "AWAY":
                            profit = pred['away_odds'] - 1
                        else:
                            profit = pred['draw_odds'] - 1
                    else:
                        profit = -1
                    
                    # Update prediction result
                    history.update_prediction_result(
                        pred['id'],
                        actual_outcome,
                        profit
                    )
                    
        except Exception as e:
            logger.error(f"Error updating match result: {str(e)}")
            continue

def process_match_prediction(match):
    """Process and save prediction for a match"""
    try:
        # Get predictions
        home_prob, draw_prob, away_prob = get_match_prediction(match)
        
        if all(prob is not None for prob in [home_prob, draw_prob, away_prob]):
            # Normalize probabilities
            probs = normalize_probabilities([home_prob, draw_prob, away_prob])
            home_prob, draw_prob, away_prob = probs
            
            # Determine predicted outcome
            max_prob = max(home_prob, draw_prob, away_prob)
            if max_prob == home_prob:
                predicted_outcome = "HOME"
            elif max_prob == away_prob:
                predicted_outcome = "AWAY"
            else:
                predicted_outcome = "DRAW"
            
            # Create prediction data
            prediction_data = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'league': get_league_name(match),
                'home_team': match.get('home_name'),
                'away_team': match.get('away_name'),
                'predicted_outcome': predicted_outcome,
                'home_odds': float(match.get('odds_ft_1', 0)),
                'draw_odds': float(match.get('odds_ft_x', 0)),
                'away_odds': float(match.get('odds_ft_2', 0)),
                'confidence': max_prob,
                'bet_amount': 1.0,  # Fixed bet amount
                'prediction_type': 'Match Result',
                'match_date': match.get('date', datetime.now().strftime('%Y-%m-%d')),
                'match_id': str(match.get('id', ''))
            }
            
            # Check for existing prediction
            history = PredictionHistory()
            existing_predictions = history.get_predictions(
                start_date=prediction_data['match_date'],
                end_date=prediction_data['match_date']
            )
            
            # Check if prediction already exists for this match
            match_exists = False
            if not existing_predictions.empty:
                match_exists = existing_predictions[
                    (existing_predictions['home_team'] == prediction_data['home_team']) &
                    (existing_predictions['away_team'] == prediction_data['away_team']) &
                    (existing_predictions['match_date'] == prediction_data['match_date'])
                ].shape[0] > 0
            
            # Only add if prediction doesn't exist
            if not match_exists:
                history.add_prediction(prediction_data)
            
            # Display prediction information
            display_probability_bars(
                home_prob, 
                draw_prob, 
                away_prob, 
                match.get('home_name', ''), 
                match.get('away_name', '')
            )
            
            # Display match odds
            display_match_odds(match)
            
            # Display prediction text
            prediction = f"Prediction: {predicted_outcome}"
            display_prediction(prediction, max_prob)
            
            return True
    except Exception as e:
        logger.error(f"Error processing prediction for match: {str(e)}", exc_info=True)
        return False
    
    return False

def show_main_app():
    # Update results automatically
    update_match_results()
    
    # Add navigation buttons
    add_navigation_buttons()
    
    st.markdown('<div id="top"></div>', unsafe_allow_html=True)
    st.markdown("<h1>⚽ Football Match Predictor ⚽</h1>", unsafe_allow_html=True)
    
    # Date selector
    date = st.date_input(
        "Select Date",
        value=datetime.now().date(),
        min_value=datetime.now().date() - timedelta(days=1),
        max_value=datetime.now().date() + timedelta(days=3),
        help="Select a date to view matches. Only showing matches from yesterday up to 3 days in the future."
    )
    
    if date:
        with st.spinner("Fetching matches..."):
            date_str = date.strftime('%Y-%m-%d')
            all_matches = get_matches(date_str)
            
            if not all_matches:
                st.warning(f"No matches found for {date_str}. Try selecting a different date.")
                st.info("Note: Match data is typically available only for dates close to today.")
                return
            
            # Get unique leagues from matches
            available_leagues = {"All Matches": None}
            for match in all_matches:
                league_name = get_league_name(match)
                if league_name and league_name != "Unknown League":
                    available_leagues[league_name] = match.get('competition_id')
            
            # League filter
            selected_league = st.selectbox(
                "Select Competition",
                options=list(available_leagues.keys()),
                index=0,
                help="Filter matches by competition"
            )
            
            # Filter matches by selected league
            if selected_league != "All Matches":
                matches = [m for m in all_matches if get_league_name(m) == selected_league]
            else:
                matches = all_matches
            
            if not matches:
                st.info(f"No matches found for {selected_league} on {date_str}.")
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
                st.markdown(f"## {league_name}")
                
                for match in league_matches:
                    process_match_prediction(match)

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
        if st.button("🏠 Home", key="home"):
            st.query_params["page"] = "main"
            st.rerun()
            
    with col2:
        if st.button("📊 History", key="history"):
            st.query_params["page"] = "history"
            st.rerun()
            
    with col3:
        if st.button("🚪 Logout", key="logout"):
            st.session_state.logged_in = False
            st.query_params.clear()
            st.rerun()

def main():
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
    else:
        show_main_app()

if __name__ == "__main__":
    main()
