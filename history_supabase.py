# history.py

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from football_api import get_match_by_teams, get_match_result
from session_state import init_session_state, check_login_state
from match_analyzer import MatchAnalyzer
from supabase_db import SupabaseDB
import logging

class PredictionHistory:
    def __init__(self):
        """Initialize the Supabase database connection."""
        self.db = SupabaseDB()

    def init_database(self):
        """Initialize the Supabase database"""
        # No need to create tables as they are managed in Supabase dashboard
        self.db.init_database()
        
    def add_prediction(self, prediction_data):
        """Add a new prediction to the database"""
        try:
            # Ensure confidence is a float
            prediction_data['confidence'] = float(prediction_data.get('confidence', 0.0))
            prediction_data['match_id'] = str(prediction_data.get('match_id', ''))
            prediction_data['home_odds'] = float(prediction_data.get('home_odds', 0.0))
            prediction_data['draw_odds'] = float(prediction_data.get('draw_odds', 0.0))
            prediction_data['away_odds'] = float(prediction_data.get('away_odds', 0.0))
            prediction_data['bet_amount'] = float(prediction_data.get('bet_amount', 0.0))
            prediction_data['profit_loss'] = 0.0  # profit_loss starts at 0
            prediction_data['status'] = 'Pending'  # status starts as Pending
            prediction_data['actual_outcome'] = None  # actual_outcome starts as None
            
            result = self.db.add_prediction(prediction_data)
            logging.info(f"Successfully added prediction for {prediction_data['home_team']} vs {prediction_data['away_team']} with match_id: {prediction_data['match_id']}")
            return True if result else False
            
        except Exception as e:
            logging.error(f"Error adding prediction: {str(e)}")
            return False

    def update_prediction_result(self, match_id, actual_outcome, profit_loss):
        """Update prediction with actual result and profit/loss"""
        try:
            update_data = {
                'actual_outcome': actual_outcome,
                'profit_loss': profit_loss,
                'status': 'Completed'
            }
            result = self.db.update_prediction(match_id, update_data)
            return True if result else False
        except Exception as e:
            logging.error(f"Error updating prediction result: {str(e)}")
            return False

    def get_predictions(self, start_date=None, end_date=None, status=None, confidence_levels=None, leagues=None):
        """Get predictions with optional filters"""
        try:
            # Get base predictions from Supabase
            df = self.db.get_predictions(start_date, end_date)
            
            if df.empty:
                return df

            # Apply additional filters
            if status:
                df = df[df['status'] == status]

            # Handle multiple confidence levels
            if confidence_levels and "All" not in confidence_levels:
                confidence_mask = pd.Series(False, index=df.index)
                for level in confidence_levels:
                    if level == "High":
                        confidence_mask |= df['confidence'] >= 70
                    elif level == "Medium":
                        confidence_mask |= (df['confidence'] >= 50) & (df['confidence'] < 70)
                    elif level == "Low":
                        confidence_mask |= df['confidence'] < 50
                df = df[confidence_mask]

            # Handle multiple leagues
            if leagues and "All" not in leagues:
                df = df[df['league'].isin(leagues)]

            return df.sort_values('date', ascending=False)

        except Exception as e:
            logging.error(f"Error getting predictions: {str(e)}")
            return pd.DataFrame()

    def calculate_statistics(self, confidence_levels=None, leagues=None):
        """Calculate prediction statistics with optional confidence level and league filters"""
        try:
            df = self.get_predictions(confidence_levels=confidence_levels, leagues=leagues)
            
            if df.empty:
                return {
                    'total_predictions': 0,
                    'completed_predictions': 0,
                    'correct_predictions': 0,
                    'pending_predictions': 0,
                    'total_profit': 0.0,
                    'total_bet_amount': 0.0,
                    'success_rate': 0.0,
                    'roi': 0.0
                }

            stats = {
                'total_predictions': len(df),
                'completed_predictions': len(df[df['status'] == 'Completed']),
                'pending_predictions': len(df[df['status'] == 'Pending']),
                'correct_predictions': len(df[(df['predicted_outcome'] == df['actual_outcome']) & 
                                            (df['status'] == 'Completed')]),
                'total_profit': df[df['status'] == 'Completed']['profit_loss'].sum(),
                'total_bet_amount': df[df['status'] == 'Completed']['bet_amount'].sum()
            }
            
            # Calculate success rate and ROI
            completed_predictions = stats['completed_predictions']
            stats['success_rate'] = (stats['correct_predictions'] / completed_predictions * 100) if completed_predictions > 0 else 0
            stats['roi'] = (stats['total_profit'] / stats['total_bet_amount'] * 100) if stats['total_bet_amount'] > 0 else 0

            return stats

        except Exception as e:
            logging.error(f"Error calculating statistics: {str(e)}")
            return None

def style_dataframe(df):
    """Style the predictions dataframe with colors and formatting"""
    def color_status(val):
        if val == 'Completed':
            return 'background-color: #4CAF50; color: white'
        elif val == 'Pending':
            return 'background-color: #FFA726; color: white'
        return ''

    def color_profit_loss(val):
        try:
            val = float(val)
            if val > 0:
                return 'color: green'
            elif val < 0:
                return 'color: red'
        except:
            pass
        return ''

    def color_confidence(val):
        try:
            val = float(val)
            if val >= 70:
                return 'background-color: #4CAF50; color: white'
            elif val >= 50:
                return 'background-color: #FFA726; color: white'
            return 'background-color: #EF5350; color: white'
        except:
            return ''

    # Apply styles
    return df.style\
        .applymap(color_status, subset=['status'])\
        .applymap(color_profit_loss, subset=['profit_loss'])\
        .applymap(color_confidence, subset=['confidence'])\
        .format({
            'confidence': '{:.1f}%',
            'profit_loss': '£{:.2f}',
            'bet_amount': '£{:.2f}',
            'home_odds': '{:.2f}',
            'draw_odds': '{:.2f}',
            'away_odds': '{:.2f}'
        })

def get_confidence_level(confidence):
    """Convert confidence value to display text"""
    try:
        confidence = float(confidence)
        if confidence >= 70:
            return "High"
        elif confidence >= 50:
            return "Medium"
        return "Low"
    except:
        return "Unknown"

def show_history_page():
    """Display prediction history page"""
    # Initialize session state
    init_session_state()
    
    # Check if user is logged in
    if not check_login_state():
        st.warning("Please log in to view prediction history.")
        return

    st.title("Prediction History")

    # Initialize PredictionHistory
    history = PredictionHistory()

    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            datetime.now().date() - timedelta(days=7)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            datetime.now().date()
        )

    # Convert dates to string format
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Get all predictions for the date range
    all_predictions = history.get_predictions(start_date_str, end_date_str)

    # Get unique leagues for filtering
    leagues = ["All"]
    if not all_predictions.empty and 'league' in all_predictions.columns:
        leagues.extend(sorted(all_predictions['league'].unique()))

    # Confidence level and league filters
    col1, col2 = st.columns(2)
    with col1:
        confidence_levels = st.multiselect(
            "Filter by Confidence Level",
            ["All", "High", "Medium", "Low"],
            default=["All"]
        )
    with col2:
        selected_leagues = st.multiselect(
            "Filter by League",
            leagues,
            default=["All"]
        )

    # Get filtered predictions
    predictions = history.get_predictions(
        start_date_str,
        end_date_str,
        confidence_levels=confidence_levels,
        leagues=selected_leagues
    )

    # Calculate statistics
    stats = history.calculate_statistics(confidence_levels, selected_leagues)

    if stats:
        # Display statistics in a nice grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", stats['total_predictions'])
        with col2:
            st.metric("Completed", stats['completed_predictions'])
        with col3:
            st.metric("Pending", stats['pending_predictions'])
        with col4:
            success_rate = f"{stats['success_rate']:.1f}%" if stats['completed_predictions'] > 0 else "N/A"
            st.metric("Success Rate", success_rate)

        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Profit/Loss", f"£{stats['total_profit']:.2f}")
        with col2:
            st.metric("Total Bet Amount", f"£{stats['total_bet_amount']:.2f}")
        with col3:
            roi = f"{stats['roi']:.1f}%" if stats['total_bet_amount'] > 0 else "N/A"
            st.metric("ROI", roi)
        with col4:
            correct = stats['correct_predictions']
            st.metric("Correct Predictions", correct)

    # Display predictions
    if not predictions.empty:
        st.write("### Predictions")
        st.dataframe(style_dataframe(predictions), use_container_width=True)
    else:
        st.info("No predictions found for the selected filters.")

    # Add refresh button
    if st.button("Refresh Data"):
        st.experimental_rerun()

    # Add JavaScript for navigation
    st.markdown("""
    <script>
        function handleLogout() {
            window.location.href = "/?action=logout";
        }
        
        function goToHome() {
            window.location.href = "/";
        }
    </script>
    """, unsafe_allow_html=True)

    # Add navigation buttons
