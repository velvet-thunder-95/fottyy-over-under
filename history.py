# history.py



import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import os
import sys
from typing import List, Dict, Any, Tuple, Optional, Union

# Add current directory to path
sys.path.append('.')

# Import project modules
try:
    from football_api import get_match_by_teams, get_match_result
    from session_state import init_session_state, check_login_state
    from match_analyzer import MatchAnalyzer
    from supabase_db import SupabaseDB
    import filter_storage
except ImportError as e:
    st.error(f"Error importing required modules: {str(e)}")
    raise

# Import AG Grid components
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, ColumnsAutoSizeMode
except ImportError:
    st.error("AG Grid components not found. Please install with: pip install streamlit-aggrid")
    raise

class PredictionHistory:
    def __init__(self):
        """Initialize the Supabase database connection."""
        try:
            self.db = SupabaseDB()
        except Exception as e:
            st.error(f"Failed to initialize database: {str(e)}")
            raise

    def init_database(self):
        """Initialize the Supabase database"""
        try:
            self.db.init_database()
        except Exception as e:
            st.error(f"Failed to initialize database: {str(e)}")
            raise

    def add_prediction(self, prediction_data):
        """Add a new prediction to the database"""
        try:
            # Ensure confidence is a float
            confidence = float(prediction_data.get('confidence', 0.0))
            match_id = str(prediction_data.get('match_id', ''))
            
            # Create clean data for insertion with fixed $1 bet amount
            clean_data = {
                'date': prediction_data['date'],
                'league': prediction_data['league'],
                'home_team': prediction_data['home_team'],
                'away_team': prediction_data['away_team'],
                'predicted_outcome': prediction_data['predicted_outcome'],
                'actual_outcome': None,  # actual_outcome starts as None
                'home_odds': float(prediction_data['home_odds']),
                'draw_odds': float(prediction_data['draw_odds']),
                'away_odds': float(prediction_data['away_odds']),
                'confidence': confidence,
                'bet_amount': 1.0,  # Fixed $1 bet amount
                'profit_loss': 0.0,  # profit_loss starts at 0
                'status': 'Pending',  # status starts as Pending
                'match_id': match_id
            }
            
            # Insert into Supabase
            result = self.db.supabase.table('predictions').insert(clean_data).execute()
            
            logging.info(f"Successfully added prediction for {prediction_data['home_team']} vs {prediction_data['away_team']} with match_id: {match_id} and confidence: {confidence}")
            return True
            
        except Exception as e:
            logging.error(f"Error adding prediction: {str(e)}")
            return False

    def update_prediction(self, prediction_id, update_data):
        """Update prediction with new data"""
        try:
            result = self.db.supabase.table('predictions').update(update_data).eq('id', prediction_id).execute()
            return bool(result.data)
        except Exception as e:
            st.error(f"Error updating prediction: {str(e)}")
            return False
            
    def delete_prediction(self, prediction_id):
        """Delete a prediction"""
        try:
            result = self.db.supabase.table('predictions').delete().eq('id', prediction_id).execute()
            return bool(result.data)
        except Exception as e:
            st.error(f"Error deleting prediction: {str(e)}")
            return False
            
    def update_prediction_result(self, prediction_id, actual_outcome, profit_loss, home_score=None, away_score=None):
        """Update prediction with actual result and profit/loss"""
        try:
            update_data = {
                'actual_outcome': actual_outcome,
                'profit_loss': profit_loss,
                'status': 'Completed'
            }
            
            if home_score is not None:
                update_data['home_score'] = home_score
            if away_score is not None:
                update_data['away_score'] = away_score
                
            return self.update_prediction(prediction_id, update_data)
            
        except Exception as e:
            logging.error(f"Error updating prediction result: {str(e)}")
            return False

    def get_predictions(self, start_date=None, end_date=None, status=None, confidence_levels=None, leagues=None):
        """Get predictions with optional filters"""
        try:
            # Get base predictions from Supabase with date filters
            predictions = self.db.get_predictions(start_date=start_date, end_date=end_date)
            
            if predictions.empty:
                return predictions
                
            # Apply additional filters in memory
            if status:
                predictions = predictions[predictions['status'] == status]
                
            if confidence_levels and "All" not in confidence_levels:
                mask = pd.Series(False, index=predictions.index)
                
                for level in confidence_levels:
                    if level == "High":
                        mask |= predictions['confidence'] >= 70
                    elif level == "Medium":
                        mask |= (predictions['confidence'] >= 50) & (predictions['confidence'] < 70)
                    elif level == "Low":
                        mask |= predictions['confidence'] < 50
                        
                predictions = predictions[mask]
                
            if leagues and "All" not in leagues:
                predictions = predictions[predictions['league'].isin(leagues)]
            
            # Ensure numeric columns
            numeric_columns = ['bet_amount', 'confidence', 'home_odds', 'draw_odds', 'away_odds', 'profit_loss']
            for col in numeric_columns:
                if col in predictions.columns:
                    predictions[col] = pd.to_numeric(predictions[col], errors='coerce')
            
            # Ensure proper profit_loss values
            predictions.loc[predictions['status'] != 'Completed', 'profit_loss'] = 0.0
            
            # Sort by date (newest first)
            predictions = predictions.sort_values('date', ascending=False)
            
            logging.info(f"After filtering: {len(predictions)} records from {predictions['date'].min()} to {predictions['date'].max()}")
            return predictions
            
        except Exception as e:
            logging.error(f"Error getting predictions: {str(e)}")
            return pd.DataFrame()

    def update_match_results(self, match_id, result):
        """Update match results in the database"""
        try:
            # First get the match details from Supabase
            match_result = self.db.supabase.table('predictions').select('predicted_outcome,home_odds,draw_odds,away_odds').eq('match_id', match_id).execute()
            
            if not match_result.data:
                print(f"No prediction found for match {match_id}")
                return
            
            match_data = match_result.data[0]
            predicted_outcome = match_data['predicted_outcome']
            home_odds = match_data['home_odds']
            draw_odds = match_data['draw_odds']
            away_odds = match_data['away_odds']
            
            # Parse the result
            if isinstance(result, dict):
                home_score = result.get('home_score')
                away_score = result.get('away_score')
                status = result.get('status', 'Completed')
            else:
                home_score = result
                away_score = None
                status = 'Completed'
            
            # Initialize variables
            actual_outcome = None
            profit_loss = 0.0  # Default to 0
            bet_amount = 1.0  # Fixed $1 bet amount
            
            # Only calculate outcome and profit/loss if the match is completed
            if status == 'Completed' and home_score is not None and away_score is not None:
                # Determine actual outcome
                if home_score > away_score:
                    actual_outcome = 'HOME'
                elif away_score > home_score:
                    actual_outcome = 'AWAY'
                else:
                    actual_outcome = 'DRAW'
                    
                # Calculate profit/loss using $1 bet amount
                try:
                    if all([home_odds, draw_odds, away_odds]):  # Only if we have odds
                        # Convert odds to float and handle any string formatting
                        home_odds = float(str(home_odds).strip())
                        away_odds = float(str(away_odds).strip())
                        draw_odds = float(str(draw_odds).strip())
                        
                        if predicted_outcome == actual_outcome:
                            # Won: Calculate profit based on the predicted outcome's odds
                            if predicted_outcome == 'HOME':
                                profit_loss = float(round((home_odds * bet_amount) - bet_amount, 2))
                            elif predicted_outcome == 'AWAY':
                                profit_loss = float(round((away_odds * bet_amount) - bet_amount, 2))
                            else:  # DRAW
                                profit_loss = float(round((draw_odds * bet_amount) - bet_amount, 2))
                            print(f'Won bet! Odds: {home_odds}/{draw_odds}/{away_odds}, Profit: {profit_loss}')
                        else:
                            # Lost: Lose the bet amount
                            profit_loss = float(-bet_amount)
                            print(f'Lost bet! Predicted: {predicted_outcome}, Actual: {actual_outcome}, Loss: {profit_loss}')
                    else:
                        print(f'Missing odds: {home_odds}/{draw_odds}/{away_odds}')
                except (ValueError, TypeError) as e:
                    print(f'Error calculating profit/loss: {str(e)}')
            
            # Prepare update data
            update_data = {
                'status': status,
                'home_score': home_score,
                'away_score': away_score,
                'actual_outcome': actual_outcome,
                'profit_loss': profit_loss
            }
            
            # Debug print before update
            print(f"Updating match {match_id} with data: {update_data}")
            
            try:
                # Get current data to check what fields exist
                current = self.db.supabase.table('predictions')\
                    .select('*')\
                    .eq('match_id', match_id)\
                    .execute()
                
                if current.data:
                    # Only include fields that exist in the table
                    existing_fields = current.data[0].keys()
                    update_data = {k: v for k, v in update_data.items() if k in existing_fields}
                    
                    # Update with only existing fields
                    self.db.supabase.table('predictions')\
                        .update(update_data)\
                        .eq('match_id', match_id)\
                        .execute()
                    print(f"Successfully updated match {match_id} with fields: {list(update_data.keys())}")
            except Exception as e:
                print(f"Error updating match {match_id}: {str(e)}")
            
        except Exception as e:
            print(f"Error processing match {match_id}: {str(e)}")

    def update_match_results_all(self):
        """Update match results for pending predictions only"""
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        analyzer = MatchAnalyzer("633379bdd5c4c3eb26919d8570866801e1c07f399197ba8c5311446b8ea77a49")
        
        # Get only pending predictions that have a match_id
        result = self.db.supabase.table('predictions') \
            .select('id,match_id,home_team,away_team,date,predicted_outcome,home_odds,draw_odds,away_odds') \
            .filter('match_id', 'neq', None) \
            .filter('status', 'eq', 'Pending') \
            .execute()
        pending_predictions = result.data
        logger.info(f"Found {len(pending_predictions)} pending predictions to check")
        
        updated_count = 0
        for pred in pending_predictions:
            try:
                match_id = pred['match_id']
                home_team = pred['home_team']
                away_team = pred['away_team']
                match_date = pred['date']
                
                if not match_id:
                    logger.warning(f"Missing match_id for {home_team} vs {away_team} on {match_date}")
                    continue
                
                # Get current match result from API
                result = analyzer.analyze_match_result(match_id)
                if not result:
                    logger.debug(f"Match still pending: {home_team} vs {away_team}")
                    continue
                
                # Only update if the API shows the match is completed
                api_status = result.get('status')
                if api_status == 'Completed':
                    # Update the result
                    self.update_match_results(match_id, result)
                    logger.info(f"Updated {home_team} vs {away_team} - Match completed with result")
                    updated_count += 1
                
            except Exception as e:
                logger.error(f"Error processing match {match_id}: {str(e)}")
                continue
        
        logger.info(f"Updated {updated_count} pending matches")

    def calculate_statistics(self, confidence_levels=None, leagues=None, start_date=None, end_date=None):
        """Calculate prediction statistics with optional confidence level and league filters"""
        try:
            # Get all predictions first using our paginated get_predictions method
            predictions = self.get_predictions(
                start_date=start_date,
                end_date=end_date,
                confidence_levels=confidence_levels,
                leagues=leagues
            )
            
            if predictions.empty:
                return [0, 0, 0.0, 0.0, 0.0], 0
            
            # Calculate statistics
            completed_predictions = predictions[predictions['status'] == 'Completed']
            pending_predictions = predictions[predictions['status'] == 'Pending']
            
            total_predictions = len(predictions)
            completed_count = len(completed_predictions)
            pending_count = len(pending_predictions)
            
            if completed_count == 0:
                return [total_predictions, 0, 0.0, 0.0, 0.0], pending_count
            
            # Calculate correct predictions
            correct_predictions = len(
                completed_predictions[
                    completed_predictions['predicted_outcome'] == 
                    completed_predictions['actual_outcome']
                ]
            )
            
            # Calculate success rate
            success_rate = (correct_predictions / completed_count * 100) if completed_count > 0 else 0.0
            
            # Calculate total profit/loss and ROI
            total_profit = completed_predictions['profit_loss'].sum()
            
            # Calculate ROI using completed bets only (each bet is £1)
            roi = (total_profit / completed_count * 100) if completed_count > 0 else 0.0
            
            # Debug info
            logging.info(f"Statistics calculation:")
            logging.info(f"Total predictions: {total_predictions}")
            logging.info(f"Completed predictions: {completed_count}")
            logging.info(f"Pending predictions: {pending_count}")
            logging.info(f"Correct predictions: {correct_predictions}")
            logging.info(f"Success rate: {success_rate:.2f}%")
            logging.info(f"Total profit: £{total_profit:.2f}")
            logging.info(f"ROI: {roi:.2f}%")
            logging.info(f"Date range: {predictions['date'].min()} to {predictions['date'].max()}")
            
            return [total_predictions, correct_predictions, success_rate, total_profit, roi], pending_count
            
        except Exception as e:
            logging.error(f"Error calculating statistics: {str(e)}")
            return [0, 0, 0.0, 0.0, 0.0], 0

def edit_prediction(history, pred_id, df):
    """Display a form to edit a prediction"""
    prediction = df[df['id'] == pred_id].iloc[0].to_dict()
    
    with st.form(key=f"edit_form_{pred_id}"):
        st.subheader(f"Edit Prediction: {prediction['home_team']} vs {prediction['away_team']}")
        
        # Create form columns
        col1, col2 = st.columns(2)
        
        with col1:
            date = st.date_input("Date", value=pd.to_datetime(prediction['date']))
            home_team = st.text_input("Home Team", value=prediction['home_team'])
            away_team = st.text_input("Away Team", value=prediction['away_team'])
            league = st.text_input("League", value=prediction['league'])
            
        with col2:
            status = st.selectbox(
                "Status",
                ["Pending", "Completed"],
                index=0 if prediction.get('status') == "Pending" else 1
            )
            predicted_outcome = st.selectbox(
                "Predicted Outcome",
                ["HOME", "DRAW", "AWAY"],
                index=["HOME", "DRAW", "AWAY"].index(prediction['predicted_outcome'])
            )
            actual_outcome = st.selectbox(
                "Actual Outcome",
                ["", "HOME", "DRAW", "AWAY"],
                index=["", "HOME", "DRAW", "AWAY"].index(prediction.get('actual_outcome', ""))
            )
            confidence = st.slider("Confidence", 0, 100, int(prediction.get('confidence', 50)))
        
        # Odds
        st.subheader("Odds")
        odds_col1, odds_col2, odds_col3 = st.columns(3)
        with odds_col1:
            home_odds = st.number_input("Home Odds", value=float(prediction.get('home_odds', 2.0)), step=0.1)
        with odds_col2:
            draw_odds = st.number_input("Draw Odds", value=float(prediction.get('draw_odds', 3.0)), step=0.1)
        with odds_col3:
            away_odds = st.number_input("Away Odds", value=float(prediction.get('away_odds', 3.5)), step=0.1)
        
        # Form buttons
        submit_button = st.form_submit_button("💾 Save Changes")
        cancel_button = st.form_submit_button("❌ Cancel")
        
        if submit_button:
            # Prepare update data
            update_data = {
                'date': date.strftime('%Y-%m-%d %H:%M:%S'),
                'home_team': home_team,
                'away_team': away_team,
                'league': league,
                'status': status,
                'predicted_outcome': predicted_outcome,
                'actual_outcome': actual_outcome if actual_outcome else None,
                'home_odds': home_odds,
                'draw_odds': draw_odds,
                'away_odds': away_odds,
                'confidence': confidence
            }
            
            # Update in database
            if history.update_prediction(pred_id, update_data):
                st.success("✅ Prediction updated successfully!")
                st.session_state.refresh_data = True
                st.rerun()
            else:
                st.error("❌ Failed to update prediction.")

def delete_prediction(history, pred_id):
    """Delete a prediction from the database"""
    if history.delete_prediction(pred_id):
        st.success("✅ Prediction deleted successfully!")
        st.session_state.refresh_data = True
    else:
        st.error("❌ Failed to delete prediction.")

def style_dataframe(df):
    """Style the predictions dataframe with colors and formatting"""
    def style_row(row):
        styles = {}
        
        # Base style for all cells
        base_style = 'font-size: 14px; padding: 12px 15px; border-bottom: 1px solid #e0e0e0;'
        
        # Background color based on status
        if row.get('Status') == 'Pending':
            bg_color = '#f8f9fa'  # Light gray for pending
        elif row.get('Result') == '✅ Won':
            bg_color = '#e8f5e9'  # Light green for wins
        elif row.get('Result') == '❌ Lost':
            bg_color = '#fce4ec'  # Light red for losses
        else:
            bg_color = '#ffffff'  # White for others
            
        # Add background color to base style
        base_style += f' background-color: {bg_color};'
        
        # Style Result column
        if row.get('Result') == '✅ Won':
            styles['Result'] = base_style + 'color: #28a745; font-weight: bold'
        elif row.get('Result') == '❌ Lost':
            styles['Result'] = base_style + 'color: #dc3545; font-weight: bold'
        else:
            styles['Result'] = base_style + 'color: #6c757d; font-style: italic'
            
        # Style Profit/Loss column
        try:
            if row.get('Profit/Loss') == '-':
                styles['Profit/Loss'] = base_style + 'color: #6c757d'
            elif row.get('Profit/Loss', '').startswith('+'):
                styles['Profit/Loss'] = base_style + 'color: #28a745; font-weight: bold'  # Green for profits
            elif row.get('Profit/Loss', '').startswith('-'):
                styles['Profit/Loss'] = base_style + 'color: #dc3545; font-weight: bold'  # Red for losses
            else:
                styles['Profit/Loss'] = base_style + 'color: #6c757d'  # Gray for zero/neutral
        except (AttributeError, TypeError):
            styles['Profit/Loss'] = base_style + 'color: #6c757d'
            
        # Style Confidence column
        confidence_style = base_style
        if row.get('Confidence') == 'High':
            confidence_style += 'background-color: #d4edda; color: #155724; font-weight: bold'
        elif row.get('Confidence') == 'Medium':
            confidence_style += 'background-color: #fff3cd; color: #856404; font-weight: bold'
        elif row.get('Confidence') == 'Low':
            confidence_style += 'background-color: #f8d7da; color: #721c24; font-weight: bold'
        styles['Confidence'] = confidence_style
            
        # Status column styling
        if row.get('Status') == 'Completed':
            styles['Status'] = base_style + 'color: #28a745'
        elif row.get('Status') == 'Pending':
            styles['Status'] = base_style + 'color: #ffc107'
        else:
            styles['Status'] = base_style + 'color: #6c757d'
            
        # Default style for other columns
        for col in df.columns:
            if col not in styles:
                styles[col] = base_style
                
        return pd.Series(styles)
    
    # Apply the styles and add table-level styling
    return df.style.apply(style_row, axis=1).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#f8f9fa'),
            ('color', '#333333'),
            ('font-weight', '600'),
            ('font-size', '14px'),
            ('text-align', 'left'),
            ('padding', '12px 15px'),
            ('border-bottom', '2px solid #dee2e6')
        ]},
        {'selector': 'td', 'props': [
            ('text-align', 'left'),
            ('white-space', 'nowrap'),
            ('min-width', '100px')
        ]},
        {'selector': 'table', 'props': [
            ('border-collapse', 'collapse'),
            ('width', '100%'),
            ('margin', '10px 0'),
            ('font-family', '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif')
        ]},
        {'selector': 'tr:hover td', 'props': [
            ('background-color', 'rgba(0,0,0,0.05) !important')
        ]}
    ])

def get_confidence_level(confidence):
    """Convert confidence value to display text"""
    try:
        # Handle None, NaN, and empty values
        if confidence is None or pd.isna(confidence) or confidence == "":
            return "Unknown"
            
        # Convert to float and handle string values
        conf_value = float(str(confidence).strip())
        
        # Categorize confidence
        if conf_value >= 70:
            return "High"
        elif conf_value >= 50:
            return "Medium"
        elif conf_value >= 0:
            return "Low"
        else:
            return "Unknown"
    except (ValueError, TypeError, AttributeError):
        return "Unknown"

def show_history_page():
    """Display prediction history page with AG Grid editing capabilities"""
    st.markdown("""
        <style>
        .ag-header-cell-label {
            justify-content: center;
        }
        .ag-cell {
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .ag-theme-streamlit {
            --ag-font-size: 14px;
            --ag-font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }
        .action-button {
            padding: 4px 8px;
            margin: 2px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.3s;
        }
        .edit-button {
            background-color: #4CAF50;
            color: white;
        }
        .delete-button {
            background-color: #f44336;
            color: white;
        }
        .action-button:hover {
            opacity: 0.8;
            transform: translateY(-1px);
        }
        </style>
    """, unsafe_allow_html=True)
    
    if not check_login_state():
        st.warning("Please log in to view prediction history.")
        return

    # Custom CSS for styling
    st.markdown("""
        <style>
        /* Main container styling */
        .main {
            background-color: #f8f9fa;
            padding: 20px;
        }
        
        /* Title styling */
        .title-container {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 30px 20px;
            border-radius: 15px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .title {
            color: white;
            font-size: 2.2em;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        }
        
        /* Metrics styling */
        .metrics-container {
            width: 200px;
            margin-bottom: 20px;
        }
        
        .metric-box {
            background: white;
            padding: 8px 12px;
            border-radius: 6px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            text-align: left;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .metric-label {
            color: #666;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin: 0;
        }
        
        .metric-value {
            font-size: 0.9rem;
            font-weight: bold;
            color: #2c5282;
            margin: 0;
        }
        
        .positive-value {
            color: #48bb78;
        }
        
        .negative-value {
            color: #f56565;
        }
        </style>
    """, unsafe_allow_html=True)

    # Display title with custom styling
    st.markdown("""
        <div class="title-container">
            <h1 class="title">Prediction History</h1>
        </div>
    """, unsafe_allow_html=True)
    
    try:
        # Initialize session state variables if they don't exist
        if 'history_data_loaded' not in st.session_state:
            st.session_state.history_data_loaded = False
            
        # Initialize PredictionHistory
        history = PredictionHistory()
        
        # Handle row editing
        if 'editing_row' not in st.session_state:
            st.session_state.editing_row = None
            
        # Initialize or update history_df in session state
        if 'history_df' not in st.session_state or st.session_state.get('refresh_data', False):
            # Get initial data with default filters
            df = history.get_predictions()
            st.session_state.history_df = df
            st.session_state.history_data_loaded = True
            st.session_state.refresh_data = False
        
        # Initialize filter state variables
        if 'history_filter_params' not in st.session_state:
            all_predictions = history.get_predictions()
            if not all_predictions.empty:
                min_date = pd.to_datetime(all_predictions['date']).min().date()
                max_date = pd.to_datetime(all_predictions['date']).max().date()
            else:
                min_date = datetime.now().date() - timedelta(days=30)
                max_date = datetime.now().date()
                
            unique_leagues = sorted(all_predictions['league'].unique()) if not all_predictions.empty else []
            
            st.session_state.history_filter_params = {
                'start_date': min_date,
                'end_date': max_date,
                'leagues': ["All"],
                'confidence_levels': ["All"],
                'status': "All",
                'min_date': min_date,
                'max_date': max_date,
                'unique_leagues': unique_leagues
            }
        
        # --- Filters section should appear above savable filters ---
        # Add date filter in sidebar
        st.sidebar.markdown("## Filters", help="Filter your prediction history")
        
        # Get filter parameters from session state
        params = st.session_state.history_filter_params
        
        # Create a form to prevent automatic rerun on every input change
        with st.sidebar.form(key="history_filter_form"):
            # Date Range Filter
            start_date = st.date_input(
                "Start Date",
                value=params['start_date'],
                min_value=params['min_date'],
                max_value=params['max_date'],
                help="Filter predictions from this date"
            )
            
            end_date = st.date_input(
                "End Date",
                value=params['end_date'],
                min_value=params['min_date'],
                max_value=params['max_date'],
                help="Filter predictions until this date"
            )
            
            # Validate dates
            if start_date > end_date:
                st.error("Error: End date must be after start date")
            
            # League Filter
            leagues = st.multiselect(
                "Select Competitions",
                options=["All"] + params['unique_leagues'],
                default=params['leagues'],
                help="Filter predictions by competition. Select multiple competitions or 'All'"
            )
            
            # Handle empty selection
            if not leagues:
                leagues = ["All"]
            
            # Confidence Level Filter
            confidence_levels = st.multiselect(
                "Confidence Levels",
                options=["All", "High", "Medium", "Low"],
                default=params['confidence_levels'],
                help="Filter predictions by confidence level: High (≥70%), Medium (50-69%), Low (<50%). Select multiple levels or 'All'"
            )
            
            # Handle empty selection
            if not confidence_levels:
                confidence_levels = ["All"]
                
            # Status filter
            status = st.selectbox(
                "Status",
                options=["All", "Completed", "Pending"],
                index=0 if params['status'] == "All" else (1 if params['status'] == "Completed" else 2),
                help="Filter by prediction status"
            )
            
            # Submit button
            submit_button = st.form_submit_button(label="Apply Filters", type="primary")
        
        # Process form submission
        if submit_button:
            # Update filter parameters in session state
            st.session_state.history_filter_params.update({
                'start_date': start_date,
                'end_date': end_date,
                'leagues': leagues,
                'confidence_levels': confidence_levels,
                'status': status
            })
            
            # Format dates for database query
            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")  # Include end date
            
            # Get filtered data
            predictions = history.get_predictions(
                start_date=start_date_str,
                end_date=end_date_str,
                status=None if status == "All" else status,
                confidence_levels=None if "All" in confidence_levels else confidence_levels,
                leagues=None if "All" in leagues else leagues
            )
            
            # Update the dataframe in session state
            st.session_state.history_df = predictions
        
        # Use values from session state for the rest of the code
        params = st.session_state.history_filter_params
        start_date = params['start_date']
        end_date = params['end_date']
        selected_leagues = params['leagues']
        confidence_levels = params['confidence_levels']
        selected_status = params['status']
        
        # Format dates for display and calculations
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")  # Include end date
        
        # --- History Savable Filters UI ---
        st.sidebar.markdown('### History Page Filter Presets', help="Save and apply filter combinations for the history page.")
        with st.sidebar.container():
            history_filter_name = st.text_input("Save History Filter Preset", key="history_filter_name")
            if st.button("Save History Filter Preset", key="save_history_filter"):
                if history_filter_name:
                    # Save exactly what the user selected, even if it's only one league
                    leagues_to_save = params['leagues'].copy()
                    if "All" in leagues_to_save and len(leagues_to_save) > 1:
                        leagues_to_save.remove("All")  # Remove 'All' if other leagues are selected
                    if leagues_to_save == ["All"]:
                        leagues_to_save = []  # If only 'All' is selected, treat as no specific filter
                    st.session_state.history_saved_filters = filter_storage.save_history_filter(
                        history_filter_name,
                        params['start_date'].strftime("%Y-%m-%d"),
                        params['end_date'].strftime("%Y-%m-%d"),
                        leagues_to_save,
                        params['confidence_levels'],
                        params['status']
                    )
                    st.success(f"Saved history filter preset '{history_filter_name}'!")
                else:
                    st.error("Please enter a filter name.")
            # --- Ensure history_saved_filters is initialized before use ---
            if 'history_saved_filters' not in st.session_state:
                st.session_state.history_saved_filters = filter_storage.load_history_saved_filters()
            if st.session_state.history_saved_filters:
                st.markdown("#### Saved History Filters")
                for idx, sf in enumerate(st.session_state.history_saved_filters):
                    st.write(f"**{sf['name']}** | {sf['start_date']} to {sf['end_date']} | Leagues: {', '.join(sf['leagues'])} | Confidence: {', '.join(sf['confidence'])} | Status: {sf['status'] if sf['status'] else 'All'}")
                    cols = st.columns([1,1])
                    if cols[0].button("Apply", key=f"apply_hist_filter_{idx}"):
                        # Update the filter parameters in session state
                        leagues_to_apply = sf['leagues'] if sf['leagues'] else ["All"]
                        confidence_to_apply = sf['confidence'] if sf['confidence'] else ["All"]
                        status_to_apply = sf['status'] if sf['status'] else "All"
                        
                        # Update filter parameters
                        st.session_state.history_filter_params.update({
                            'start_date': pd.to_datetime(sf['start_date']).date(),
                            'end_date': pd.to_datetime(sf['end_date']).date(),
                            'leagues': leagues_to_apply,
                            'confidence_levels': confidence_to_apply,
                            'status': status_to_apply
                        })
                        
                        # Format dates for database query
                        start_date_str = sf['start_date']
                        end_date_str = (pd.to_datetime(sf['end_date']).date() + timedelta(days=1)).strftime("%Y-%m-%d")
                        
                        # Get filtered data based on preset
                        predictions = history.get_predictions(
                            start_date=start_date_str,
                            end_date=end_date_str,
                            status=None if status_to_apply == "All" else status_to_apply,
                            confidence_levels=None if "All" in confidence_to_apply else confidence_to_apply,
                            leagues=None if "All" in leagues_to_apply else leagues_to_apply
                        )
                        
                        # Update the dataframe in session state
                        st.session_state.history_df = predictions
                        st.rerun()
                    if cols[1].button("Delete", key=f"delete_hist_filter_{idx}"):
                        st.session_state.history_saved_filters = filter_storage.delete_history_filter(sf['id'])
                        st.rerun()
        
        # Get the filtered predictions from session state
        predictions = st.session_state.history_df
        
        # Debug info
        logging.info(f"Date range: {start_date_str} to {end_date_str}")
        logging.info(f"Total predictions: {len(predictions) if not predictions.empty else 0}")
        
        # If predictions is None or empty, show message
        if predictions is None or predictions.empty:
            st.info("No predictions found for the selected filters.")
            return
        
        if not predictions.empty:
            # Update any pending predictions
            history.update_match_results_all()
            
            # Calculate statistics
            current_confidence = None if "All" in confidence_levels else confidence_levels
            current_leagues = None if "All" in selected_leagues else selected_leagues
            stats, pending_count = history.calculate_statistics(
                confidence_levels=current_confidence,
                leagues=current_leagues,
                start_date=start_date_str,
                end_date=end_date_str
            )
            
            # Create metrics container
            st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
            
            # Display each metric
            metrics = [
                {"label": "Total Predictions", "value": float(stats[0]), "is_percentage": False, "is_currency": False},
                {"label": "Correct Predictions", "value": float(stats[1]), "is_percentage": False, "is_currency": False},
                {"label": "Success Rate", "value": float(stats[2]), "is_percentage": True, "is_currency": False},
                {"label": "Total Profit", "value": float(stats[3]), "is_currency": True, "is_percentage": False},
                {"label": "ROI", "value": float(stats[4]), "is_percentage": True, "is_currency": False},
                {"label": "Pending Predictions", "value": float(pending_count), "is_percentage": False, "is_currency": False}
            ]
            
            for metric in metrics:
                if metric.get("is_currency"):
                    formatted_value = f"{metric['value']:.2f}U"
                elif metric.get("is_percentage"):
                    if metric['label'] == "ROI":
                        formatted_value = f"{metric['value']:.2f}%"
                    else:
                        formatted_value = f"{metric['value']:.1f}%"
                else:
                    formatted_value = f"{metric['value']:.1f}"
                
                value_class = ""
                if metric.get("is_currency") or metric.get("is_percentage"):
                    try:
                        value = float(metric['value'])
                        value_class = " positive-value" if value > 0 else " negative-value" if value < 0 else ""
                    except (ValueError, TypeError):
                        value_class = ""
                
                st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">{metric['label']}</div>
                        <div class="metric-value{value_class}">{formatted_value}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display predictions table
            if not predictions.empty:
                st.markdown("""
                    <h2 style='color: #1e3c72; font-size: 1.5em; margin: 30px 0 20px;'>
                        Recent Predictions
                    </h2>
                """, unsafe_allow_html=True)
                
                try:
                    # Ensure predictions is a DataFrame
                    if not isinstance(predictions, pd.DataFrame):
                        predictions = pd.DataFrame(predictions)
                    
                    # Convert confidence to numeric and create display version
                    predictions['confidence_num'] = pd.to_numeric(predictions['confidence'], errors='coerce')
                    predictions['Confidence'] = predictions['confidence_num'].apply(get_confidence_level)
                    
                    # Convert date to datetime
                    predictions['date'] = pd.to_datetime(predictions['date']).dt.strftime('%Y-%m-%d')
                    
                    # Create Result column
                    predictions['Result'] = predictions.apply(
                        lambda x: '✅ Won' if pd.notna(x['predicted_outcome']) and pd.notna(x['actual_outcome']) and x['predicted_outcome'] == x['actual_outcome']
                        else '❌ Lost' if pd.notna(x['actual_outcome'])
                        else '⏳ Pending',
                        axis=1
                    )
                    
                    # Define display columns mapping
                    display_columns = {
                        'date': 'Date',
                        'league': 'League',
                        'home_team': 'Home Team',
                        'away_team': 'Away Team',
                        'predicted_outcome': 'Prediction',
                        'Confidence': 'Confidence',
                        'actual_outcome': 'Actual Outcome',
                        'Result': 'Result',
                        'profit_loss': 'Profit/Loss',
                        'status': 'Status'  # Add back status column
                    }
                    
                    # Create final dataframe with explicit column selection
                    final_df = predictions[list(display_columns.keys())].copy()
                    
                    # Format profit/loss values
                    def format_pl(row):
                        if pd.isna(row['profit_loss']) or row['status'] != 'Completed':
                            return '-'
                        try:
                            value = float(row['profit_loss'])
                            return f'+{value:.2f}U' if value > 0 else f'-{abs(value):.2f}U' if value < 0 else '0.00U'
                        except (ValueError, TypeError):
                            return '-'
                    
                    # Display the filtered data with a simple table
                    if not final_df.empty:
                        # Display the dataframe with Streamlit's native table
                        st.dataframe(
                            final_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "id": None,  # Hide ID column
                                "date": "Date",
                                "league": "League",
                                "home_team": "Home Team",
                                "away_team": "Away Team",
                                "predicted_outcome": "Prediction",
                                "confidence": st.column_config.NumberColumn(
                                    "Confidence",
                                    format="%.0f%%"
                                ),
                                "status": "Status"
                            }
                        )
                        
                        # Add edit/delete buttons for each row
                        st.subheader("Edit Prediction")
                        with st.form(key="edit_prediction_form"):
                            row_id = st.selectbox(
                                "Select prediction to edit",
                                options=final_df.apply(
                                    lambda x: f"{x['home_team']} vs {x['away_team']} ({x['date'].split()[0]})", 
                                    axis=1
                                )
                            )
                            
                            if row_id:
                                selected_idx = final_df.apply(
                                    lambda x: f"{x['home_team']} vs {x['away_team']} ({x['date'].split()[0]})", 
                                    axis=1
                                ).tolist().index(row_id)
                                selected_row = final_df.iloc[selected_idx].to_dict()
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    date = st.date_input("Date", value=pd.to_datetime(selected_row.get('date', datetime.now())))
                                    home_team = st.text_input("Home Team", value=selected_row.get('home_team', ''))
                                    away_team = st.text_input("Away Team", value=selected_row.get('away_team', ''))
                                    league = st.text_input("League", value=selected_row.get('league', ''))
                                    
                                with col2:
                                    status = st.selectbox(
                                        "Status",
                                        ["Pending", "Completed"],
                                        index=0 if selected_row.get('status') == "Pending" else 1
                                    )
                                    predicted_outcome = st.selectbox(
                                        "Predicted Outcome",
                                        ["HOME", "DRAW", "AWAY"],
                                        index=["HOME", "DRAW", "AWAY"].index(selected_row.get('predicted_outcome', 'HOME'))
                                    )
                                    actual_outcome = st.selectbox(
                                        "Actual Outcome",
                                        ["", "HOME", "DRAW", "AWAY"],
                                        index=["", "HOME", "DRAW", "AWAY"].index(selected_row.get('actual_outcome', ""))
                                    )
                                    confidence = st.slider("Confidence", 0, 100, int(selected_row.get('confidence', 50)))
                                
                                # Form buttons
                                col1, col2 = st.columns([1, 1])
                                
                                with col1:
                                    if st.form_submit_button("💾 Save Changes"):
                                        # Prepare update data
                                        update_data = {
                                            'date': date.strftime('%Y-%m-%d %H:%M:%S'),
                                            'home_team': home_team,
                                            'away_team': away_team,
                                            'league': league,
                                            'status': status,
                                            'predicted_outcome': predicted_outcome,
                                            'actual_outcome': actual_outcome if actual_outcome else None,
                                            'confidence': confidence
                                        }
                                        
                                        try:
                                            # Update in database
                                            if history.update_prediction(selected_row['id'], update_data):
                                                st.success("✅ Prediction updated successfully!")
                                                st.session_state.refresh_data = True
                                                st.rerun()
                                            else:
                                                st.error("❌ Failed to update prediction.")
                                        except Exception as e:
                                            st.error(f"❌ Error updating prediction: {str(e)}")
                                
                                with col2:
                                    if st.form_submit_button("🗑️ Delete Prediction"):
                                        try:
                                            if history.delete_prediction(selected_row['id']):
                                                st.success("✅ Prediction deleted successfully!")
                                                st.session_state.refresh_data = True
                                                st.rerun()
                                            else:
                                                st.error("❌ Failed to delete prediction.")
                                        except Exception as e:
                                            st.error(f"❌ Error deleting prediction: {str(e)}")
                    else:
                        st.warning("No predictions found matching the selected filters.")
                    
                    # Add some space at the bottom of the page
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.exception(e)
                    
            else:
                st.info("No predictions found for the selected date range.")
        
    except Exception as e:
        st.error(f"Error displaying predictions table: {str(e)}")
        st.exception(e)

    # Add navigation buttons
    def add_navigation_buttons():
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("Home", key="home"):
                st.query_params["page"] = "main"
                st.rerun()
                
        with col2:
            if st.button("Trend history", key="graph"):
                st.query_params["page"] = "graph"
                st.rerun()
                
        with col3:
            if st.button("Refresh", key="refresh"):
                st.rerun()
        
    add_navigation_buttons()


