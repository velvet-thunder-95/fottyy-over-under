# history.py

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from football_api import get_match_by_teams, get_match_result
from session_state import init_session_state, check_login_state
from match_analyzer import MatchAnalyzer
from supabase_db import SupabaseDB
import logging
import sys
import time
sys.path.append('.')
import importlib
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
filter_storage = importlib.import_module('filter_storage')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_date_format(date_value):
    """Convert any date format to YYYY-MM-DD string"""
    try:
        # Handle different date formats
        if pd.isna(date_value) or date_value is None:
            return ""
        
        # If it's already a properly formatted string, return it
        if isinstance(date_value, str) and len(date_value) == 10 and date_value[4] == '-' and date_value[7] == '-':
            # Validate that it's a valid date
            try:
                pd.to_datetime(date_value)
                return date_value
            except:
                pass  # If validation fails, continue with other methods
            
        # Handle timestamp in milliseconds (as int or float)
        if isinstance(date_value, (int, float)) or (
           isinstance(date_value, str) and date_value.replace('.', '', 1).isdigit()):
            # Convert to string first to check length
            date_str = str(date_value).split('.')[0]  # Remove decimal part if present
            
            # If it looks like a Unix timestamp in milliseconds
            if len(date_str) > 10:
                try:
                    # Try milliseconds first
                    date_obj = pd.to_datetime(float(date_value), unit='ms')
                    return date_obj.strftime('%Y-%m-%d')
                except:
                    # If that fails, try seconds
                    try:
                        date_obj = pd.to_datetime(float(date_value) / 1000, unit='s')
                        return date_obj.strftime('%Y-%m-%d')
                    except:
                        pass  # Continue to general handling
        
        # General handling for any other date format
        date_obj = pd.to_datetime(date_value)
        return date_obj.strftime('%Y-%m-%d')
        
    except Exception as e:
        logger.error(f"Date conversion error for {date_value}: {e}")
        # Return original value as fallback
        return str(date_value)

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
                
            result = self.db.supabase.table('predictions').update(update_data).eq('id', prediction_id).execute()
            return True
            
        except Exception as e:
            logging.error(f"Error updating prediction result: {str(e)}")
            return False
            
    def update_prediction(self, prediction_id, update_data):
        """Update prediction with custom data"""
        try:
            # Convert prediction_id to int if it's a string
            if isinstance(prediction_id, str):
                try:
                    prediction_id = int(prediction_id)
                except ValueError:
                    logging.error(f"Invalid prediction ID format: {prediction_id}")
                    return False
            
            # Verify the prediction exists before updating
            check = self.db.supabase.table('predictions').select('id').eq('id', prediction_id).execute()
            if not check.data:
                logging.error(f"Prediction ID {prediction_id} not found in database")
                return False
                
            # Ensure date is properly formatted
            if 'date' in update_data:
                update_data['date'] = ensure_date_format(update_data['date'])
                
            # Ensure numeric values are properly formatted
            numeric_fields = ['confidence', 'home_odds', 'draw_odds', 'away_odds', 'profit_loss', 'home_score', 'away_score']
            for field in numeric_fields:
                if field in update_data and update_data[field] is not None:
                    try:
                        update_data[field] = float(update_data[field])
                    except (ValueError, TypeError):
                        logging.error(f"Invalid numeric value for {field}: {update_data[field]}")
                        return False
            
            # Log the update data for debugging
            logging.info(f"Updating prediction ID {prediction_id} with data: {update_data}")
                
            # Perform the update
            result = self.db.supabase.table('predictions').update(update_data).eq('id', prediction_id).execute()
            logging.info(f"Successfully updated prediction ID: {prediction_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error updating prediction: {str(e)}")
            return False
            
    def delete_prediction(self, prediction_id):
        """Delete a prediction from the database"""
        try:
            # Convert prediction_id to int if it's a string
            if isinstance(prediction_id, str):
                try:
                    prediction_id = int(prediction_id)
                except ValueError:
                    logging.error(f"Invalid prediction ID format: {prediction_id}")
                    return False
            
            # Verify the prediction exists before deleting
            check = self.db.supabase.table('predictions').select('id').eq('id', prediction_id).execute()
            if not check.data:
                # If the prediction doesn't exist, it might have already been deleted
                # We'll consider this a success since the end result is what we want
                logging.info(f"Prediction ID {prediction_id} not found in database - may have already been deleted")
                return True
                
            # Delete the prediction
            result = self.db.supabase.table('predictions').delete().eq('id', prediction_id).execute()
            logging.info(f"Successfully deleted prediction ID: {prediction_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error deleting prediction: {str(e)}")
            return False

    def get_predictions(self, start_date=None, end_date=None, status=None, confidence_levels=None, leagues=None):
        """Get predictions with optional filters"""
        try:
            # Get base predictions from Supabase with date filters
            predictions = self.db.get_predictions(start_date=start_date, end_date=end_date)
            
            if predictions.empty:
                return predictions
                
            # Ensure dates are properly formatted
            if 'date' in predictions.columns:
                predictions['date'] = predictions['date'].apply(ensure_date_format)
                
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
    """Display prediction history page"""
    st.markdown("""
        <style>
        .stDataFrame {
            font-size: 14px;
            width: 100%;
        }
        .stDataFrame [data-testid="StyledDataFrameDataCell"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            padding: 12px 15px;
            color: #333333;
        }
        .stDataFrame [data-testid="StyledDataFrameHeaderCell"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            padding: 12px 15px;
            background-color: #f8f9fa;
            color: #333333;
            font-weight: 600;
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
            
        if 'history_df' not in st.session_state:
            # Initialize PredictionHistory
            history = PredictionHistory()
            # Get initial data with default filters
            df = history.get_predictions()
            st.session_state.history_df = df
            st.session_state.history_data_loaded = True
        else:
            history = PredictionHistory()
        
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
                    
                    # Prepare data for editable table
                    editable_data = []
                    
                    for i, row in predictions.iterrows():
                        # Ensure date is in YYYY-MM-DD format
                        date_str = ensure_date_format(row['date'])
                            
                        # Store original values for reference
                        row_id = str(row['id'])
                        
                        # Determine result status
                        if pd.notna(row['predicted_outcome']) and pd.notna(row['actual_outcome']) and row['predicted_outcome'] == row['actual_outcome']:
                            result_str = '✅ Won'
                        elif pd.notna(row['actual_outcome']):
                            result_str = '❌ Lost'
                        else:
                            result_str = '⏳ Pending'
                            
                        # Convert confidence to text format
                        confidence_value = float(row['confidence']) if pd.notna(row['confidence']) else 0.0
                        if confidence_value >= 70:
                            confidence_str = 'High'
                        elif confidence_value >= 50:
                            confidence_str = 'Medium'
                        else:
                            confidence_str = 'Low'
                        
                        # Create row data with original values including odds and result
                        row_data = {
                            'id': row_id,
                            'date': date_str,
                            'league': str(row['league']),
                            'home_team': str(row['home_team']),
                            'away_team': str(row['away_team']),
                            'predicted_outcome': str(row['predicted_outcome']),
                            'confidence': confidence_str,  # Display as text (High/Medium/Low)
                            'confidence_value': confidence_value,  # Keep numeric value for calculations
                            'actual_outcome': str(row['actual_outcome']) if pd.notna(row['actual_outcome']) else '',
                            'status': str(row['status']),
                            'result': result_str,  # Add result column
                            'home_odds': float(row['home_odds']) if pd.notna(row['home_odds']) else 0.0,
                            'draw_odds': float(row['draw_odds']) if pd.notna(row['draw_odds']) else 0.0,
                            'away_odds': float(row['away_odds']) if pd.notna(row['away_odds']) else 0.0,
                            'profit_loss': float(row['profit_loss']) if pd.notna(row['profit_loss']) else 0.0,
                            'edit': False,  # Add an edit column for row selection
                            'delete': False,  # Add a delete column for row deletion
                            'apply': False   # Add an apply column to confirm changes
                        }
                        
                        editable_data.append(row_data)
                    
                    # Create dataframe for editing
                    if not editable_data:
                        st.info("No predictions found for the selected filters.")
                        return
                        
                    edit_df = pd.DataFrame(editable_data)
                    
                    # Store the original dataframe in session state for comparison
                    # Make sure we're storing a complete copy with all columns
                    if 'original_edit_df' not in st.session_state:
                        st.session_state.original_edit_df = edit_df.copy(deep=True)
                    else:
                        # Update the original dataframe if columns don't match
                        if set(st.session_state.original_edit_df.columns) != set(edit_df.columns):
                            st.session_state.original_edit_df = edit_df.copy(deep=True)
                    

                    
                    # Ensure all dates are properly formatted as YYYY-MM-DD strings
                    if 'date' in edit_df.columns:
                        # First convert any timestamps to proper date strings
                        edit_df['date'] = edit_df['date'].apply(ensure_date_format)
                        
                        # Force string format to prevent Streamlit from converting to timestamps
                        edit_df['date'] = edit_df['date'].astype(str)
                        
                    # Add instructions for using the editor
                    st.info("📝 To edit a prediction: 1) Click directly on the cell you want to edit, 2) Make your changes, 3) Check 'Apply' to save changes")
                    st.warning("⚠️ To delete a prediction: 1) Check the 'Delete' box for the row, 2) Check 'Apply' to confirm deletion")
                    
                    # Initialize session state for edit tracking
                    if 'edit_state' not in st.session_state:
                        st.session_state.edit_state = {
                            'current_df': edit_df.copy(deep=True),
                            'last_edit_time': time.time(),
                            'edit_in_progress': False
                        }
                    
                    # Store the current dataframe in session state to avoid refreshes
                    if 'original_edit_df' not in st.session_state:
                        st.session_state.original_edit_df = edit_df.copy(deep=True)
                    
                    # Define callbacks to handle edit/delete button clicks without refreshing
                    def handle_edit_click(current_df=None):
                        try:
                            # This function processes the changes in the data editor
                            # If current_df is not provided, try to get it from session state
                            if current_df is None:
                                if 'prediction_editor' not in st.session_state:
                                    logger.warning("prediction_editor not found in session state")
                                    return
                                current_df = st.session_state.prediction_editor
                            
                            # Make sure it's a dataframe
                            if not isinstance(current_df, pd.DataFrame):
                                logger.warning(f"prediction_editor is not a DataFrame: {type(current_df)}")
                                return
                                
                            # Check if required columns exist
                            required_cols = ['delete', 'apply', 'id']
                            for col in required_cols:
                                if col not in current_df.columns:
                                    logger.warning(f"Column '{col}' not found in dataframe")
                                    return
                                    
                            # Store the current dataframe in session state
                            st.session_state.edit_state['current_df'] = current_df.copy(deep=True)
                                    
                            # Find rows with Apply checked - these are the ones we need to process
                            apply_rows = current_df[current_df['apply'] == True]
                            if apply_rows.empty:
                                # No rows with Apply checked, just update the state and return
                                st.session_state.edit_state['current_df'] = current_df.copy(deep=True)
                                return
                                
                            # Log that we're processing rows with Apply checked
                            logger.info(f"Processing {len(apply_rows)} rows with Apply checked")
                            
                            # Force processing of all rows with Apply checked
                            need_refresh = False
                                    
                            # Store the current dataframe in session state for future reference
                            st.session_state.edit_state['current_df'] = current_df.copy(deep=True)
                            
                            # We no longer need to process edit checkboxes as we can edit cells directly
                                    
                            # Process all rows with Apply checked
                            apply_rows = current_df[current_df['apply'] == True]
                            updated_count = 0
                            
                            if not apply_rows.empty:
                                # Log how many rows we're processing
                                logger.info(f"Processing {len(apply_rows)} rows with Apply checked")
                                
                                # Create lists to store success and error messages
                                success_messages = []
                                error_messages = []
                                
                                # Process each row with Apply checked
                                for idx, row in apply_rows.iterrows():
                                    row_id = row['id']
                                    
                                    # Skip rows marked for deletion as they're handled separately
                                    if row['delete'] == True:
                                        continue
                                    
                                    try:
                                        # Log that we're processing this row
                                        logger.info(f"Processing row ID {row_id} with Apply checked")
                                        
                                        # Always update the odds values regardless of whether they've changed
                                        # This ensures that any edits are saved
                                        update_data = {
                                            'home_odds': float(row['home_odds']) if pd.notna(row['home_odds']) else 0.0,
                                            'draw_odds': float(row['draw_odds']) if pd.notna(row['draw_odds']) else 0.0,
                                            'away_odds': float(row['away_odds']) if pd.notna(row['away_odds']) else 0.0
                                        }
                                        
                                        # If profit/loss is editable, include it in the update
                                        if 'profit_loss' in row and pd.notna(row['profit_loss']):
                                            update_data['profit_loss'] = float(row['profit_loss'])
                                        
                                        # Log the update data
                                        logger.info(f"Updating prediction ID: {row_id} with data: {update_data}")
                                        
                                        # Update prediction in database
                                        if history.update_prediction(row_id, update_data):
                                            success_messages.append(f"Updated prediction for {row['home_team']} vs {row['away_team']}")
                                            updated_count += 1
                                            # Clear the apply checkbox after processing
                                            current_df.at[idx, 'apply'] = False
                                            # Update the original dataframe to reflect the changes
                                            if 'original_edit_df' in st.session_state:
                                                st.session_state.original_edit_df.loc[st.session_state.original_edit_df['id'] == row_id, 'home_odds'] = update_data['home_odds']
                                                st.session_state.original_edit_df.loc[st.session_state.original_edit_df['id'] == row_id, 'draw_odds'] = update_data['draw_odds']
                                                st.session_state.original_edit_df.loc[st.session_state.original_edit_df['id'] == row_id, 'away_odds'] = update_data['away_odds']
                                                if 'profit_loss' in update_data:
                                                    st.session_state.original_edit_df.loc[st.session_state.original_edit_df['id'] == row_id, 'profit_loss'] = update_data['profit_loss']
                                        else:
                                            error_messages.append(f"Failed to update prediction ID: {row_id}")
                                    except Exception as row_error:
                                        error_message = f"Error processing row {row_id}: {str(row_error)}"
                                        error_messages.append(error_message)
                                        logger.error(f"Error processing row {row_id}: {str(row_error)}")
                                        continue
                                
                                # Display all success and error messages
                                if updated_count > 0:
                                    st.success(f"Successfully updated {updated_count} predictions")
                                    # Clear history dataframe to force reload
                                    if 'history_df' in st.session_state:
                                        del st.session_state.history_df
                                    # Set flag to refresh the page
                                    need_refresh = True
                                
                                # Display any error messages
                                for error in error_messages:
                                    st.error(error)
                            
                            # Process delete checkboxes - only if Apply is also checked
                            delete_rows = current_df[(current_df['delete'] == True) & (current_df['apply'] == True)]
                            deleted_count = 0
                            
                            if not delete_rows.empty:
                                # Create a list to store success messages to show after all operations
                                success_messages = []
                                error_messages = []
                                
                                for idx, row in delete_rows.iterrows():
                                    row_id = row['id']
                                    # We'll handle the logging in the delete_prediction method
                                    
                                    # Attempt to delete from database
                                    try:
                                        # First check if the row still exists in the database
                                        check = history.db.supabase.table('predictions').select('id').eq('id', row_id).execute()
                                        if not check.data:
                                            # Row already deleted, just update UI
                                            success_messages.append(f"Prediction for {row['home_team']} vs {row['away_team']} already deleted")
                                            deleted_count += 1
                                            # Clear the checkboxes
                                            current_df.at[idx, 'delete'] = False
                                            current_df.at[idx, 'apply'] = False
                                            continue
                                            
                                        # Try to delete the prediction
                                        success = history.delete_prediction(row_id)
                                        if success:
                                            success_messages.append(f"Deleted prediction for {row['home_team']} vs {row['away_team']}")
                                            deleted_count += 1
                                            # Clear the checkboxes to prevent multiple deletions
                                            current_df.at[idx, 'delete'] = False
                                            current_df.at[idx, 'apply'] = False
                                        else:
                                            error_messages.append(f"Failed to delete prediction ID: {row_id}")
                                    except Exception as delete_error:
                                        error_message = f"Error deleting prediction: {str(delete_error)}"
                                        error_messages.append(error_message)
                                        logger.error(f"Delete error: {str(delete_error)}")
                                
                                # Display all success and error messages
                                if deleted_count > 0:
                                    st.success(f"Successfully deleted {deleted_count} predictions")
                                    # Clear session state to force data reload
                                    if 'history_df' in st.session_state:
                                        del st.session_state.history_df
                                    # Set flag to refresh the page
                                    need_refresh = True
                                
                                # Display any error messages
                                for error in error_messages:
                                    st.error(error)
                            
                            # Update the dataframe in session state
                            st.session_state.edit_state['current_df'] = current_df.copy(deep=True)
                            
                            # Refresh the page if needed (after edits or deletions)
                            if need_refresh:
                                # Clear the apply checkboxes before refreshing
                                for idx in current_df.index:
                                    current_df.at[idx, 'apply'] = False
                                st.session_state.edit_state['current_df'] = current_df.copy(deep=True)
                                st.rerun()
                            
                        except Exception as e:
                            # Log the error but don't crash
                            logger.error(f"Error in handle_edit_click: {str(e)}")
                            # Don't re-raise the exception to prevent the app from crashing
                    
                    # Store the previous state of the dataframe for comparison
                    if 'previous_df' not in st.session_state:
                        st.session_state.previous_df = st.session_state.edit_state['current_df'].copy(deep=True)
                    
                    # Define a callback function for when the data editor changes
                    def on_data_editor_change():
                        # Get the current dataframe from session state
                        if 'prediction_editor' in st.session_state:
                            current_df = st.session_state.prediction_editor
                            
                            # Check if any apply checkboxes are checked
                            apply_rows = current_df[current_df['apply'] == True]
                            if not apply_rows.empty:
                                # Process the changes for rows with Apply checked
                                handle_edit_click(current_df)
                    
                    # Use Streamlit's data editor with on_change callback
                    edited_df = st.data_editor(
                        st.session_state.edit_state['current_df'],
                        column_config={
                            "id": st.column_config.TextColumn("ID", disabled=True),
                            "date": st.column_config.TextColumn("Date", disabled=True, help="Format: YYYY-MM-DD"),
                            "league": st.column_config.TextColumn("League", disabled=True),
                            "home_team": st.column_config.TextColumn("Home Team", disabled=True),
                            "away_team": st.column_config.TextColumn("Away Team", disabled=True),
                            "predicted_outcome": st.column_config.SelectboxColumn(
                                "Prediction",
                                options=["HOME", "DRAW", "AWAY"],
                                disabled=True
                            ),
                            "confidence": st.column_config.TextColumn(
                                "Confidence",
                                disabled=True,
                                help="High: ≥70%, Medium: 50-70%, Low: <50%"
                            ),
                            "confidence_value": st.column_config.NumberColumn(
                                "Confidence Value",
                                min_value=0.0,
                                max_value=100.0,
                                format="%.1f%%",
                                disabled=True,
                                help="Numeric confidence value"
                            ),
                            "actual_outcome": st.column_config.SelectboxColumn(
                                "Actual Outcome",
                                options=["", "HOME", "DRAW", "AWAY"],
                                disabled=True
                            ),
                            "status": st.column_config.SelectboxColumn(
                                "Status",
                                options=["Pending", "Completed"],
                                disabled=True
                            ),
                            "result": st.column_config.TextColumn(
                                "Result",
                                disabled=True
                            ),
                            "home_odds": st.column_config.NumberColumn(
                                "Home Odds", 
                                format="%.2f",
                                min_value=1.01,
                                max_value=100.0
                            ),
                            "draw_odds": st.column_config.NumberColumn(
                                "Draw Odds", 
                                format="%.2f",
                                min_value=1.01,
                                max_value=100.0
                            ),
                            "away_odds": st.column_config.NumberColumn(
                                "Away Odds", 
                                format="%.2f",
                                min_value=1.01,
                                max_value=100.0
                            ),
                            "profit_loss": st.column_config.NumberColumn("Profit/Loss", format="%.2fU"),
                            "delete": st.column_config.CheckboxColumn("Delete"),
                            "apply": st.column_config.CheckboxColumn("Apply")
                        },
                        column_order=[
                            "date", "league", "home_team", "away_team", "predicted_outcome", 
                            "confidence", "actual_outcome", "status", "result", 
                            "home_odds", "draw_odds", "away_odds", "profit_loss", "delete", "apply"
                        ],
                        hide_index=True,
                        num_rows="fixed",
                        use_container_width=True,
                        key="prediction_editor",
                        on_change=on_data_editor_change
                    )
                    
                    # Add instructions about using the Apply checkbox
                    st.info("📝 To edit a prediction: 1) Click directly on the cell you want to edit, 2) Make your changes, 3) Check 'Apply' to save changes")
                    st.warning("⚠️ To delete a prediction: 1) Check the 'Delete' box for the row, 2) Check 'Apply' to confirm deletion")
                    
                    # Process edits and deletions
                    if edited_df is not None:
                        # Display delete confirmation if a row was selected for deletion
                        if 'delete_row_data' in st.session_state:
                            row = st.session_state.delete_row_data
                            st.warning(f"To delete prediction for {row['home_team']} vs {row['away_team']}, check the Apply box.")
                            # Clear the delete_row_data after displaying the warning
                            del st.session_state.delete_row_data
                            
                        # Process Apply button actions - SIMPLIFIED APPROACH
                        try:
                            # Find rows with both Delete and Apply checked
                            delete_rows = edited_df[(edited_df['delete'] == True) & (edited_df['apply'] == True)]
                            if not delete_rows.empty:
                                for idx, row in delete_rows.iterrows():
                                    row_id = row['id']
                                    
                                    # This is a confirmed deletion request
                                    logger.info(f"Deleting prediction ID: {row_id}")
                                    
                                    # Attempt to delete from database
                                    try:
                                        success = history.delete_prediction(row_id)
                                        if success:
                                            st.success(f"Deleted prediction for {row['home_team']} vs {row['away_team']}")
                                            # Clear session state to force data reload
                                            if 'history_df' in st.session_state:
                                                del st.session_state.history_df
                                            if 'edit_state' in st.session_state:
                                                del st.session_state.edit_state
                                            if 'original_edit_df' in st.session_state:
                                                del st.session_state.original_edit_df
                                            
                                            # Force page refresh using the correct method
                                            st.rerun()
                                        else:
                                            st.error(f"Failed to delete prediction ID: {row_id}")
                                    except Exception as delete_error:
                                        st.error(f"Error deleting prediction: {str(delete_error)}")
                                        logger.error(f"Delete error: {str(delete_error)}")
                            
                            # For rows with just Apply checked (but not Delete)
                            apply_only_rows = edited_df[(edited_df['apply'] == True) & (edited_df['delete'] == False)]
                            if not apply_only_rows.empty:
                                for idx, row in apply_only_rows.iterrows():
                                    # Clear the apply checkbox after processing
                                    edited_df.at[idx, 'apply'] = False
                        except Exception as e:
                            logger.error(f"Error processing Apply actions: {str(e)}")
                            st.error(f"An error occurred while processing your request. Please try again.")
                            # Don't crash the app
                        
                        # Check for edit requests with Apply checked
                        try:
                            # Check if we have an edit request pending
                            if 'edit_row_id' in st.session_state and st.session_state.edit_row_id:
                                # Find the row with the matching ID and Apply checked
                                apply_rows = edited_df[edited_df['apply'] == True]
                                if not apply_rows.empty:
                                    for idx, row in apply_rows.iterrows():
                                        if str(row['id']) == str(st.session_state.edit_row_id):
                                            row_id = row['id']
                                            logger.info(f"Processing edit for prediction ID: {row_id}")
                                            
                                            # Store edit data in session state for the form
                                            st.session_state.edit_prediction = True
                                            st.session_state.edit_prediction_id = row_id
                                            st.session_state.edit_prediction_data = row.to_dict()
                                            
                                            # Show success message
                                            st.success(f"Editing prediction for {row['home_team']} vs {row['away_team']}")
                                            
                                            # Clear the checkboxes after processing
                                            edited_df.at[idx, 'apply'] = False
                                            edited_df.at[idx, 'edit'] = False
                                            st.session_state.edit_state['current_df'] = edited_df.copy(deep=True)
                                            
                                            # Clear the edit row ID
                                            del st.session_state.edit_row_id
                                            break
                        except Exception as e:
                            logger.error(f"Error processing edit request: {str(e)}")
                            # Don't crash the app
                            
                        # Process direct edits (changes to odds or profit/loss) with Apply checked
                        # Only process rows that have Apply checked but not marked for delete or edit
                        direct_edit_rows = edited_df[(edited_df['apply'] == True) & 
                                                    (edited_df['delete'] == False) & 
                                                    (edited_df['edit'] == False)]
                        
                        if not direct_edit_rows.empty and 'original_edit_df' in st.session_state:
                            original_df = st.session_state.original_edit_df
                            
                            # Compare each row with Apply checked for changes
                            for i, row in direct_edit_rows.iterrows():
                                row_id = row['id']
                                
                                try:
                                    # Find the original row data
                                    original_rows = original_df[original_df['id'] == row_id]
                                    if len(original_rows) == 0:
                                        logger.warning(f"Original row not found for ID: {row_id}")
                                        continue
                                        
                                    original_row = original_rows.iloc[0]
                                    
                                    # Compare with original to detect changes
                                    # Ensure we're comparing the same types
                                    home_odds_changed = float(row['home_odds']) != float(original_row['home_odds'])
                                    draw_odds_changed = float(row['draw_odds']) != float(original_row['draw_odds'])
                                    away_odds_changed = float(row['away_odds']) != float(original_row['away_odds'])
                                    profit_loss_changed = float(row['profit_loss']) != float(original_row['profit_loss'])
                                    
                                    # Only process changes if Apply is checked and changes detected
                                    if (home_odds_changed or draw_odds_changed or away_odds_changed or profit_loss_changed):
                                        # Get original data for odds values
                                        original_data = predictions.loc[predictions['id'] == int(row_id)].iloc[0] if not predictions.empty else None
                                        
                                        if original_data is None:
                                            st.error(f"Could not find original data for prediction ID {row_id}")
                                            continue
                                        
                                        # Prepare update data - use edited odds values directly
                                        # Use confidence_value instead of confidence (which is now a text field)
                                        update_data = {
                                            'date': ensure_date_format(row['date']),  # Use our consistent date formatter
                                            'league': row['league'],
                                            'home_team': row['home_team'],
                                            'away_team': row['away_team'],
                                            'predicted_outcome': row['predicted_outcome'],
                                            'confidence': float(row['confidence_value']) if 'confidence_value' in row else float(original_row['confidence']),
                                            'home_odds': float(row['home_odds']) if pd.notna(row['home_odds']) else 0.0,
                                            'draw_odds': float(row['draw_odds']) if pd.notna(row['draw_odds']) else 0.0,
                                            'away_odds': float(row['away_odds']) if pd.notna(row['away_odds']) else 0.0,
                                            'status': row['status']
                                        }
                                        
                                        # Add completed match details if status is Completed
                                        if row['status'] == 'Completed':
                                            # Get scores from original data
                                            if original_data is not None:
                                                home_score = int(original_data['home_score']) if pd.notna(original_data['home_score']) else 0
                                                away_score = int(original_data['away_score']) if pd.notna(original_data['away_score']) else 0
                                            else:
                                                # Fallback if we can't find original data
                                                home_score = 0
                                                away_score = 0
                                            
                                            # Determine actual outcome based on scores
                                            if home_score > away_score:
                                                actual_outcome = "HOME"
                                            elif away_score > home_score:
                                                actual_outcome = "AWAY"
                                            else:
                                                actual_outcome = "DRAW"
                                                
                                            # Calculate profit/loss
                                            if actual_outcome == row['predicted_outcome']:
                                                # Win: (odds * bet_amount) - bet_amount
                                                if actual_outcome == "HOME":
                                                    profit_loss = (float(row['home_odds']) - 1.0) * 1.0  # $1 bet
                                                elif actual_outcome == "AWAY":
                                                    profit_loss = (float(row['away_odds']) - 1.0) * 1.0
                                                else:  # DRAW
                                                    profit_loss = (float(row['draw_odds']) - 1.0) * 1.0
                                            else:
                                                # Loss: -bet_amount
                                                profit_loss = -1.0  # $1 bet
                                                
                                            # Add match details to update data
                                            update_data.update({
                                                'home_score': home_score,
                                                'away_score': away_score,
                                                'actual_outcome': actual_outcome,
                                                'profit_loss': profit_loss
                                            })
                                        
                                        # Update prediction in database with confirmation
                                        logger.info(f"Updating prediction ID: {row_id} with data: {update_data}")
                                        if history.update_prediction(row_id, update_data):
                                            st.toast(f"Updated prediction for {row['home_team']} vs {row['away_team']}")
                                            # Update session state
                                            st.session_state.edit_state['current_df'] = edited_df.copy(deep=True)
                                            st.session_state.original_edit_df = edited_df.copy(deep=True)
                                            # Clear history dataframe to force reload
                                            st.session_state.pop('history_df', None)
                                            # Clear the apply checkbox after processing
                                            edited_df.at[i, 'apply'] = False
                                except (IndexError, KeyError) as e:
                                    logger.error(f"Error processing row: {e}")
                                    continue
                    

                    # Create edit expander instead of dialog
                    if st.session_state.get('edit_prediction', False):
                        edit_data = st.session_state.edit_prediction_data
                        
                        with st.expander(f"Edit: {edit_data.get('home_team')} vs {edit_data.get('away_team')}", expanded=True):
                            # Create form for editing
                            with st.form("edit_prediction_form"):
                                st.subheader(f"Edit Prediction Details")
                                st.caption("Make your changes and click 'Save Changes' when done.")
                                
                                # Match details
                                col1, col2 = st.columns(2)
                                with col1:
                                    date = st.date_input("Date", value=pd.to_datetime(edit_data.get('date')).date())
                                    league = st.text_input("League", value=edit_data.get('league', ''))
                                    home_team = st.text_input("Home Team", value=edit_data.get('home_team', ''))
                                    away_team = st.text_input("Away Team", value=edit_data.get('away_team', ''))
                                
                                with col2:
                                    predicted_outcome = st.selectbox(
                                        "Predicted Outcome", 
                                        options=["HOME", "DRAW", "AWAY"],
                                        index=["HOME", "DRAW", "AWAY"].index(edit_data.get('predicted_outcome', 'HOME'))
                                    )
                                    
                                    confidence = st.number_input(
                                        "Confidence (%)", 
                                        min_value=0.0, 
                                        max_value=100.0, 
                                        value=float(edit_data.get('confidence', 50.0))
                                    )
                                    
                                    status = st.selectbox(
                                        "Status", 
                                        options=["Pending", "Completed"],
                                        index=0 if edit_data.get('status') == "Pending" else 1
                                    )
                                
                                # Odds
                                st.subheader("Odds")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    home_odds = st.number_input(
                                        "Home Odds", 
                                        min_value=1.0, 
                                        value=float(edit_data.get('home_odds', 1.5))
                                    )
                                with col2:
                                    draw_odds = st.number_input(
                                        "Draw Odds", 
                                        min_value=1.0, 
                                        value=float(edit_data.get('draw_odds', 3.5))
                                    )
                                with col3:
                                    away_odds = st.number_input(
                                        "Away Odds", 
                                        min_value=1.0, 
                                        value=float(edit_data.get('away_odds', 5.0))
                                    )
                                
                                # Result (only if status is Completed)
                                if status == "Completed":
                                    st.subheader("Match Result")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        home_score = st.number_input(
                                            "Home Score", 
                                            min_value=0, 
                                            value=int(edit_data.get('home_score', 0) or 0)
                                        )
                                    with col2:
                                        away_score = st.number_input(
                                            "Away Score", 
                                            min_value=0, 
                                            value=int(edit_data.get('away_score', 0) or 0)
                                        )
                                    
                                    # Determine actual outcome based on scores
                                    if home_score > away_score:
                                        actual_outcome = "HOME"
                                    elif away_score > home_score:
                                        actual_outcome = "AWAY"
                                    else:
                                        actual_outcome = "DRAW"
                                    
                                    # Calculate profit/loss
                                    bet_amount = 1.0  # Fixed $1 bet
                                    if predicted_outcome == actual_outcome:
                                        if predicted_outcome == "HOME":
                                            profit_loss = float(round((home_odds * bet_amount) - bet_amount, 2))
                                        elif predicted_outcome == "AWAY":
                                            profit_loss = float(round((away_odds * bet_amount) - bet_amount, 2))
                                        else:  # DRAW
                                            profit_loss = float(round((draw_odds * bet_amount) - bet_amount, 2))
                                    else:
                                        profit_loss = float(-bet_amount)
                                else:
                                    home_score = None
                                    away_score = None
                                    actual_outcome = None
                                    profit_loss = 0.0
                                
                                # Submit and Delete buttons
                                col1, col2 = st.columns(2)
                                submitted = col1.form_submit_button("Save Changes")
                                deleted = col2.form_submit_button("Delete Prediction")
                                
                                if submitted:
                                    # Prepare update data
                                    update_data = {
                                        'date': date.strftime("%Y-%m-%d"),
                                        'league': league,
                                        'home_team': home_team,
                                        'away_team': away_team,
                                        'predicted_outcome': predicted_outcome,
                                        'confidence': confidence,
                                        'home_odds': home_odds,
                                        'draw_odds': draw_odds,
                                        'away_odds': away_odds,
                                        'status': status
                                    }
                                    
                                    if status == "Completed":
                                        update_data.update({
                                            'home_score': home_score,
                                            'away_score': away_score,
                                            'actual_outcome': actual_outcome,
                                            'profit_loss': profit_loss
                                        })
                                    
                                    # Update prediction in database
                                    if history.update_prediction(st.session_state.edit_prediction_id, update_data):
                                        st.success("Prediction updated successfully!")
                                        # Clear session state and refresh
                                        st.session_state.pop('edit_prediction', None)
                                        st.session_state.pop('edit_prediction_id', None)
                                        st.session_state.pop('edit_prediction_data', None)
                                        st.session_state.pop('history_df', None)
                                        st.rerun()
                                    else:
                                        st.error("Failed to update prediction.")
                                
                                elif deleted:
                                    # Delete prediction from database
                                    if history.delete_prediction(st.session_state.edit_prediction_id):
                                        st.success("Prediction deleted successfully!")
                                        # Clear session state and refresh
                                        st.session_state.pop('edit_prediction', None)
                                        st.session_state.pop('edit_prediction_id', None)
                                        st.session_state.pop('edit_prediction_data', None)
                                        st.session_state.pop('history_df', None)
                                        st.rerun()
                                    else:
                                        st.error("Failed to delete prediction.")
                            
                            # Cancel button outside the form
                            if st.button("Cancel"):
                                st.session_state.pop('edit_prediction', None)
                                st.session_state.pop('edit_prediction_id', None)
                                st.session_state.pop('edit_prediction_data', None)
                                st.rerun()
                    
                except Exception as e:
                    st.error(f"Error displaying predictions table: {str(e)}")
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


