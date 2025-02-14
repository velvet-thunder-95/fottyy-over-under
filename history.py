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

    def get_predictions(self, start_date=None, end_date=None, status=None, confidence_levels=None, leagues=None):
        """Get predictions with optional filters"""
        try:
            # Start building the query with all columns
            columns = ['date', 'league', 'home_team', 'away_team', 'predicted_outcome', 
                      'actual_outcome', 'home_odds', 'draw_odds', 'away_odds', 'confidence', 
                      'bet_amount', 'profit_loss', 'status', 'match_id', 'home_score', 'away_score']
            query = self.db.supabase.table('predictions').select(','.join(columns))
            
            # Apply filters
            if start_date:
                query = query.gte('date', start_date)
            if end_date:
                query = query.lte('date', end_date)
            if status:
                query = query.eq('status', status)
                
            # Handle confidence levels
            if confidence_levels and "All" not in confidence_levels:
                # Start with an empty list for our predictions
                predictions_list = []
                
                # Get predictions for each confidence level
                if "Medium" in confidence_levels:
                    medium_query = self.db.supabase.table('predictions').select('*')\
                        .gte('confidence', 50).lt('confidence', 70)
                    medium_result = medium_query.execute()
                    predictions_list.extend(medium_result.data)
                    
                if "High" in confidence_levels:
                    high_query = self.db.supabase.table('predictions').select('*')\
                        .gte('confidence', 70)
                    high_result = high_query.execute()
                    predictions_list.extend(high_result.data)
                    
                if "Low" in confidence_levels:
                    low_query = self.db.supabase.table('predictions').select('*')\
                        .lt('confidence', 50)
                    low_result = low_query.execute()
                    predictions_list.extend(low_result.data)
                
                # Convert to DataFrame
                return pd.DataFrame(predictions_list)
            
            # If no confidence levels selected or 'All' is selected, continue with original query
            result = query.execute()
            return pd.DataFrame(result.data)
                        
            # Handle leagues
            if leagues and "All" not in leagues:
                query = query.in_('league', leagues)
                
            # Execute query and order by date
            result = query.order('date.desc').execute()
            
            # Convert to DataFrame
            df = pd.DataFrame(result.data)
            if not df.empty:
                # Ensure profit_loss is numeric and has proper default values
                df['profit_loss'] = pd.to_numeric(df['profit_loss'], errors='coerce').fillna(0.0)
                
                # Convert other numeric columns
                numeric_columns = ['bet_amount', 'confidence', 'home_odds', 'draw_odds', 'away_odds']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Ensure proper profit_loss values based on status
                df.loc[df['status'] != 'Completed', 'profit_loss'] = 0.0
                
                # Debug info
                print("\nDataFrame info after conversion:")
                print(df.info())
                print("\nProfit/Loss values:")
                print(df[['profit_loss', 'status', 'predicted_outcome', 'actual_outcome']].head())
            
            return df
            
        except Exception as e:
            logging.error(f"Error getting predictions: {str(e)}")
            return pd.DataFrame()

    def calculate_statistics(self, confidence_levels=None, leagues=None):
        """Calculate prediction statistics with optional confidence level and league filters"""
        try:
            # Get predictions from Supabase with filters
            query = self.db.supabase.table('predictions').select('*')
            
            # Apply confidence level filters
            if confidence_levels and "All" not in confidence_levels:
                # Start with an empty list for our predictions
                predictions_list = []
                
                # Get predictions for each confidence level
                if "Medium" in confidence_levels:
                    medium_query = self.db.supabase.table('predictions').select('*')\
                        .gte('confidence', 50).lt('confidence', 70)
                    medium_result = medium_query.execute()
                    predictions_list.extend(medium_result.data)
                    
                if "High" in confidence_levels:
                    high_query = self.db.supabase.table('predictions').select('*')\
                        .gte('confidence', 70)
                    high_result = high_query.execute()
                    predictions_list.extend(high_result.data)
                    
                if "Low" in confidence_levels:
                    low_query = self.db.supabase.table('predictions').select('*')\
                        .lt('confidence', 50)
                    low_result = low_query.execute()
                    predictions_list.extend(low_result.data)
                
                # Convert to DataFrame and calculate statistics
                predictions = pd.DataFrame(predictions_list)
            else:
                # If no confidence levels selected or 'All' is selected, continue with original query
                result = query.execute()
                predictions = pd.DataFrame(result.data)
            
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
            
            # Calculate total profit/loss
            total_profit = completed_predictions['profit_loss'].sum()
            
            # Calculate ROI
            total_bet_amount = completed_predictions['bet_amount'].sum()
            roi = (total_profit / total_bet_amount * 100) if total_bet_amount > 0 else 0.0
            
            return [total_predictions, correct_predictions, success_rate, total_profit, roi], pending_count
                        
            # Apply league filters
            if leagues and "All" not in leagues:
                query = query.in_('league', leagues)
                
            # Execute query
            result = query.execute()
            predictions = pd.DataFrame(result.data)
            
            if predictions.empty:
                return [0, 0, 0.0, 0.0, 0.0], 0
            
            # Calculate statistics
            completed_predictions = predictions[predictions['status'] == 'Completed']
            pending_predictions = predictions[predictions['status'] == 'Pending']
            
            total_predictions = len(predictions)
            completed_count = len(completed_predictions)
            pending_count = len(pending_predictions)
            
            # Calculate correct predictions
            correct_predictions = len(
                completed_predictions[
                    completed_predictions['predicted_outcome'] == 
                    completed_predictions['actual_outcome']
                ]
            )
            
            # Calculate success rate
            success_rate = (correct_predictions / completed_count * 100) \
                if completed_count > 0 else 0.0
            
            # Calculate profit and ROI
            total_profit = completed_predictions['profit_loss'].sum() \
                if not completed_predictions.empty else 0.0
            
            total_bet_amount = completed_count * float(completed_predictions['bet_amount'].iloc[0]) \
                if not completed_predictions.empty else 0.0
            
            roi = (total_profit / total_bet_amount * 100) \
                if total_bet_amount > 0 else 0.0
            
            return [
                total_predictions,
                correct_predictions,
                success_rate,
                total_profit,
                roi
            ], pending_count
            
        except Exception as e:
            logging.error(f"Error calculating statistics: {str(e)}")
            return [0, 0, 0.0, 0.0, 0.0], 0

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



def style_dataframe(df):
    """Style the predictions dataframe with colors and formatting"""
    
    def style_row(row):
        base_style = [
            'font-size: 14px',
            'font-weight: 400',
            'color: #333333',
            'padding: 12px 15px',
            'border-bottom: 1px solid #e0e0e0'
        ]
        
        styles = [base_style.copy() for _ in range(len(row))]
        
        # Style row background based on result
        bg_style = None
        if row['Status'] == 'Pending' or pd.isna(row.get('Actual Outcome', '')) or row.get('Actual Outcome', '') == '':
            bg_style = 'background-color: #f5f5f5'
        elif row.get('Prediction', '') == row.get('Actual Outcome', ''):
            bg_style = 'background-color: #e8f5e9'  # Lighter green
        else:
            bg_style = 'background-color: #fce4ec'  # Lighter red
        
        if bg_style:
            for style in styles:
                style.append(bg_style)
        
        return [';'.join(style) for style in styles]
    
    def style_profit_loss(val):
        if not isinstance(val, str) or val == '-':
            return ''
        if val.startswith('+'):
            return 'color: #4CAF50; font-weight: 600'
        elif val.startswith('-'):
            return 'color: #f44336; font-weight: 600'
        elif val == '£0.00':
            return 'color: #6c757d'
        return ''
    
    def style_confidence(val):
        if 'High' in str(val):
            return 'background-color: #d4edda; color: #155724; font-weight: bold'
        elif 'Medium' in str(val):
            return 'background-color: #fff3cd; color: #856404; font-weight: bold'
        elif 'Low' in str(val):
            return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
        return ''
    
    def style_result(val):
        if '✅' in str(val):  # Won
            return 'color: #28a745; font-weight: bold'
        elif '❌' in str(val):  # Lost
            return 'color: #dc3545; font-weight: bold'
        return ''
    
    # Create the styled DataFrame
    styled = df.style\
        .apply(style_row, axis=1)\
        .map(style_profit_loss, subset=['Profit/Loss'])\
        .map(style_confidence, subset=['Confidence'])\
        .map(style_result, subset=['Result'])
    
    return styled
    
    # Apply styling
    def style_row(row):
        base_style = [
            'font-size: 14px',
            'font-weight: 400',
            'color: #333333',
            'padding: 12px 15px',
            'border-bottom: 1px solid #e0e0e0'
        ]
        
        styles = [base_style.copy() for _ in range(len(row))]
        
        # Get column indices
        profit_loss_idx = None
        try:
            profit_loss_idx = display_df.columns.get_loc('Profit/Loss')
        except KeyError:
            pass
        
        # Style row background
        bg_style = None
        if row['Status'] == 'Pending' or pd.isna(row['Actual Outcome']) or row['Actual Outcome'] == '':
            bg_style = 'background-color: #f5f5f5'
        elif row['Prediction'] == row['Actual Outcome']:
            bg_style = 'background-color: #e8f5e9'  # Lighter green
        else:
            bg_style = 'background-color: #fce4ec'  # Lighter red
        
        if bg_style:
            for style in styles:
                style.append(bg_style)
        
        return [';'.join(style) for style in styles]
    
    # Function to style profit/loss values
    def style_profit_loss(val):
        if not isinstance(val, str):
            return ''
        if val.startswith('+'):
            return 'color: #4CAF50; font-weight: 600'
        elif val.startswith('-'):
            return 'color: #f44336; font-weight: 600'
        return ''
    
    # Create the styled DataFrame
    styled_df = display_df.style\
        .apply(style_row, axis=1)\
        .map(style_profit_loss, subset=['Profit/Loss'])
    
    # Add table styles
    styled_df.set_table_styles([
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
    
    return styled_df

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
            padding: 12px 15px !important;
            color: #333333 !important;
        }
        .stDataFrame [data-testid="StyledDataFrameHeaderCell"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            padding: 12px 15px !important;
            background-color: #f8f9fa !important;
            color: #333333 !important;
            font-weight: 600 !important;
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

        @media (max-width: 768px) {
            .metrics-grid {
                flex-wrap: wrap;
            }
            .metric-box {
                min-width: calc(50% - 10px);
            }
        }
        
        /* Table styling */
        .dataframe {
            border-collapse: separate !important;
            border-spacing: 0;
            width: 100%;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }
        
        .dataframe th {
            background-color: #f8f9fa !important;
            color: #495057 !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            padding: 15px !important;
            border-bottom: 2px solid #dee2e6 !important;
        }
        
        .dataframe td {
            padding: 12px 15px !important;
            border-bottom: 1px solid #e9ecef !important;
            color: #212529 !important;
            background-color: white !important;
        }
        
        .dataframe tr:hover td {
            background-color: #f8f9fa !important;
        }
        
        /* Status indicators */
        .status-pending {
            color: #6c757d;
            background-color: #f8f9fa;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: 500;
        }
        
        .status-win {
            color: #28a745;
            background-color: #e8f5e9;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: 500;
        }
        
        .status-loss {
            color: #dc3545;
            background-color: #fbe9e7;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: 500;
        }
        
        /* Date filter styling */
        .date-filter {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            margin-bottom: 20px;
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #555;
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
        # Initialize PredictionHistory
        history = PredictionHistory()
        
        # Add date filter in sidebar with custom styling
        st.sidebar.markdown("""
            <div class="date-filter">
                <h2 style='color: #1e3c72; font-size: 1.2em; margin-bottom: 15px;'>Filters</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Add filters to sidebar
        st.sidebar.markdown("## Filters", help="Filter your prediction history")
        
        # Date filters
        start_date = st.sidebar.date_input(
            "Start Date",
            value=datetime.now().date() - timedelta(days=30),
            help="Filter predictions from this date"
        )
        
        end_date = st.sidebar.date_input(
            "End Date",
            value=datetime.now().date(),
            help="Filter predictions until this date"
        )

        # Validate dates
        if start_date > end_date:
            st.sidebar.error("Error: End date must be after start date")
            start_date, end_date = end_date, start_date  # Swap dates to ensure valid range

        # Format dates for database query
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")  # Add 1 day to include end date
        
        # Get unique leagues from predictions
        all_predictions = history.get_predictions()
        available_leagues = ["All"] + sorted(all_predictions['league'].unique().tolist()) if not all_predictions.empty else ["All"]
        
        # League multiselect
        selected_leagues = st.sidebar.multiselect(
            "Select Competitions",
            options=available_leagues,
            default=["All"],
            help="Filter predictions by competition. Select multiple competitions or 'All'"
        )
        
        if not selected_leagues:
            selected_leagues = ["All"]
        
        # Confidence level multiselect
        confidence_levels = st.sidebar.multiselect(
            "Confidence Levels",
            options=["All", "High", "Medium", "Low"],
            default=["All"],
            help="Filter predictions by confidence level: High (≥70%), Medium (50-69%), Low (<50%). Select multiple levels or 'All'"
        )
        
        if not confidence_levels:
            confidence_levels = ["All"]
        
        # Get filtered predictions
        predictions = history.get_predictions(
            start_date=start_date_str,
            end_date=end_date_str,
            confidence_levels=None if "All" in confidence_levels else confidence_levels,
            leagues=None if "All" in selected_leagues else selected_leagues
        )
        
        if not predictions.empty:
            # Update any pending predictions
            history.update_match_results_all()
            
            # Refresh predictions after update
            predictions = history.get_predictions(
                start_date=start_date_str,
                end_date=end_date_str,
                confidence_levels=None if "All" in confidence_levels else confidence_levels,
                leagues=None if "All" in selected_leagues else selected_leagues
            )
            
            # Calculate statistics
            current_confidence = None if "All" in confidence_levels else confidence_levels
            current_leagues = None if "All" in selected_leagues else selected_leagues
            stats, pending_count = history.calculate_statistics(
                confidence_levels=current_confidence,
                leagues=current_leagues
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
                    formatted_value = f"£{metric['value']:.2f}"
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
                        'profit_loss': 'Profit/Loss',  # Keep original column name
                        'status': 'Status'
                    }
                    
                    # Create final dataframe with explicit column selection
                    final_df = predictions[list(display_columns.keys())].copy()
                    
                    # Debug print raw values
                    print("\nRaw profit/loss values from database:")
                    print(predictions[['profit_loss', 'status', 'predicted_outcome', 'actual_outcome']].to_string())
                    
                    # Format Result column
                    def format_result(row):
                        if row['status'] == 'SCHEDULED':
                            return '🗓️ Scheduled'
                        elif row['status'] == 'Pending':
                            return '⏳ Pending'
                        elif pd.isna(row['actual_outcome']) or row['actual_outcome'] == '':
                            return '⏳ Pending'
                        elif row['predicted_outcome'] == row['actual_outcome']:
                            return '✅ Won'
                        else:
                            return '❌ Lost'
                    
                    # Format Actual Outcome column
                    def format_actual_outcome(row):
                        if row['status'] in ['SCHEDULED', 'Pending'] or pd.isna(row['actual_outcome']) or row['actual_outcome'] == '':
                            return '-'
                        return row['actual_outcome']
                    
                    # Format profit/loss values
                    def format_pl(row):
                        if row['status'] != 'Completed':
                            return '-'
                        try:
                            value = float(row['profit_loss'])
                            if pd.isna(value):
                                return '£0.00'
                            return f'+£{value:.2f}' if value > 0 else f'-£{abs(value):.2f}' if value < 0 else '£0.00'
                        except (ValueError, TypeError):
                            return '£0.00'
                    
                    # Apply all formatting
                    final_df['Result'] = final_df.apply(format_result, axis=1)
                    final_df['actual_outcome'] = final_df.apply(format_actual_outcome, axis=1)
                    final_df['profit_loss'] = final_df.apply(format_pl, axis=1)
                    
                    # Print debug info
                    print("\nFormatted profit/loss values:")
                    print(final_df[['profit_loss', 'status']].head().to_string())
                    
                    # Rename columns for display
                    final_df = final_df.rename(columns=display_columns)
                    
                    # Apply styling
                    styled_df = style_dataframe(final_df)
                    
                    # Display the styled dataframe
                    st.dataframe(
                        styled_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                except Exception as e:
                    st.error(f"Error displaying predictions table: {str(e)}")
                    st.exception(e)
            else:
                st.info("No predictions found for the selected date range.")
        
    except Exception as e:
        st.error(f"Error displaying predictions table: {str(e)}")
        st.exception(e)

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

    # Add navigation buttons
    def add_navigation_buttons():
        col1, col2, col3 = st.columns([2, 2, 2])
        
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

    # Call the function to add navigation buttons
    add_navigation_buttons()

    # Add back button
    if st.button("Back to Predictions"):
        st.query_params["page"] = "main"
        st.rerun()
