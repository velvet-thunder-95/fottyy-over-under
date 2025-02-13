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
            # Start building the query
            query = self.db.supabase.table('predictions').select('*')
            
            # Apply filters
            if start_date:
                query = query.gte('date', start_date)
            if end_date:
                query = query.lte('date', end_date)
            if status:
                query = query.eq('status', status)
                
            # Handle confidence levels
            if confidence_levels and "All" not in confidence_levels:
                for level in confidence_levels:
                    if level == "High":
                        query = query.filter('confidence', 'gte', 70)
                    elif level == "Medium":
                        query = query.filter('confidence', 'gte', 50).filter('confidence', 'lt', 70)
                    elif level == "Low":
                        query = query.filter('confidence', 'lt', 50)
                        
            # Handle leagues
            if leagues and "All" not in leagues:
                query = query.in_('league', leagues)
                
            # Execute query and order by date
            result = query.order('date.desc').execute()
            
            # Convert to DataFrame
            df = pd.DataFrame(result.data)
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
                for level in confidence_levels:
                    if level == "High":
                        query = query.gte('confidence', 70)
                    elif level == "Medium":
                        query = query.and_(f"confidence.gte.50,confidence.lt.70")
                    elif level == "Low":
                        query = query.lt('confidence', 50)
                        
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
                'scores': f"{home_score}-{away_score}" if home_score is not None and away_score is not None else None,
                'actual_outcome': actual_outcome,
                'profit_loss': profit_loss
            }
            
            # Debug print before update
            print(f"Updating match {match_id} with data: {update_data}")
            
            try:
                self.db.supabase.table('predictions')\
                    .update(update_data)\
                    .eq('match_id', match_id)\
                    .execute()
                print(f"Successfully updated match {match_id}")
            except Exception as e:
                print(f"Error updating match {match_id}: {str(e)}")
            
        except Exception as e:
            print(f"Error processing match {match_id}: {str(e)}")

    def update_match_results_all(self):
        """Update completed match results using match_analyzer"""
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        analyzer = MatchAnalyzer("633379bdd5c4c3eb26919d8570866801e1c07f399197ba8c5311446b8ea77a49")
        
        # Get completed predictions from Supabase that need profit/loss recalculation
        result = self.db.supabase.table('predictions') \
            .select('id,match_id,home_team,away_team,date,predicted_outcome,home_odds,draw_odds,away_odds,actual_outcome') \
            .filter('status', 'eq', 'Completed') \
            .filter('match_id', 'neq', None) \
            .execute()
        completed_predictions = result.data
        logger.info(f"Found {len(completed_predictions)} completed predictions")
        
        for pred in completed_predictions:
            try:
                match_id = pred['match_id']
                home_team = pred['home_team']
                away_team = pred['away_team']
                match_date = pred['date']
                
                if not match_id:
                    logger.warning(f"Missing match_id for {home_team} vs {away_team} on {match_date}")
                    continue
                    
                # Get match result
                result = analyzer.analyze_match_result(match_id)
                if not result:
                    logger.info(f"Match not complete: {home_team} vs {away_team}")
                    continue
                    
                # Update the result
                self.update_match_results(match_id, result)
                logger.info(f"Updated result for {home_team} vs {away_team}")
                
            except Exception as e:
                logger.error(f"Error processing match: {str(e)}")
                continue



def style_dataframe(df):
    """Style the predictions dataframe with colors and formatting"""
    
    def format_result(row):
        if row['Status'] == 'SCHEDULED':
            return '🗓️ Scheduled'
        elif row['Status'] == 'Pending':
            return '⏳ Pending'
        elif pd.isna(row['Actual Outcome']) or row['Actual Outcome'] == '':
            return '⏳ Pending'
        elif row['Prediction'] == row['Actual Outcome']:
            return '✅ Won'
        else:
            return '❌ Lost'
            
    def format_actual_outcome(row):
        if row['Status'] in ['SCHEDULED', 'Pending'] or pd.isna(row['Actual Outcome']) or row['Actual Outcome'] == '':
            return '-'  # Show dash for scheduled/pending matches
        return row['Actual Outcome']

    def format_profit_loss(row):
        if row['Status'] in ['SCHEDULED', 'Pending']:
            return '-'
        try:
            # Use the original profit_loss value before formatting
            profit = row.get('profit_loss', 0)
            if pd.isna(profit):
                return '-'
            profit = float(profit)
            if profit > 0:
                return f'+£{profit:.2f}'
            elif profit < 0:
                return f'-£{abs(profit):.2f}'
            return '£0.00'
        except (ValueError, TypeError):
            return '-'

    # Create a copy to avoid modifying the original
    display_df = df.copy()
    
    # Format the Result column
    display_df['Result'] = display_df.apply(format_result, axis=1)
    
    # Format the Actual Outcome column
    display_df['Actual Outcome'] = display_df.apply(format_actual_outcome, axis=1)
    
    # Format Profit/Loss
    display_df['Profit/Loss'] = display_df.apply(format_profit_loss, axis=1)
    
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
                    
                    # Create final dataframe
                    final_df = predictions[list(display_columns.keys())].copy()
                    
                    # Debug print raw values
                    print("\nRaw profit/loss values from database:")
                    print(predictions[['profit_loss', 'status']].to_string())
                    
                    # Convert profit/loss column to float, replacing NaN with 0.0
                    final_df['profit_loss'] = pd.to_numeric(predictions['profit_loss'], errors='coerce').fillna(0.0)
                    
                    # Set profit/loss to 0 for non-completed matches
                    final_df.loc[final_df['status'] != 'Completed', 'profit_loss'] = 0.0
                    
                    # Debug print after conversion
                    print("\nProfit/loss values after conversion:")
                    print(final_df[['profit_loss']].to_string())
                    
                    # Format profit/loss values with proper currency symbol
                    final_df['profit_loss'] = final_df['profit_loss'].apply(
                        lambda x: f'+£{x:.2f}' if x > 0 else f'-£{abs(x):.2f}' if x < 0 else '£0.00'
                    )
                    
                    # Print formatted values
                    print("\nFormatted profit/loss values:")
                    print(final_df[['profit_loss']].to_string())
                    
                    # Rename columns
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
