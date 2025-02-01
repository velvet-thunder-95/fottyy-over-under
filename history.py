# history.py

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
from football_api import get_match_by_teams, get_match_result
from session_state import init_session_state, check_login_state
from match_analyzer import MatchAnalyzer
import logging

class PredictionHistory:
    def __init__(self):
        self.db_path = 'predictions.db'
        self.init_database()

    def init_database(self):
        """Initialize the SQLite database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create predictions table with all necessary fields
        c.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                league TEXT,
                home_team TEXT,
                away_team TEXT,
                predicted_outcome TEXT,
                actual_outcome TEXT,
                home_odds REAL,
                draw_odds REAL,
                away_odds REAL,
                confidence REAL,
                bet_amount REAL,
                profit_loss REAL,
                prediction_type TEXT,
                status TEXT,
                match_date TEXT,
                match_id TEXT
            )
        ''')
        
        # Convert existing confidence values to REAL if needed
        try:
            c.execute('SELECT confidence FROM predictions LIMIT 1')
            sample = c.fetchone()
            if sample and isinstance(sample[0], bytes):
                # Create a temporary table
                c.execute('''
                    CREATE TABLE temp_predictions AS 
                    SELECT id, date, league, home_team, away_team, 
                           predicted_outcome, actual_outcome, home_odds, 
                           draw_odds, away_odds, 
                           CAST((SELECT CAST(confidence AS REAL)) AS REAL) as confidence,
                           bet_amount, profit_loss, prediction_type, 
                           status, match_date, match_id 
                    FROM predictions
                ''')
                
                # Drop the original table
                c.execute('DROP TABLE predictions')
                
                # Rename temp table to original
                c.execute('ALTER TABLE temp_predictions RENAME TO predictions')
        except:
            pass
        
        conn.commit()
        conn.close()

    def add_prediction(self, prediction_data):
        """Add a new prediction to the database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Ensure confidence is a float
        try:
            confidence = float(prediction_data['confidence'])
        except (TypeError, ValueError):
            confidence = 0.0
        
        c.execute('''
            INSERT INTO predictions (
                date, league, home_team, away_team, predicted_outcome,
                home_odds, draw_odds, away_odds, confidence,
                bet_amount, prediction_type, status, match_date, match_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            prediction_data['date'],
            prediction_data['league'],
            prediction_data['home_team'],
            prediction_data['away_team'],
            prediction_data['predicted_outcome'],
            prediction_data['home_odds'],
            prediction_data['draw_odds'],
            prediction_data['away_odds'],
            confidence,  # Use the converted float value
            prediction_data['bet_amount'],
            prediction_data['prediction_type'],
            'Pending',
            prediction_data.get('match_date', prediction_data['date']),
            prediction_data['match_id']
        ))
        
        conn.commit()
        conn.close()

    def update_prediction_result(self, prediction_id, actual_outcome, profit_loss):
        """Update prediction with actual result and profit/loss"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            UPDATE predictions 
            SET actual_outcome = ?, profit_loss = ?, status = 'Completed'
            WHERE id = ?
        ''', (actual_outcome, profit_loss, prediction_id))
        
        conn.commit()
        conn.close()

    def get_predictions(self, start_date=None, end_date=None, status=None, confidence_level=None, league=None):
        """Get predictions with optional filters"""
        conn = sqlite3.connect(self.db_path)
        
        # Convert binary confidence to float in the query
        query = """
            SELECT 
                id, date, league, home_team, away_team, 
                predicted_outcome, actual_outcome, 
                home_odds, draw_odds, away_odds,
                CAST(confidence AS REAL) as confidence,
                bet_amount, profit_loss, prediction_type,
                status, match_date, match_id
            FROM predictions WHERE 1=1
        """
        params = []
        
        if start_date:
            query += " AND match_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND match_date <= ?"
            params.append(end_date)
        if status:
            query += " AND status = ?"
            params.append(status)
        if confidence_level:
            if confidence_level == "High":
                query += " AND CAST(confidence AS REAL) >= 70"  # High confidence: 70% or higher
            elif confidence_level == "Medium":
                query += " AND CAST(confidence AS REAL) >= 50 AND CAST(confidence AS REAL) < 70"  # Medium: 50-70%
            elif confidence_level == "Low":
                query += " AND CAST(confidence AS REAL) < 50"  # Low: below 50%
        if league and league != "All":
            query += " AND league = ?"
            params.append(league)
            
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    def calculate_statistics(self, confidence_level=None, league=None):
        """Calculate prediction statistics with optional confidence level and league filter"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # First, let's check what status values exist in the database
        c.execute("SELECT DISTINCT status FROM predictions")
        statuses = c.fetchall()
        print(f"Available status values: {statuses}")
        
        # Base query with confidence level and league filtering
        query = """
            SELECT 
                COUNT(*) as total_predictions,
                SUM(CASE WHEN predicted_outcome = actual_outcome THEN 1 ELSE 0 END) as correct_predictions,
                COALESCE(SUM(profit_loss), 0) as total_profit,
                COALESCE(SUM(bet_amount), 0) as total_bet_amount,
                AVG(CAST(confidence AS REAL)) as avg_confidence
            FROM predictions 
            WHERE status IN ('Completed', 'complete')
        """
        
        params = []
        # Add confidence level filtering if specified
        if confidence_level and confidence_level != "All":
            if confidence_level == "High":
                query += " AND CAST(confidence AS REAL) >= 70"
            elif confidence_level == "Medium":
                query += " AND CAST(confidence AS REAL) >= 50 AND CAST(confidence AS REAL) < 70"
            elif confidence_level == "Low":
                query += " AND CAST(confidence AS REAL) < 50"
        
        # Add league filtering if specified
        if league and league != "All":
            query += " AND league = ?"
            params.append(league)
        
        print(f"Executing query with confidence_level: {confidence_level}, league: {league}")
        print(f"Query: {query}")
        
        c.execute(query, params)
        stats = c.fetchone()
        print(f"Raw stats from database: {stats}")
        
        total_predictions = stats[0] or 0
        correct_predictions = stats[1] or 0
        total_profit = stats[2] or 0
        total_bet_amount = stats[3] or 0
        avg_confidence = stats[4] or 0
        
        # Calculate success rate and ROI
        success_rate = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        roi = (total_profit / total_bet_amount * 100) if total_bet_amount > 0 else 0
        
        conn.close()
        
        return {
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'success_rate': success_rate,
            'total_profit': total_profit,
            'roi': roi,
            'avg_confidence': avg_confidence
        }

    def update_match_results(self):
        """Update completed match results using match_analyzer"""
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        analyzer = MatchAnalyzer("633379bdd5c4c3eb26919d8570866801e1c07f399197ba8c5311446b8ea77a49")
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Get pending predictions
        c.execute("SELECT * FROM predictions WHERE status IN ('Pending', 'pending')")
        pending_predictions = c.fetchall()
        logger.info(f"Found {len(pending_predictions)} pending predictions")
        
        for pred in pending_predictions:
            try:
                match_id = pred[16]  # match_id
                if not match_id:
                    logger.warning(f"Skipping prediction {pred[0]}: No match_id")
                    continue
                    
                logger.info(f"Processing match {match_id} for prediction {pred[0]}")
                
                # Get match data from analyzer
                match_data = analyzer.get_match_details(match_id)
                if not match_data:
                    logger.warning(f"No match data found for match {match_id}")
                    continue
                    
                # Get match result
                result = analyzer.analyze_match_result(match_data)
                if not result:
                    logger.warning(f"No result data for match {match_id}")
                    continue
                    
                match_status = result.get('status', '').lower()
                logger.info(f"Match {match_id} status: {match_status}")
                
                # Only update if match is complete (handle different status values)
                if match_status in ['complete', 'finished', 'completed']:
                    # Calculate profit/loss
                    bet_amount = pred[10]  # bet_amount
                    predicted_outcome = pred[5]  # predicted_outcome
                    actual_outcome = result.get('winner', 'Pending')
                    
                    logger.info(f"Match {match_id}: Predicted={predicted_outcome}, Actual={actual_outcome}")
                    
                    # Get the odds based on predicted outcome
                    if predicted_outcome == 'HOME':
                        odds = pred[7]  # home_odds
                    elif predicted_outcome == 'AWAY':
                        odds = pred[9]  # away_odds
                    else:  # DRAW
                        odds = pred[8]  # draw_odds
                    
                    # Calculate profit/loss
                    if predicted_outcome == actual_outcome:
                        profit = bet_amount * (odds - 1)  # Winning bet
                    else:
                        profit = -bet_amount  # Losing bet
                    
                    # Update prediction with result
                    c.execute('''
                        UPDATE predictions 
                        SET actual_outcome = ?, profit_loss = ?, status = 'Completed'
                        WHERE id = ?
                    ''', (actual_outcome, profit, pred[0]))
                    
                    conn.commit()
                    logger.info(f"Updated result for match {match_id}: {actual_outcome} (Profit: {profit})")
                else:
                    logger.info(f"Match {match_id} not complete, status: {match_status}")
                    
            except Exception as e:
                logger.error(f"Error updating match result for prediction {pred[0]}: {str(e)}", exc_info=True)
                continue
        
        conn.close()

def style_dataframe(df):
    """Style the predictions dataframe with colors and formatting"""
    # Create a copy of the DataFrame to avoid modifying the original
    styled_df = df.copy()
    
    # Format status and result columns
    def format_status(row):
        # Check if actual outcome is None, empty, or 'Pending'
        if pd.isna(row['Actual Outcome']) or row['Actual Outcome'] == '' or row['Actual Outcome'] == 'Pending':
            return '⏳ Pending'
        elif row['Prediction'] == row['Actual Outcome']:
            return '✅ Won'
        else:
            return '❌ Lost'
    
    def format_profit(row):
        try:
            # Only show profit/loss for completed matches
            if pd.notna(row['Actual Outcome']) and row['Actual Outcome'] not in ['', 'Pending']:
                profit = float(row.get('Profit/Loss', 0))
                return f"{profit:+.2f}" if profit != 0 else "0.00"
            return "Pending"
        except:
            return "0.00"
    
    # Apply formatting
    styled_df['Result'] = styled_df.apply(format_status, axis=1)
    styled_df['Profit/Loss'] = styled_df.apply(format_profit, axis=1)
    
    # Reset index to remove it from display
    styled_df = styled_df.reset_index(drop=True)
    
    # Apply styling
    return styled_df.style\
        .apply(lambda x: ['background-color: #e6ffe6' if v == '✅ Won'
                         else 'background-color: #ffe6e6' if v == '❌ Lost'
                         else 'background-color: #f0f9ff' if v == '⏳ Pending'
                         else '' for v in x], subset=['Result'])\
        .set_properties(**{
            'text-align': 'center',
            'white-space': 'nowrap',
            'padding': '5px'
        })\
        .set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center')]},
            {'selector': 'td', 'props': [('text-align', 'center')]},
            {'selector': 'thead', 'props': [('display', 'none')]}
        ])

def show_history_page():
    """Display prediction history page"""
    # Check login state
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
    
    # Initialize PredictionHistory
    history = PredictionHistory()
    
    # Add date filter in sidebar with custom styling
    st.sidebar.markdown("""
        <div class="date-filter">
            <h2 style='color: #1e3c72; font-size: 1.2em; margin-bottom: 15px;'>Filters</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Get min and max dates from predictions
    all_predictions = history.get_predictions()
    if not all_predictions.empty:
        min_date = pd.to_datetime(all_predictions['match_date']).min()
        max_date = pd.to_datetime(all_predictions['match_date']).max()
        
        # Date range selector
        start_date = st.sidebar.date_input(
            "Start Date",
            min_date,
            min_value=min_date,
            max_value=max_date
        )
        
        end_date = st.sidebar.date_input(
            "End Date",
            max_date,
            min_value=start_date,
            max_value=max_date
        )

        # Add confidence level filter
        confidence_level = st.sidebar.selectbox(
            "Confidence Level",
            ["All", "High", "Medium", "Low"],
            help="Filter predictions by confidence level: High (≥70%), Medium (50-69%), Low (<50%)"
        )
        
        # Add league filter
        leagues = all_predictions['league'].unique().tolist()
        leagues.insert(0, "All")
        league = st.sidebar.selectbox(
            "League",
            leagues,
            help="Filter predictions by league"
        )
        
        # Get filtered predictions
        predictions = history.get_predictions(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            confidence_level=None if confidence_level == "All" else confidence_level,
            league=league
        )
        
        if not predictions.empty:
            # Update any pending predictions
            history.update_match_results()
            
            # Refresh predictions after update
            predictions = history.get_predictions(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                confidence_level=None if confidence_level == "All" else confidence_level,
                league=league
            )
            
            # Calculate statistics
            current_confidence = None if confidence_level == "All" else confidence_level
            current_league = None if league == "All" else league
            print(f"Selected confidence level: {confidence_level}, Selected league: {league}")
            stats = history.calculate_statistics(confidence_level=current_confidence, league=current_league)
            
            # Create metrics container
            st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
            
            # Display each metric
            metrics = [
                {"label": "Total Predictions", "value": stats['total_predictions'], "is_percentage": False, "is_currency": False},
                {"label": "Correct Predictions", "value": stats['correct_predictions'], "is_percentage": False, "is_currency": False},
                {"label": "Success Rate", "value": stats['success_rate'], "is_percentage": True, "is_currency": False},
                {"label": "Total Profit", "value": stats['total_profit'], "is_currency": True, "is_percentage": False},
                {"label": "ROI", "value": stats['roi'], "is_percentage": True, "is_currency": False}
            ]
            
            for metric in metrics:
                if metric.get("is_currency"):
                    formatted_value = f"£{metric['value']:.2f}"
                elif metric.get("is_percentage"):
                    formatted_value = f"{metric['value']:.1f}%"
                else:
                    formatted_value = str(metric['value'])
                
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
                    # Format DataFrame for display
                    display_df = predictions.copy()
                    
                    # Convert date columns
                    display_df['Date'] = pd.to_datetime(display_df['match_date'])\
                                          .dt.strftime('%Y-%m-%d')
                    
                    # Create Result column
                    display_df['Result'] = display_df.apply(
                        lambda x: '✅ Won' if pd.notna(x['predicted_outcome']) and x['predicted_outcome'] == x['actual_outcome']
                        else '❌ Lost' if pd.notna(x['actual_outcome'])
                        else '⏳ Pending',
                        axis=1
                    )
                    
                    # Format confidence as High/Medium/Low
                    def get_confidence_level(x):
                        try:
                            if pd.isna(x):
                                return 'N/A'
                            
                            # Try to convert to float
                            try:
                                confidence = float(x)
                            except (ValueError, TypeError):
                                # If direct conversion fails, try to decode bytes
                                if isinstance(x, bytes):
                                    import struct
                                    confidence = struct.unpack('f', x)[0]
                                else:
                                    return 'N/A'
                            
                            # Map confidence to levels
                            if confidence >= 70:
                                return 'High'
                            elif confidence >= 50:
                                return 'Medium'
                            else:
                                return 'Low'
                        except Exception as e:
                            print(f"Error processing confidence value: {x}, type: {type(x)}, error: {str(e)}")
                            return 'N/A'
                    
                    # Convert confidence values
                    display_df['confidence'] = pd.to_numeric(display_df['confidence'], errors='coerce')
                    display_df['Confidence'] = display_df['confidence'].apply(get_confidence_level)
                    
                    # Select and rename columns for display
                    display_columns = {
                        'Date': 'Match Date',
                        'league': 'League',
                        'home_team': 'Home Team',
                        'away_team': 'Away Team',
                        'predicted_outcome': 'Prediction',
                        'Confidence': 'Confidence',
                        'actual_outcome': 'Actual Outcome',
                        'Result': 'Result',
                        'profit_loss': 'Profit/Loss',
                        'status': 'Status'
                    }
                    
                    # Filter and rename columns
                    display_df = display_df[display_columns.keys()].rename(columns=display_columns)
                    
                    # Convert numeric columns
                    numeric_cols = ['Profit/Loss']
                    display_df[numeric_cols] = display_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
                    
                    # Fill missing values with appropriate defaults
                    defaults = {
                        'Match Date': 'Upcoming',
                        'League': 'Unknown League',
                        'Home Team': 'TBD',
                        'Away Team': 'TBD',
                        'Prediction': 'No Prediction',
                        'Confidence': 'N/A',
                        'Actual Outcome': 'Pending',
                        'Result': '⏳ Pending',
                        'Profit/Loss': 0.0,
                        'Status': 'Pending'
                    }
                    
                    # Update DataFrame with defaults
                    display_df = display_df.fillna(defaults)
                    
                    # Sort by date descending
                    display_df = display_df.sort_values('Match Date', ascending=False)
                    
                    # Apply styling
                    styled_df = style_dataframe(display_df)
                    
                    # Display the styled DataFrame
                    st.dataframe(
                        styled_df,
                        use_container_width=True,
                        height=400
                    )
                    
                except Exception as e:
                    st.error(f"Error displaying predictions table: {str(e)}")
                    st.exception(e)
            else:
                st.info("No predictions found for the selected date range.")

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
