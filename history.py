# history.py

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
from football_api import get_match_by_teams, get_match_result
from session_state import init_session_state, check_login_state
from match_analyzer import MatchAnalyzer

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
        
        conn.commit()
        conn.close()

    def add_prediction(self, prediction_data):
        """Add a new prediction to the database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
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
            prediction_data['confidence'],
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

    def get_predictions(self, start_date=None, end_date=None, status=None):
        """Get predictions with optional filters"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM predictions WHERE 1=1"
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
            
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    def calculate_statistics(self):
        """Calculate prediction statistics"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Get all completed predictions
        c.execute("""
            SELECT 
                COUNT(*) as total_predictions,
                SUM(CASE WHEN predicted_outcome = actual_outcome THEN 1 ELSE 0 END) as correct_predictions,
                SUM(profit_loss) as total_profit,
                SUM(bet_amount) as total_bet_amount
            FROM predictions 
            WHERE status = 'complete'
        """)
        
        stats = c.fetchone()
        total_predictions = stats[0] or 0
        correct_predictions = stats[1] or 0
        total_profit = stats[2] or 0
        total_bet_amount = stats[3] or 0
        
        # Calculate success rate and ROI
        success_rate = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        roi = (total_profit / total_bet_amount * 100) if total_bet_amount > 0 else 0
        
        conn.close()
        
        return {
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'success_rate': success_rate,
            'total_profit': total_profit,
            'roi': roi
        }

    def update_match_results(self):
        """Update completed match results using match_analyzer"""
        analyzer = MatchAnalyzer("633379bdd5c4c3eb26919d8570866801e1c07f399197ba8c5311446b8ea77a49")
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Get pending predictions
        c.execute("SELECT * FROM predictions WHERE status = 'Pending'")
        pending_predictions = c.fetchall()
        
        for pred in pending_predictions:
            try:
                match_id = pred[16]  # match_id
                if not match_id:
                    continue
                    
                # Get match data from analyzer
                match_data = analyzer.get_match_details(match_id)
                if not match_data:
                    continue
                    
                # Get match result
                result = analyzer.analyze_match_result(match_data)
                if not result:
                    continue
                    
                match_status = result.get('status', '').lower()
                
                # Only update if match is complete
                if match_status == 'complete':
                    # Calculate profit/loss
                    bet_amount = pred[10]  # bet_amount
                    predicted_outcome = pred[5]  # predicted_outcome
                    actual_outcome = result.get('winner', 'Pending')
                    
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
                        SET actual_outcome = ?, profit_loss = ?, status = 'complete'
                        WHERE id = ?
                    ''', (actual_outcome, profit, pred[0]))
                    
                    conn.commit()
                    print(f"Updated result for match {match_id}: {actual_outcome} (Profit: {profit})")
                else:
                    print(f"Match not complete, status: {match_status}")
                    
            except Exception as e:
                print(f"Error updating match result: {str(e)}")
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
            display: flex;
            justify-content: space-between;
            margin-bottom: 30px;
            gap: 15px;
        }
        
        .metric-container {
            background: white;
            padding: 25px 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            text-align: center;
            transition: transform 0.2s ease;
            border: 1px solid #e9ecef;
            flex: 1;
        }
        
        .metric-container:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        }
        
        .metric-label {
            color: #495057;
            font-size: 0.95em;
            font-weight: 600;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .metric-value {
            color: #212529;
            font-size: 2em;
            font-weight: 700;
            line-height: 1.2;
        }
        
        .positive-value {
            color: #28a745;
            background: linear-gradient(45deg, #28a745 0%, #34ce57 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .negative-value {
            color: #dc3545;
            background: linear-gradient(45deg, #dc3545 0%, #e4606d 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
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
            <h2 style='color: #1e3c72; font-size: 1.2em; margin-bottom: 15px;'>Date Filter</h2>
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
        
        # Get filtered predictions
        predictions = history.get_predictions(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if not predictions.empty:
            # Update any pending predictions
            history.update_match_results()
            
            # Refresh predictions after update
            predictions = history.get_predictions(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            # Calculate statistics
            stats = history.calculate_statistics()
            
            # Display statistics in columns with custom styling
            def display_metric(container, label, value, is_percentage=False, is_currency=False, color_value=False):
                # Format the value based on type
                if isinstance(value, (int, float)):
                    if is_percentage:
                        formatted_value = f"{value:.1f}"
                    elif is_currency:
                        formatted_value = f"{value:.2f}"
                    else:
                        formatted_value = f"{int(value)}"
                else:
                    formatted_value = str(value)
                
                # Add currency symbol or percentage sign
                value_display = f"£{formatted_value}" if is_currency else f"{formatted_value}{'%' if is_percentage else ''}"
                
                # Determine if value is positive/negative for color coding
                try:
                    numeric_value = float(formatted_value)
                    value_class = ' positive-value' if numeric_value > 0 else ' negative-value' if numeric_value < 0 else ''
                except ValueError:
                    value_class = ''
                
                container.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value{value_class if color_value else ''}">
                            {value_display}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Create a container for metrics
            st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                display_metric(st, "Total Predictions", stats['total_predictions'])
            with col2:
                display_metric(st, "Correct Predictions", stats['correct_predictions'])
            with col3:
                display_metric(st, "Success Rate", stats['success_rate'], is_percentage=True)
            with col4:
                display_metric(st, "Total Profit", stats['total_profit'], is_currency=True, color_value=True)
            with col5:
                display_metric(st, "ROI", stats['roi'], is_percentage=True, color_value=True)
            
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
                    
                    # Select and rename columns for display
                    display_columns = {
                        'Date': 'Match Date',
                        'league': 'League',
                        'home_team': 'Home Team',
                        'away_team': 'Away Team',
                        'predicted_outcome': 'Prediction',
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
