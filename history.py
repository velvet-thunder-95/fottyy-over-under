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
        """Calculate overall prediction statistics"""
        df = self.get_predictions(status='Completed')
        
        total_predictions = len(df)
        if total_predictions > 0:
            correct_predictions = len(df[df['predicted_outcome'] == df['actual_outcome']])
            total_profit = df['profit_loss'].sum()
            roi = (total_profit / total_predictions * 100) if total_predictions > 0 else 0
        else:
            correct_predictions = 0
            total_profit = 0
            roi = 0
        
        return {
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'success_rate': (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0,
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
                    
                # Analyze match result
                result = analyzer.analyze_match_result(match_data)
                if result and result.get('status') == 'complete':
                    # Calculate profit/loss
                    bet_amount = pred[10]  # bet_amount
                    predicted_outcome = pred[5]  # predicted_outcome
                    actual_outcome = result['winner']
                    
                    if predicted_outcome == actual_outcome:
                        if actual_outcome == 'HOME':
                            profit = bet_amount * (pred[7] - 1)  # home_odds
                        elif actual_outcome == 'AWAY':
                            profit = bet_amount * (pred[9] - 1)  # away_odds
                        else:  # DRAW
                            profit = bet_amount * (pred[8] - 1)  # draw_odds
                    else:
                        profit = -bet_amount
                    
                    # Update prediction
                    self.update_prediction_result(
                        pred[0],  # id
                        actual_outcome,
                        profit
                    )
                    
                    print(f"Updated result for match {match_id}: {actual_outcome} (Profit: {profit})")
                    
            except Exception as e:
                print(f"Error updating match result: {str(e)}")
                continue
        
        conn.close()

def style_dataframe(df):
    # Create a copy of the DataFrame to avoid modifying the original
    styled_df = df.copy()
    
    # Replace None/NaN values with appropriate strings
    styled_df['Result'] = styled_df['Result'].fillna('⏳ Pending')
    styled_df['Profit/Loss'] = styled_df['Profit/Loss'].fillna(0.0)
    
    return styled_df.style\
        .apply(lambda x: ['background-color: #e6ffe6' if v == '✅ Won'
                         else 'background-color: #ffe6e6' if v == '❌ Lost'
                         else '' for v in x], subset=['Result'])\
        .format({'Profit/Loss': '{:.2f}', 'Date': '{}'})

def show_history_page():
    # Check login state
    if not check_login_state():
        st.query_params["page"] = "main"
        st.rerun()
    
    st.title("Match Prediction History")
    
    # Initialize history
    history = PredictionHistory()
    
    # Update results
    with st.spinner("Updating match results..."):
        history.update_match_results()
    
    # Date range filter
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            datetime.now().date() - timedelta(days=30),
            key="history_start_date"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            datetime.now().date(),
            key="history_end_date"
        )
    
    # Get predictions
    predictions_df = history.get_predictions(
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    # Calculate statistics
    stats = history.calculate_statistics()
    
    # Display statistics in a modern card layout
    st.markdown("""
        <style>
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin: 1rem 0;
        }
        .stat-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .stat-value {
            font-size: 1.8rem;
            font-weight: bold;
            color: #2c5282;
            margin-bottom: 0.5rem;
        }
        .stat-label {
            font-size: 0.9rem;
            color: #4a5568;
            font-weight: 500;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Display statistics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{stats['total_predictions']}</div>
                <div class="stat-label">Total Predictions</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{stats['success_rate']:.1f}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">£{stats['total_profit']:.2f}</div>
                <div class="stat-label">Total Profit</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{stats['roi']:.1f}%</div>
                <div class="stat-label">ROI</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Display predictions table
    if not predictions_df.empty:
        st.markdown("### Recent Predictions")
        
        try:
            # Format DataFrame for display
            display_df = predictions_df.copy()
            
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
                'confidence': 'Confidence',
                'status': 'Status'
            }
            
            # Filter and rename columns
            display_df = display_df[display_columns.keys()].rename(columns=display_columns)
            
            # Convert numeric columns
            numeric_cols = ['Profit/Loss', 'Confidence']
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
                'Confidence': 0.0,
                'Status': 'Pending'
            }
            
            # Update DataFrame with defaults
            display_df = display_df.fillna(defaults)
            
            # Sort by date descending
            display_df = display_df.sort_values('Match Date', ascending=False)
            
            # Apply styling
            styled_df = display_df.style\
                .format({
                    'Profit/Loss': '£{:.2f}',
                    'Confidence': '{:.1%}'
                })\
                .applymap(lambda x: 'background-color: #e6ffe6' if x == '✅ Won'
                         else 'background-color: #ffe6e6' if x == '❌ Lost'
                         else '', subset=['Result'])\
                .set_properties(**{
                    'text-align': 'center',
                    'font-size': '14px'
                })
            
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
