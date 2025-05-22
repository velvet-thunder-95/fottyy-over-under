# graph_page.py


import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from history import PredictionHistory, get_confidence_level
import sys
import os
sys.path.append('.')
import importlib
filter_storage = importlib.import_module('filter_storage')  # For filter presets

# --- Helper Functions ---

def get_confidence_band(confidence):
    if confidence >= 70:
        return 'High'
    elif confidence >= 50:
        return 'Mid'
    else:
        return 'Low'

def calc_profit(row):
    # Use the stored profit_loss value from the database if available
    if 'profit_loss' in row and pd.notnull(row['profit_loss']):
        return float(row['profit_loss'])
    
    # Fallback calculation if profit_loss is not available
    bet_amount = 1.0  # Fixed $1 bet amount
    
    if row['predicted_outcome'] == row['actual_outcome']:
        # Won: Calculate profit based on the predicted outcome's odds
        if row['predicted_outcome'] == 'HOME':
            return float(round((row['home_odds'] * bet_amount) - bet_amount, 2))
        elif row['predicted_outcome'] == 'AWAY':
            return float(round((row['away_odds'] * bet_amount) - bet_amount, 2))
        elif row['predicted_outcome'] == 'DRAW':
            return float(round((row['draw_odds'] * bet_amount) - bet_amount, 2))
    # Lost: Lose the bet amount
    return float(-bet_amount)

def league_table_agg(df):
    # Do NOT assign conf_band here; it is assigned in render_graph_page
    
    # Ensure we have a profit column that uses stored profit_loss values when available
    if 'profit_loss' in df.columns:
        # Use profit_loss directly from the database
        df['profit'] = df['profit_loss'].apply(lambda x: float(x) if pd.notnull(x) else 0.0)
    else:
        # Fallback to calculation if profit_loss is not available
        df['profit'] = df.apply(calc_profit, axis=1)
    
    # Group by country, league, conf_band
    grouped = df.groupby(['country', 'league', 'conf_band'])
    agg = grouped.agg(
        Games=('correct', 'size'),  # count rows in group
        Correct=('correct', 'sum'),
        RatePct=('correct', 'mean'),
        Profit=('profit', 'sum')
    ).reset_index()
    # ROI: profit / total bet
    agg['ROI'] = agg['Profit'] / agg['Games'] * 100
    agg['RatePct'] = agg['RatePct'] * 100

    # Add 'All' confidence band for each (country, league)
    all_rows = []
    for (country, league), sub in agg.groupby(['country', 'league']):
        games = sub['Games'].sum()
        correct = sub['Correct'].sum()
        profit = sub['Profit'].sum()
        ratepct = (correct / games * 100) if games > 0 else 0
        roi = (profit / games * 100) if games > 0 else 0
        all_rows.append({
            'country': country,
            'league': league,
            'conf_band': 'All',
            'Games': games,
            'Correct': correct,
            'RatePct': ratepct,
            'Profit': profit,
            'ROI': roi
        })
    agg = pd.concat([agg, pd.DataFrame(all_rows)], ignore_index=True)
    return agg

def style_league_table(df):
    # Color styling for Rate%, Profit, ROI
    def highlight(val, col):
        if col == 'ROI' or col == 'Profit':
            if val > 0: return 'background-color: #d0f5d8; color: #1a4d1a;'
            if val < 0: return 'background-color: #fbe9e7; color: #b71c1c;'
        if col == 'RatePct':
            if val >= 70: return 'background-color: #34c759;'  # green
            if val < 50: return 'background-color: #ff9800;'  # orange
            if val < 40: return 'background-color: #ff3737;'  # red
        return ''
    styled = df.style.apply(lambda x: [highlight(v, c) for v, c in zip(x, x.index)], axis=1)
    return styled

def add_navigation_buttons():
    col1, col2, col3 = st.columns([2,2,2])
    
    with col1:
        if st.button("Home", key="main"):
            st.query_params["page"] = "main"
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

def render_graph_page():
    # Initialize session state variables if they don't exist
    if 'graph_data_loaded' not in st.session_state:
        st.session_state.graph_data_loaded = False
        
    if 'graph_df' not in st.session_state:
        # Initialize PredictionHistory
        ph = PredictionHistory()
        # Get initial data with default filters
        df = ph.get_predictions(status='Completed')
        st.session_state.graph_df = df
        st.session_state.graph_data_loaded = True
    
    if 'graph_all_predictions' not in st.session_state:
        ph = PredictionHistory()
        all_predictions = ph.get_predictions()
        st.session_state.graph_all_predictions = all_predictions
    
    # Initialize filter state variables
    if 'graph_filter_params' not in st.session_state:
        all_predictions = st.session_state.graph_all_predictions
        if not all_predictions.empty:
            min_date = pd.to_datetime(all_predictions['date']).min().date()
            max_date = pd.to_datetime(all_predictions['date']).max().date()
        else:
            min_date = datetime.now().date() - timedelta(days=30)
            max_date = datetime.now().date()
            
        unique_leagues = sorted(all_predictions['league'].unique()) if not all_predictions.empty else []
        
        st.session_state.graph_filter_params = {
            'start_date': min_date,
            'end_date': max_date,
            'leagues': ["All"],
            'confidence_levels': ["All"],
            'min_date': min_date,
            'max_date': max_date,
            'unique_leagues': unique_leagues
        }
    
    # Add some space before the title
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    # Display title with custom styling
    st.markdown("""
        <div class="title-container">
            <h1 class="title">TREND HISTORY PAGE</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Add navigation buttons
    add_navigation_buttons()
    
    # Apply styling
    st.markdown('''
    <style>
    .block-container {padding: 0 0 0 0;}
    .stDataFrame {font-size: 13px !important;}
    .stDataFrame th, .stDataFrame td {text-align: center !important;}
    .stDataFrame tbody tr:last-child {background: #e6f4ea !important; font-weight: bold;}
    
    /* Title styling */
    .title-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 15px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .title {
        color: white;
        font-size: 1.8em;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 0;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
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
    ''', unsafe_allow_html=True)
    
    # --- Sidebar Filters ---
    st.sidebar.markdown("## Filters", help="Filter analytics data")
    
    # Get filter parameters from session state
    params = st.session_state.graph_filter_params
    
    # Create a form to prevent automatic rerun on every input change
    with st.sidebar.form(key="filter_form"):
        # Date Range Filter
        start_date = st.date_input(
            "Start Date",
            value=params['start_date'],
            min_value=params['min_date'],
            max_value=params['max_date'],
            help="Filter data from this date"
        )
        
        end_date = st.date_input(
            "End Date",
            value=params['end_date'],
            min_value=params['min_date'],
            max_value=params['max_date'],
            help="Filter data until this date"
        )
        
        # Validate dates
        if start_date > end_date:
            st.error("Error: End date must be after start date")
        
        # League Filter
        leagues = st.multiselect(
            "Select Competitions",
            options=["All"] + params['unique_leagues'],
            default=params['leagues'],
            help="Filter data by competition. Select multiple competitions or 'All'"
        )
        
        # Handle empty selection
        if not leagues:
            leagues = ["All"]
        
        # Confidence Level Filter
        confidence_levels = st.multiselect(
            "Confidence Levels",
            options=["All", "High", "Medium", "Low"],
            default=params['confidence_levels'],
            help="Filter by confidence level: High (≥70%), Medium (50-69%), Low (<50%). Select multiple levels or 'All'"
        )
        
        # Handle empty selection
        if not confidence_levels:
            confidence_levels = ["All"]
        
        # Submit button
        submit_button = st.form_submit_button(label="Apply Filters", type="primary")
    
    # Process form submission
    if submit_button:
        # Update filter parameters in session state
        st.session_state.graph_filter_params.update({
            'start_date': start_date,
            'end_date': end_date,
            'leagues': leagues,
            'confidence_levels': confidence_levels
        })
        
        # Apply filters and update the dataframe
        ph = PredictionHistory()
        
        # Format dates for database query
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")  # Include end date
        
        # Get filtered data
        df = ph.get_predictions(
            start_date=start_date_str,
            end_date=end_date_str,
            status='Completed'
        )
        
        # Apply additional filters in memory
        if not df.empty:
            # Apply league filter
            if leagues and "All" not in leagues:
                df = df[df['league'].isin(leagues)]
            
            # Apply confidence filter
            if confidence_levels and "All" not in confidence_levels:
                mask = pd.Series(False, index=df.index)
                
                for level in confidence_levels:
                    if level == "High":
                        mask |= df['confidence'] >= 70
                    elif level == "Medium":
                        mask |= (df['confidence'] >= 50) & (df['confidence'] < 70)
                    elif level == "Low":
                        mask |= df['confidence'] < 50
                        
                df = df[mask]
        
        # Update the dataframe in session state
        st.session_state.graph_df = df
        st.session_state.graph_data_loaded = True
    
    # --- Filter Presets UI ---
    st.sidebar.markdown('### Analytics Filter Presets', help="Save and apply filter combinations for the analytics page.")
    with st.sidebar.container():
        graph_filter_name = st.text_input("Save Filter Preset", key="graph_filter_name")
        if st.button("Save Filter Preset", key="save_graph_filter"):
            if graph_filter_name:
                # Get current filter parameters
                params = st.session_state.graph_filter_params
                
                # Save exactly what the user selected, even if it's only one league
                leagues_to_save = params['leagues'].copy()
                if "All" in leagues_to_save and len(leagues_to_save) > 1:
                    leagues_to_save.remove("All")  # Remove 'All' if other leagues are selected
                if leagues_to_save == ["All"]:
                    leagues_to_save = []  # If only 'All' is selected, treat as no specific filter
                
                # Convert confidence levels to match the format expected by the filter_storage module
                conf_levels = []
                for level in params['confidence_levels']:
                    if level == "All":
                        conf_levels = ["All"]
                        break
                    elif level == "High":
                        conf_levels.append("High")
                    elif level == "Medium":
                        conf_levels.append("Medium")
                    elif level == "Low":
                        conf_levels.append("Low")
                
                # Save the filter preset using the history filter table
                st.session_state.graph_saved_filters = filter_storage.save_history_filter(
                    graph_filter_name,
                    params['start_date'].strftime("%Y-%m-%d"),
                    params['end_date'].strftime("%Y-%m-%d"),
                    leagues_to_save,
                    conf_levels,
                    None  # No status filter for graph page
                )
                st.success(f"Saved filter preset '{graph_filter_name}'!")
            else:
                st.error("Please enter a filter name.")
        
        # --- Load saved filters ---
        if 'graph_saved_filters' not in st.session_state:
            st.session_state.graph_saved_filters = filter_storage.load_history_saved_filters()
        
        if st.session_state.graph_saved_filters:
            st.markdown("#### Saved Filter Presets")
            for idx, sf in enumerate(st.session_state.graph_saved_filters):
                st.write(f"**{sf['name']}** | {sf['start_date']} to {sf['end_date']} | Leagues: {', '.join(sf['leagues']) if sf['leagues'] else 'All'} | Confidence: {', '.join(sf['confidence'])}")
                cols = st.columns([1,1])
                if cols[0].button("Apply", key=f"apply_graph_filter_{idx}"):
                    # Apply the filter preset
                    start_date = pd.to_datetime(sf['start_date']).date()
                    end_date = pd.to_datetime(sf['end_date']).date()
                    leagues = sf['leagues'] if sf['leagues'] else ["All"]
                    confidence = sf['confidence'] if sf['confidence'] else ["All"]
                    
                    # Update filter parameters in session state
                    st.session_state.graph_filter_params.update({
                        'start_date': start_date,
                        'end_date': end_date,
                        'leagues': leagues,
                        'confidence_levels': confidence
                    })
                    
                    # Apply filters and update the dataframe
                    ph = PredictionHistory()
                    
                    # Format dates for database query
                    start_date_str = start_date.strftime("%Y-%m-%d")
                    end_date_str = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")  # Include end date
                    
                    # Get filtered data
                    df = ph.get_predictions(
                        start_date=start_date_str,
                        end_date=end_date_str,
                        status='Completed'
                    )
                    
                    # Apply additional filters in memory
                    if not df.empty:
                        # Apply league filter
                        if leagues and "All" not in leagues:
                            df = df[df['league'].isin(leagues)]
                        
                        # Apply confidence filter
                        if confidence and "All" not in confidence:
                            mask = pd.Series(False, index=df.index)
                            
                            for level in confidence:
                                if level == "High":
                                    mask |= df['confidence'] >= 70
                                elif level == "Medium":
                                    mask |= (df['confidence'] >= 50) & (df['confidence'] < 70)
                                elif level == "Low":
                                    mask |= df['confidence'] < 50
                                    
                            df = df[mask]
                    
                    # Update the dataframe in session state
                    st.session_state.graph_df = df
                    st.rerun()
                if cols[1].button("Delete", key=f"delete_graph_filter_{idx}"):
                    st.session_state.graph_saved_filters = filter_storage.delete_history_filter(sf['id'])
                    st.rerun()
    
    # Use the dataframe from session state
    df = st.session_state.graph_df
    
    # Process the dataframe for display
    if not df.empty:
        # Drop unwanted columns if they exist
        for col in ['home_market_value', 'away_market_value', 'prediction_type']:
            if col in df.columns:
                df = df.drop(columns=col)
        
        # Split league names that contain hyphens into country and league parts
        if 'country' not in df.columns:
            # Create a function to split league names
            def split_league_name(league_name):
                if isinstance(league_name, str) and ' - ' in league_name:
                    parts = league_name.split(' - ', 1)
                    return parts[0], parts[1]
                return league_name, league_name
            
            # Apply the function to extract country and league
            df['country'], df['league'] = zip(*df['league'].apply(split_league_name))
        
        # Calculate correct predictions
        df['correct'] = (df['predicted_outcome'] == df['actual_outcome']).astype(int)
        
        # Add confidence band
        if 'confidence' in df.columns:
            df['conf_band'] = df['confidence'].apply(get_confidence_band)
        else:
            pass
        
        # Aggregate data for display
        agg = league_table_agg(df)
    else:
        st.info('No completed predictions to display.')
        return
    
    # Pivot for display with MultiIndex columns
    pivot = agg.pivot(index=['country', 'league'], columns='conf_band', values=['Games','Correct','RatePct','Profit','ROI'])
    # Swap MultiIndex levels so first is Confidence (band), second is Metric
    pivot = pivot.swaplevel(axis=1)
    pivot = pivot.sort_index(axis=1, level=[0,1])  # Optional: sort for clean order
    pivot.columns.set_names(['Confidence', 'Metric'], inplace=True)
    col_tuples = [(band, stat) for band in ['High', 'Mid', 'Low', 'All'] for stat in ['Games', 'Correct', 'RatePct', 'Profit', 'ROI']]
    pivot = pivot.reindex(columns=pd.MultiIndex.from_tuples(col_tuples, names=['Confidence', 'Metric']))
    # Define the full MultiIndex columns
    col_tuples = [("country", ""), ("league", "")] + [(band, stat) for band in ['High', 'Mid', 'Low', 'All'] for stat in ['Games', 'Correct', 'RatePct', 'Profit', 'ROI']]
    multi_cols = pd.MultiIndex.from_tuples(col_tuples, names=['Confidence', 'Metric'])

    pivot = pivot.reset_index()
    # Set columns to the full MultiIndex
    pivot.columns = multi_cols
    pivot = pivot.fillna(np.nan)

    # Add summary rows as in the image
    def summary_row_combined(bands, label, fill_bands):
        mask = agg['conf_band'].isin(bands)
        sub = agg[mask]
        row = {("country", ""): label, ("league", ""): ''}
        for band in ['High','Mid','Low','All']:
            for stat in ['Games','Correct','RatePct','Profit','ROI']:
                col = (band, stat)
                if band in fill_bands:
                    if stat == 'Games':
                        row[col] = sub[sub['conf_band']==band]['Games'].sum() if band != 'All' else sub['Games'].sum()
                    elif stat == 'Correct':
                        row[col] = sub[sub['conf_band']==band]['Correct'].sum() if band != 'All' else sub['Correct'].sum()
                    elif stat == 'RatePct':
                        games = sub[sub['conf_band']==band]['Games'].sum() if band != 'All' else sub['Games'].sum()
                        correct = sub[sub['conf_band']==band]['Correct'].sum() if band != 'All' else sub['Correct'].sum()
                        row[col] = (correct/games*100) if games > 0 else np.nan
                    elif stat == 'Profit':
                        row[col] = sub[sub['conf_band']==band]['Profit'].sum() if band != 'All' else sub['Profit'].sum()
                    elif stat == 'ROI':
                        games = sub[sub['conf_band']==band]['Games'].sum() if band != 'All' else sub['Games'].sum()
                        profit = sub[sub['conf_band']==band]['Profit'].sum() if band != 'All' else sub['Profit'].sum()
                        row[col] = (profit/games*100) if games > 0 else np.nan
                else:
                    row[col] = ''
        return row
    # Create a single row with all calculations and two blank rows
    summary_rows = [
        summary_row_combined(['High','Mid','Low'], 'Performance Summary', ['High','Mid','Low','All'])
    ]
    
    # Create completely empty rows
    empty_row = {}
    for col in multi_cols:
        empty_row[col] = ''
    
    # Add two empty rows
    summary_rows.append(empty_row.copy())
    summary_rows.append(empty_row.copy())
    # Convert summary rows to DataFrame with MultiIndex columns and reindex to match pivot
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.reindex(columns=multi_cols, fill_value='')
    # Concat main and summary
    full_df = pd.concat([pivot, summary_df], ignore_index=True)

    # Country group shading and thick borders
    def get_country_blocks(df):
        # Returns [(start_idx, end_idx, country_name), ...]
        blocks = []
        last_country = None
        start = 0
        for i, country in enumerate(df[('country', '')]):
            if country in ['Performance Summary', '']:
                break
            if country != last_country:
                if last_country is not None:
                    blocks.append((start, i-1, last_country))
                start = i
                last_country = country
        if last_country is not None:
            blocks.append((start, i, last_country))
        return blocks
    country_blocks = get_country_blocks(full_df)
    # Identify international/cup section (bottom block)
    intl_start = None
    for idx, (_, _, country) in enumerate(country_blocks):
        if isinstance(country, str) and country.lower() in ['champions league','champions league woman','conference league','europa league','nations league','nations league woman','copa libertadores','copa sudamericana']:
            intl_start = country_blocks[idx][0]
            break
    def row_style(row):
        idx = row.name
        # Summary rows
        if row[('country', '')] == 'Performance Summary':
            return ['background-color:#e6f4ea;font-weight:bold;border-top:3px solid #222;' for _ in row]
        # International/cup section
        if intl_start is not None and idx >= intl_start:
            return ['background-color:#e0f7fa;' for _ in row]
        # Country block shading
        for i, (start, end, country) in enumerate(country_blocks):
            if start <= idx <= end:
                color = '#e8f5e9' if i % 2 == 0 else '#fff'
                border = 'border-bottom:3px solid #222;' if idx == end else ''
                return [f'background-color:{color};{border}' for _ in row]
        # Fallback
        return ['' for _ in row]
    # Color coding for metrics (cell-wise)
    def color_cell(val, col):
        # Profit/ROI
        if col[1] in ['Profit','ROI']:
            if pd.isna(val) or val=='': return ''
            if val > 0: return 'background-color:#b6fcb6;color:#1a4d1a;'
            if val < 0: return 'background-color:#ffb3b3;color:#b71c1c;'
            return ''
        # RatePct
        if col[1]=='RatePct':
            if pd.isna(val) or val=='': return ''
            if val >= 70: return 'background-color:#34c759;color:#000;'
            if val >= 60: return 'background-color:#b6fcb6;'
            if val >= 50: return ''
            if val >= 40: return 'background-color:#ffe0b2;'
            if val < 40: return 'background-color:#ffb3b3;'
            return ''
        # Correct
        if col[1]=='Correct':
            if pd.isna(val) or val=='': return ''
            if val == 100: return 'background-color:#34c759;color:#000;'
            if val >= 50: return 'background-color:#ffe0b2;'
            if val < 50: return 'background-color:#ffb3b3;'
            return ''
        return ''

    # Apply row and cell styles
    def style_dataframe(df):
        # Row-wise shading
        def row_style(row):
            idx = row.name
            # Summary rows
            if row[('country', '')] == 'Performance Summary':
                return ['background-color:#e6f4ea;font-weight:bold;border-top:3px solid #222;' for _ in row]
            # International/cup section
            if intl_start is not None and idx >= intl_start:
                return ['background-color:#e0f7fa;' for _ in row]
            # Country block shading
            for i, (start, end, country) in enumerate(country_blocks):
                if start <= idx <= end:
                    color = '#e8f5e9' if i % 2 == 0 else '#fff'
                    border = 'border-bottom:3px solid #222;' if idx == end else ''
                    return [f'background-color:{color};{border}' for _ in row]
            # Fallback
            return ['' for _ in row]

        # Cell-wise color
        def cell_style(val, col):
            return color_cell(val, col)

        styler = df.style.apply(row_style, axis=1)
        for band in ['High','Mid','Low','All']:
            for stat in ['Games', 'Correct', 'Profit','ROI','RatePct']:
                styler = styler.applymap(lambda v: color_cell(v, (band,stat)), subset=pd.IndexSlice[:, (band,stat)])
        
        # Ensure numeric columns are properly typed as floats
        for band in ['High','Mid','Low','All']:
            for stat in ['Games', 'Correct', 'Profit','ROI','RatePct']:
                col = (band, stat)
                if col in df.columns:
                    # Convert empty strings to NaN first, then to numeric
                    df[col] = df[col].replace('', np.nan)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        # Format ALL numeric columns with appropriate formatting
        for band in ['High','Mid','Low','All']:
            # Apply formatting to ALL columns
            for stat in ['Games', 'Correct']:
                # Integer columns (no decimal places)
                styler = styler.format({(band,stat): lambda x: f"{int(x)}" if pd.notnull(x) else ''})
            for stat in ['RatePct', 'Profit', 'ROI']:
                # Float columns (2 decimal places with comma separator)
                styler = styler.format({(band,stat): lambda x: f"{x:.2f}".replace('.', ',') if pd.notnull(x) else ''})
        return styler

    # Remove the first table display - we'll only keep the last one
    # st.dataframe(style_dataframe(full_df), use_container_width=True)

    # Alternative: Apply custom style_func (duplicate styling logic, kept for user request)
    def style_func(df):
        styles = pd.DataFrame('', index=df.index, columns=df.columns)
        for i, row in df.iterrows():
            # Row striping and summary
            row_styles = row_style(row)
            for j, col in enumerate(df.columns):
                metric_style = color_cell(row[col], col)
                if metric_style:
                    styles.iloc[i, j] += metric_style
                styles.iloc[i, j] += row_styles[j]
                
                # Format the values for display
                if col[1] in ['Games', 'Correct'] and pd.notnull(row[col]) and row[col] != '':
                    try:
                        df.iloc[i, j] = f"{int(row[col])}"
                    except:
                        pass
                elif col[1] in ['RatePct', 'Profit', 'ROI'] and pd.notnull(row[col]) and row[col] != '':
                    try:
                        df.iloc[i, j] = f"{float(row[col]):.2f}".replace('.', ',')
                    except:
                        pass
        return styles

    # Apply styling to the dataframe
    styled = full_df.style.apply(style_func, axis=None)
    
    # Set table styles for borders, font, alignment
    styled = styled.set_table_styles([
        {'selector': 'th', 'props': [('font-size', '13px'), ('background', '#f8fafc'), ('border', '2px solid #bbb'), ('text-align','center')]},
        {'selector': 'td', 'props': [('border', '1px solid #ddd'), ('font-size', '13px'), ('text-align','center')]},
        {'selector': 'th.col_heading.level0', 'props': [('border-top', '3px solid #222'), ('font-size', '14px'), ('font-weight','bold'), ('background','#e8f5e9')]},
        {'selector': 'th.col_heading.level1', 'props': [('border-bottom', '2px solid #bbb')]},
        {'selector': 'th.row_heading', 'props': [('border-right', '2px solid #bbb')]},
    ], overwrite=False)
    
    # Create a copy of the original dataframe for styling
    styled_df = full_df.copy()
    
    # Define styling functions for different columns
    def style_ratepct(val):
        if pd.isna(val) or val == '':
            return ''
        try:
            val = float(val)
            if val >= 70:
                return 'background-color: #34c759;'
            elif val < 50:
                return 'background-color: #ff9800;'
            elif val < 40:
                return 'background-color: #ff3737;'
        except:
            pass
        return ''
    
    def style_profit_roi(val):
        if pd.isna(val) or val == '':
            return ''
        try:
            val = float(val)
            if val > 0:
                return 'background-color: #d0f5d8; color: #1a4d1a;'
            elif val < 0:
                return 'background-color: #fbe9e7; color: #b71c1c;'
        except:
            pass
        return ''
    
    # Apply styling to each cell based on column type
    def style_dataframe(df):
        # Create a DataFrame of empty strings with same shape as original
        styles = pd.DataFrame('', index=df.index, columns=df.columns)
        
        # Apply styling to each cell
        for i, row in df.iterrows():
            for j, col in enumerate(df.columns):
                # Apply alternating row colors
                if i % 2 == 1:
                    styles.iloc[i, j] += 'background-color: #f8f9fa;'
                
                # Apply specific styling based on column type
                if isinstance(col, tuple) and len(col) > 1:
                    band, metric = col
                    if metric == 'RatePct':
                        styles.iloc[i, j] += style_ratepct(row[col])
                    elif metric in ['Profit', 'ROI']:
                        styles.iloc[i, j] += style_profit_roi(row[col])
        
        return styles
    
    # Apply styling to the dataframe
    styled = styled_df.style.apply(style_dataframe, axis=None)
    
    # Format the values
    for band in ['High', 'Mid', 'Low', 'All']:
        for stat in ['Games', 'Correct']:
            # Integer columns (no decimal places)
            styled = styled.format({(band, stat): lambda x: f"{int(x)}" if pd.notnull(x) and x != '' else ''})
        for stat in ['RatePct', 'ROI']:
            # Percentage columns (2 decimal places)
            styled = styled.format({(band, stat): lambda x: f"{float(x):.2f}%" if pd.notnull(x) and x != '' else ''})
        for stat in ['Profit']:
            # Currency columns (2 decimal places)
            styled = styled.format({(band, stat): lambda x: f"{float(x):.2f}" if pd.notnull(x) and x != '' else ''})
    
    # Set table styles for borders, font, alignment
    styled = styled.set_table_styles([
        {'selector': 'th', 'props': [('font-size', '13px'), ('background', '#f8fafc'), ('border', '2px solid #bbb'), ('text-align','center')]},
        {'selector': 'td', 'props': [('border', '1px solid #ddd'), ('font-size', '13px'), ('text-align','center')]},
        {'selector': 'th.col_heading.level0', 'props': [('border-top', '3px solid #222'), ('font-size', '14px'), ('font-weight','bold'), ('background','#e8f5e9')]},
        {'selector': 'th.col_heading.level1', 'props': [('border-bottom', '2px solid #bbb')]},
        {'selector': 'th.row_heading', 'props': [('border-right', '2px solid #bbb')]},
        {'selector': 'tr:hover td', 'props': [('background-color', '#e8f5e9 !important')]},
    ], overwrite=False)
    
    # Create a flattened version of the dataframe for sorting
    flat_df = full_df.copy()
    
    # Create flattened column names by joining the MultiIndex levels with underscores
    flat_column_names = []
    for col in flat_df.columns:
        if isinstance(col, tuple):
            # Join the non-empty parts of the tuple with underscores
            parts = [str(part) for part in col if part != '']
            flat_column_names.append('_'.join(parts))
        else:
            flat_column_names.append(str(col))
    
    # Rename columns to flat names for sorting capability
    flat_df.columns = flat_column_names
    
    # Make sure numeric columns are properly typed for sorting
    for col in flat_df.columns:
        if 'Games' in col or 'Correct' in col or 'RatePct' in col or 'Profit' in col or 'ROI' in col:
            # Convert to numeric first to ensure proper sorting
            flat_df[col] = pd.to_numeric(flat_df[col], errors='coerce')
    
    # Create column configuration for proper sorting
    column_config = {}
    for col in flat_df.columns:
        if 'country' in col.lower() or 'league' in col.lower():
            column_config[col] = st.column_config.TextColumn(col)
        elif 'Games' in col or 'Correct' in col:
            column_config[col] = st.column_config.NumberColumn(col, format="%d")
        elif 'RatePct' in col:
            column_config[col] = st.column_config.NumberColumn(col, format="%.2f%%")
        elif 'Profit' in col:
            column_config[col] = st.column_config.NumberColumn(col, format="%.2f")
        elif 'ROI' in col:
            column_config[col] = st.column_config.NumberColumn(col, format="%.2f%%")
    
    # Add custom CSS for better styling
    st.markdown("""
    <style>
    [data-testid="stDataFrame"] table {
        border-collapse: collapse;
        width: 100%;
    }
    [data-testid="stDataFrame"] th {
        background-color: #f8fafc !important;
        border: 2px solid #bbb !important;
        font-size: 13px !important;
        text-align: center !important;
        cursor: pointer !important;
    }
    [data-testid="stDataFrame"] td {
        border: 1px solid #ddd !important;
        font-size: 13px !important;
        text-align: center !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create a display dataframe for formatting and a numeric dataframe for sorting
    display_df = full_df.copy()
    numeric_df = full_df.copy()
    
    # Create a dictionary to store the numeric values for sorting
    numeric_values = {}
    
    # Format the values for display while preserving numeric values for sorting
    for band in ['High', 'Mid', 'Low', 'All']:
        for stat in ['Games', 'Correct']:
            # Format integer columns
            col = (band, stat)
            if col in display_df.columns:
                # Store numeric values for sorting
                numeric_values[col] = numeric_df[col].apply(lambda x: float(x) if pd.notnull(x) and x != '' else float('-inf'))
                # Format for display
                display_df[col] = display_df[col].apply(lambda x: f"{int(x):,}" if pd.notnull(x) and x != '' else '')
        
        for stat in ['RatePct']:
            # Format percentage columns with 1 decimal place
            col = (band, stat)
            if col in display_df.columns:
                # Store numeric values for sorting
                numeric_values[col] = numeric_df[col].apply(lambda x: float(x) if pd.notnull(x) and x != '' else float('-inf'))
                # Format for display
                display_df[col] = display_df[col].apply(lambda x: f"{float(x):.1f}%" if pd.notnull(x) and x != '' else '')
        
        for stat in ['ROI']:
            # Format ROI with 2 decimal places
            col = (band, stat)
            if col in display_df.columns:
                # Store numeric values for sorting
                numeric_values[col] = numeric_df[col].apply(lambda x: float(x) if pd.notnull(x) and x != '' else float('-inf'))
                # Format for display
                display_df[col] = display_df[col].apply(lambda x: f"{float(x):.2f}%" if pd.notnull(x) and x != '' else '')
        
        for stat in ['Profit']:
            # Format currency columns with U suffix
            col = (band, stat)
            if col in display_df.columns:
                # Store numeric values for sorting
                numeric_values[col] = numeric_df[col].apply(lambda x: float(x) if pd.notnull(x) and x != '' else float('-inf'))
                # Format for display
                display_df[col] = display_df[col].apply(lambda x: f"{float(x):.2f}U" if pd.notnull(x) and x != '' else '')
    
    # Apply styling with colors
    def highlight_cells(val, col):
        if pd.isna(val) or val == '':
            return ''
        
        # Extract the metric type from the column
        metric = col[1] if isinstance(col, tuple) and len(col) > 1 else ''
        
        if metric == 'RatePct':
            try:
                # Extract numeric value from formatted string if needed
                if isinstance(val, str) and '%' in val:
                    val_float = float(val.replace('%', ''))
                else:
                    val_float = float(val)
                
                if val_float >= 70:
                    return 'background-color: rgba(52, 199, 89, 0.25);'  # Light green
                elif val_float < 50:
                    return 'background-color: rgba(255, 152, 0, 0.25);'  # Light orange
                elif val_float < 40:
                    return 'background-color: rgba(255, 55, 55, 0.25);'  # Light red
            except:
                pass
        
        elif metric in ['Profit', 'ROI']:
            try:
                # Extract numeric value from formatted string if needed
                if isinstance(val, str):
                    val_clean = val.replace('%', '').replace('U', '')
                    val_float = float(val_clean)
                else:
                    val_float = float(val)
                
                if val_float > 0:
                    return 'background-color: #d0f5d8; color: #1a4d1a;'
                elif val_float < 0:
                    return 'background-color: #fbe9e7; color: #b71c1c;'
            except:
                pass
        
        return ''
    
    # Apply styling to each cell
    def apply_styling(df):
        styles = pd.DataFrame('', index=df.index, columns=df.columns)
        
        for i, row in df.iterrows():
            # Determine row background based on performance
            # Check if this is an 'All' confidence band row and evaluate overall performance
            all_roi = None
            all_ratepct = None
            all_profit = None
            
            # Extract key metrics for the 'All' confidence band if available
            for col in df.columns:
                if isinstance(col, tuple) and len(col) > 1:
                    band, metric = col
                    if band == 'All':
                        if metric == 'ROI':
                            try:
                                # Extract numeric value from formatted string if needed
                                if isinstance(row[col], str) and '%' in row[col]:
                                    all_roi = float(row[col].replace('%', ''))
                                else:
                                    all_roi = float(row[col]) if pd.notnull(row[col]) else None
                            except:
                                pass
                        elif metric == 'RatePct':
                            try:
                                # Extract numeric value from formatted string if needed
                                if isinstance(row[col], str) and '%' in row[col]:
                                    all_ratepct = float(row[col].replace('%', ''))
                                else:
                                    all_ratepct = float(row[col]) if pd.notnull(row[col]) else None
                            except:
                                pass
                        elif metric == 'Profit':
                            try:
                                # Extract numeric value from formatted string if needed
                                if isinstance(row[col], str) and 'U' in row[col]:
                                    all_profit = float(row[col].replace('U', ''))
                                else:
                                    all_profit = float(row[col]) if pd.notnull(row[col]) else None
                            except:
                                pass
            
            # Determine row background color based on overall performance
            row_bg = ''
            
            # High performing leagues - good ROI, good success rate, and positive profit
            if (all_roi is not None and all_roi > 15 and 
                all_ratepct is not None and all_ratepct > 60 and 
                all_profit is not None and all_profit > 0):
                row_bg = 'background-color: rgba(52, 199, 89, 0.2);'  # Light green
            
            # Problematic leagues - negative ROI, low success rate
            elif (all_roi is not None and all_roi < -10 and 
                  all_ratepct is not None and all_ratepct < 45):
                row_bg = 'background-color: rgba(255, 55, 55, 0.2);'  # Light red
            
            # Promising leagues - positive ROI but could be better
            elif (all_roi is not None and all_roi > 0 and 
                  all_ratepct is not None and all_ratepct > 50):
                row_bg = 'background-color: rgba(52, 199, 89, 0.1);'  # Very light green
            
            # Underperforming leagues - negative ROI but not terrible
            elif (all_roi is not None and all_roi < 0):
                row_bg = 'background-color: rgba(255, 152, 0, 0.1);'  # Very light orange
            
            # Default alternating row colors if no special highlighting
            elif i % 2 == 1:
                row_bg = 'background-color: #f8f9fa;'
            
            for col in df.columns:
                # Start with row background
                cell_style = row_bg
                
                # Add cell-specific styling (this will override row background for specific cells)
                if isinstance(col, tuple) and len(col) > 1:
                    cell_specific_style = highlight_cells(row[col], col)
                    if cell_specific_style:
                        cell_style = cell_specific_style  # Override row background if cell has specific styling
                
                styles.loc[i, col] = cell_style
        
        return styles
    
    # Apply styling to the dataframe
    styled = display_df.style.apply(apply_styling, axis=None)
    
    # Format the display values - only format columns that haven't been pre-formatted
    for band in ['High', 'Mid', 'Low', 'All']:
        for stat in ['Games', 'Correct']:
            # Integer columns with thousands separator for large numbers
            styled = styled.format({(band, stat): '{:,}'}, na_rep='')
        
        # Note: We're not formatting RatePct, ROI, and Profit here because they've already been formatted
        # in the display_df preparation step above
    
    # Set table styles for borders, font, alignment
    styled = styled.set_table_styles([
        {'selector': 'th', 'props': [('font-size', '13px'), ('background', '#f8fafc'), ('border', '2px solid #bbb'), ('text-align','center')]},
        {'selector': 'td', 'props': [('border', '1px solid #ddd'), ('font-size', '13px'), ('text-align','center')]},
        {'selector': 'th.col_heading.level0', 'props': [('border-top', '3px solid #222'), ('font-size', '14px'), ('font-weight','bold'), ('background','#e8f5e9')]},
        {'selector': 'th.col_heading.level1', 'props': [('border-bottom', '2px solid #bbb')]},
        {'selector': 'th.row_heading', 'props': [('border-right', '2px solid #bbb')]},
        {'selector': 'tr:hover td', 'props': [('background-color', '#e8f5e9 !important')]},
    ], overwrite=False)
    
    # Use a different approach for sorting
    # Instead of adding hidden columns, we'll use the original numeric values for sorting
    # but display the formatted values
    
    # Create a new dataframe with flattened column names for better compatibility
    flat_display_df = display_df.copy()
    
    # Flatten the column names
    flat_display_df.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in flat_display_df.columns]
    
    # Create a modified styling function that works with flattened column names
    def apply_flat_styling(df):
        styles = pd.DataFrame('', index=df.index, columns=df.columns)
        
        for i, row in df.iterrows():
            # Determine row background based on performance
            # Extract key metrics for the 'All' confidence band if available
            all_roi = None
            all_ratepct = None
            all_profit = None
            
            # Extract key metrics from flattened column names
            for col in df.columns:
                if 'All_ROI' in col:
                    try:
                        # Extract numeric value from formatted string if needed
                        val = row[col]
                        if isinstance(val, str) and '%' in val:
                            all_roi = float(val.replace('%', ''))
                        else:
                            all_roi = float(val) if pd.notnull(val) else None
                    except:
                        pass
                elif 'All_RatePct' in col:
                    try:
                        # Extract numeric value from formatted string if needed
                        val = row[col]
                        if isinstance(val, str) and '%' in val:
                            all_ratepct = float(val.replace('%', ''))
                        else:
                            all_ratepct = float(val) if pd.notnull(val) else None
                    except:
                        pass
                elif 'All_Profit' in col:
                    try:
                        # Extract numeric value from formatted string if needed
                        val = row[col]
                        if isinstance(val, str) and 'U' in val:
                            all_profit = float(val.replace('U', ''))
                        else:
                            all_profit = float(val) if pd.notnull(val) else None
                    except:
                        pass
            
            # Determine row background color based on overall performance
            row_bg = ''
            
            # High performing leagues - good ROI, good success rate, and positive profit
            if (all_roi is not None and all_roi > 15 and 
                all_ratepct is not None and all_ratepct > 60 and 
                all_profit is not None and all_profit > 0):
                row_bg = 'background-color: rgba(52, 199, 89, 0.2);'  # Light green
            
            # Problematic leagues - negative ROI, low success rate
            elif (all_roi is not None and all_roi < -10 and 
                  all_ratepct is not None and all_ratepct < 45):
                row_bg = 'background-color: rgba(255, 55, 55, 0.2);'  # Light red
            
            # Promising leagues - positive ROI but could be better
            elif (all_roi is not None and all_roi > 0 and 
                  all_ratepct is not None and all_ratepct > 50):
                row_bg = 'background-color: rgba(52, 199, 89, 0.1);'  # Very light green
            
            # Underperforming leagues - negative ROI but not terrible
            elif (all_roi is not None and all_roi < 0):
                row_bg = 'background-color: rgba(255, 152, 0, 0.1);'  # Very light orange
            
            # Default alternating row colors if no special highlighting
            elif i % 2 == 1:
                row_bg = 'background-color: #f8f9fa;'
            
            for col in df.columns:
                # Start with row background
                cell_style = row_bg
                
                # Add cell-specific styling for flattened columns
                if '_' in col:
                    band, metric = col.split('_', 1)
                    val = row[col]
                    
                    # RatePct styling
                    if 'RatePct' in metric:
                        try:
                            if isinstance(val, str) and '%' in val:
                                val_float = float(val.replace('%', ''))
                            else:
                                val_float = float(val) if pd.notnull(val) else None
                                
                            if val_float is not None:
                                if val_float >= 70:
                                    cell_style = 'background-color: rgba(52, 199, 89, 0.25);'  # Light green
                                elif val_float < 50:
                                    cell_style = 'background-color: rgba(255, 152, 0, 0.25);'  # Light orange
                                elif val_float < 40:
                                    cell_style = 'background-color: rgba(255, 55, 55, 0.25);'  # Light red
                        except:
                            pass
                    
                    # Profit styling
                    elif 'Profit' in metric:
                        try:
                            if isinstance(val, str) and 'U' in val:
                                val_float = float(val.replace('U', ''))
                            else:
                                val_float = float(val) if pd.notnull(val) else None
                                
                            if val_float is not None:
                                if val_float > 0:
                                    cell_style = 'background-color: rgba(52, 199, 89, 0.25); color: #1a4d1a;'
                                elif val_float < 0:
                                    cell_style = 'background-color: rgba(255, 55, 55, 0.25); color: #b71c1c;'
                        except:
                            pass
                    
                    # ROI styling
                    elif 'ROI' in metric:
                        try:
                            if isinstance(val, str) and '%' in val:
                                val_float = float(val.replace('%', ''))
                            else:
                                val_float = float(val) if pd.notnull(val) else None
                                
                            if val_float is not None:
                                if val_float > 0:
                                    cell_style = 'background-color: rgba(52, 199, 89, 0.25); color: #1a4d1a;'
                                elif val_float < 0:
                                    cell_style = 'background-color: rgba(255, 55, 55, 0.25); color: #b71c1c;'
                        except:
                            pass
                
                styles.loc[i, col] = cell_style
        
        return styles
    
    # Apply styling to the flattened dataframe
    styled_df = flat_display_df.style.apply(apply_flat_styling, axis=None)
    
    # Display the dataframe with sorting enabled
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=False,
        width=2000
    )


# For Streamlit navigation
if __name__ == '__main__':
    render_graph_page()
