# graph_page.py


import streamlit as st
import pandas as pd
import numpy as np
from history import PredictionHistory, get_confidence_level

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
    st.title('-------------------------------------------------')
    st.title('League & Confidence Analytics Page ')
    
    # Add navigation buttons
    add_navigation_buttons()
    
    st.markdown('''
    <style>
    .block-container {padding: 0 0 0 0;}
    .stDataFrame {font-size: 13px !important;}
    .stDataFrame th, .stDataFrame td {text-align: center !important;}
    .stDataFrame tbody tr:last-child {background: #e6f4ea !important; font-weight: bold;}
    </style>
    ''', unsafe_allow_html=True)

    ph = PredictionHistory()
    df = ph.get_predictions(status='Completed')
    
    # Drop unwanted columns if they exist
    for col in ['home_market_value', 'away_market_value', 'prediction_type']:
        if col in df.columns:
            df = df.drop(columns=col)
    if df.empty:
        st.info('No completed predictions to display.')
        return
    if 'country' not in df.columns:
        df['country'] = df['league']
    df['correct'] = (df['predicted_outcome'] == df['actual_outcome']).astype(int)
    if 'confidence' in df.columns:
        df['conf_band'] = df['confidence'].apply(get_confidence_band)
        # st.write(df['conf_band'].value_counts(dropna=False))  # Debug output
    else:
        # st.write('No confidence column found!')  # Debug output
        pass
    agg = league_table_agg(df)
    
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
    summary_rows = [
        summary_row_combined(['High','Mid'], 'Mid/High Confidence', ['High','Mid']),
        summary_row_combined(['Low'], 'Low Confidence', ['Low']),
        summary_row_combined(['High','Mid','Low'], 'All Confidences', ['All'])
    ]
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
            if country in ['Mid/High Confidence','Low Confidence','All Confidences']:
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
        if row[('country', '')] in ['Mid/High Confidence','Low Confidence','All Confidences']:
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
            if row[('country', '')] in ['Mid/High Confidence','Low Confidence','All Confidences']:
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

    styled = full_df.style.apply(style_func, axis=None)
    # Set table styles for borders, font, alignment
    styled = styled.set_table_styles([
        {'selector': 'th', 'props': [('font-size', '13px'), ('background', '#f8fafc'), ('border', '2px solid #bbb'), ('text-align','center')]},
        {'selector': 'td', 'props': [('border', '1px solid #ddd'), ('font-size', '13px'), ('text-align','center')]},
        {'selector': 'th.col_heading.level0', 'props': [('border-top', '3px solid #222'), ('font-size', '14px'), ('font-weight','bold'), ('background','#e8f5e9')]},
        {'selector': 'th.col_heading.level1', 'props': [('border-bottom', '2px solid #bbb')]},
        {'selector': 'th.row_heading', 'props': [('border-right', '2px solid #bbb')]},
    ], overwrite=False)
    st.dataframe(styled, use_container_width=True, hide_index=True, width=2000)


# For Streamlit navigation
if __name__ == '__main__':
    render_graph_page()
