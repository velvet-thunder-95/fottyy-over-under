# utils.py
import pandas as pd

def get_confidence_level(confidence):
    """Convert confidence value to display text"""
    if confidence is None:
        return "Unknown"
    
    confidence = float(confidence)
    if confidence >= 90:
        return "🔥 Very High"
    elif confidence >= 80:
        return "⭐ High"
    elif confidence >= 70:
        return "👍 Medium"
    elif confidence >= 60:
        return "🤔 Low"
    else:
        return "❓ Very Low"

def style_dataframe(df):
    """Style the predictions dataframe with colors and formatting"""
    def style_row(row):
        styles = {}
        
        # Base style for all cells
        base_style = 'font-size: 14px; padding: 12px 15px; border-bottom: 1px solid #e0e0e0;'
        
        # Apply base style to all cells
        for col in df.columns:
            styles[col] = base_style
        
        # Style for status column
        if 'status' in df.columns:
            if row['status'] == 'Won':
                styles['status'] = f"{base_style} color: #00c853; font-weight: 500;"
            elif row['status'] == 'Lost':
                styles['status'] = f"{base_style} color: #ff3d00; font-weight: 500;"
            elif row['status'] == 'Pending':
                styles['status'] = f"{base_style} color: #ffa000; font-weight: 500;"
        
        # Style for profit/loss column
        if 'profit_loss' in df.columns and pd.notna(row['profit_loss']):
            if row['profit_loss'] > 0:
                styles['profit_loss'] = f"{base_style} color: #00c853; font-weight: 500;"
            elif row['profit_loss'] < 0:
                styles['profit_loss'] = f"{base_style} color: #ff3d00; font-weight: 500;"
        
        # Style for confidence column
        if 'confidence' in df.columns and pd.notna(row['confidence']):
            confidence = float(row['confidence'])
            if confidence >= 80:
                styles['confidence'] = f"{base_style} color: #00c853;"
            elif confidence >= 60:
                styles['confidence'] = f"{base_style} color: #ffa000;"
            else:
                styles['confidence'] = f"{base_style} color: #ff3d00;"
        
        return styles
    
    # Apply styling to the dataframe
    if df.empty:
        return df
    
    # Create styled dataframe
    styled_df = df.copy()
    
    # Format specific columns
    if 'date' in styled_df.columns:
        styled_df['date'] = pd.to_datetime(styled_df['date']).dt.strftime('%Y-%m-%d %H:%M')
    
    if 'stake' in styled_df.columns:
        styled_df['stake'] = styled_df['stake'].apply(lambda x: f"{x:.2f}U" if pd.notna(x) else "")
    
    if 'odds' in styled_df.columns:
        styled_df['odds'] = styled_df['odds'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    
    if 'profit_loss' in styled_df.columns:
        styled_df['profit_loss'] = styled_df['profit_loss'].apply(
            lambda x: f"+{x:.2f}U" if pd.notna(x) and x > 0 
            else f"{x:.2f}U" if pd.notna(x) 
            else ""
        )
    
    # Apply row styles
    styled_df = styled_df.style.apply(style_row, axis=1)
    
    return styled_df
