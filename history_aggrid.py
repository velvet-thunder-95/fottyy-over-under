import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode

def display_predictions_with_buttons(predictions_df):
    """
    Display predictions dataframe with edit and delete buttons directly in the table
    
    Args:
        predictions_df: Pandas DataFrame with predictions data
    
    Returns:
        Dictionary with action and prediction_id if a button was clicked
    """
    # Create a copy of the dataframe for display
    display_df = predictions_df.copy()
    
    # Define custom cell renderers for edit and delete buttons
    edit_button_renderer = JsCode("""
    function(params) {
        return '<button style="background-color: #4CAF50; color: white; border: none; border-radius: 4px; padding: 5px 10px; cursor: pointer;">✏️ Edit</button>';
    }
    """)
    
    delete_button_renderer = JsCode("""
    function(params) {
        return '<button style="background-color: #f44336; color: white; border: none; border-radius: 4px; padding: 5px 10px; cursor: pointer;">🗑️ Delete</button>';
    }
    """)
    
    # Add button columns to the dataframe
    display_df['edit'] = 'Edit'
    display_df['delete'] = 'Delete'
    
    # Configure grid options
    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_default_column(min_column_width=100)
    
    # Configure columns
    gb.configure_column('edit', header_name='Edit', cellRenderer=edit_button_renderer, width=100)
    gb.configure_column('delete', header_name='Delete', cellRenderer=delete_button_renderer, width=100)
    gb.configure_column('ID', hide=True)
    
    # Add custom styling for the dataframe
    gb.configure_grid_options(domLayout='normal', rowHeight=50)
    
    # Build the grid options
    grid_options = gb.build()
    
    # Display the AgGrid with buttons
    grid_response = AgGrid(
        display_df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        allow_unsafe_jscode=True,
        theme='streamlit',
        height=500,
        fit_columns_on_grid_load=True,
        key=f'prediction_grid_{int(datetime.now().timestamp())}'
    )
    
    # Check if a cell was clicked
    result = {"action": None, "prediction_id": None}
    
    if grid_response['selected_rows']:
        selected_row = grid_response['selected_rows'][0]
        selected_id = selected_row['ID']
        
        # Check which column was clicked
        if 'clicked' in grid_response and grid_response['clicked'] is not None:
            clicked_column = grid_response['clicked'].get('column', None)
            
            if clicked_column == 'edit':
                result["action"] = "edit"
                result["prediction_id"] = selected_id
            elif clicked_column == 'delete':
                result["action"] = "delete"
                result["prediction_id"] = selected_id
    
    return result
