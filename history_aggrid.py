import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode, ColumnsAutoSizeMode

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
    
    # Ensure ID column is available for row identification
    if 'ID' not in display_df.columns and 'id' in display_df.columns:
        display_df = display_df.rename(columns={'id': 'ID'})
    
    # Define custom cell renderers for edit and delete buttons
    button_style = """
    function(params) {
        const button = document.createElement('button');
        button.innerHTML = params.value;
        button.style.cssText = `
            width: 100%;
            height: 100%;
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
            cursor: pointer;
            font-size: 0.9em;
            font-weight: 500;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 4px;
        `;
        
        if (params.value === 'Edit') {
            button.innerHTML = '✏️ ' + button.innerHTML;
            button.style.backgroundColor = '#4CAF50';
            button.style.color = 'white';
            button.onmouseover = function() {
                this.style.backgroundColor = '#45a049';
                this.style.transform = 'translateY(-1px)';
                this.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
            };
        } else {
            button.innerHTML = '🗑️ ' + button.innerHTML;
            button.style.backgroundColor = '#f44336';
            button.style.color = 'white';
            button.onmouseover = function() {
                this.style.backgroundColor = '#d32f2f';
                this.style.transform = 'translateY(-1px)';
                this.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
            };
        }
        
        button.onmouseout = function() {
            this.style.transform = '';
            this.style.boxShadow = '';
            if (params.value === 'Edit') {
                this.style.backgroundColor = '#4CAF50';
            } else {
                this.style.backgroundColor = '#f44336';
            }
        };
        
        button.onclick = (e) => {
            e.stopPropagation();
            const rowData = params.node.data;
            // Handle both 'ID' and 'id' column names for compatibility
            const rowId = rowData.ID !== undefined ? rowData.ID : rowData.id;
            if (rowId !== undefined) {
                document.body.dispatchEvent(
                    new CustomEvent('button_click', { 
                        detail: { 
                            action: params.value.toLowerCase(),
                            id: rowId
                        }
                    })
                );
            } else {
                console.error('Could not find ID in row data:', rowData);
            }
        };
        
        return button;
    }
    """
    
    # Add button columns to the dataframe
    display_df['Edit'] = 'Edit'
    display_df['Delete'] = 'Delete'
    
    # Configure grid options
    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_default_column(
        min_column_width=100,
        resizable=True,
        filterable=True,
        sortable=True
    )
    
    # Configure button columns
    gb.configure_column('Edit', 
                       header_name='',
                       cellRenderer=JsCode(button_style),
                       width=90,
                       pinned='left',
                       suppressMenu=True,
                       sortable=False,
                       filterable=False)
    
    gb.configure_column('Delete',
                        header_name='',
                        cellRenderer=JsCode(button_style),
                        width=90,
                        pinned='left',
                        suppressMenu=True,
                        sortable=False,
                        filterable=False)
    
    # Hide ID column from display but keep it in the data
    if 'ID' in display_df.columns:
        gb.configure_column('ID', hide=True)
    
    # Configure grid options
    grid_options = gb.build()
    
    # Set grid display options
    grid_options.update({
        'domLayout': 'autoHeight',
        'animateRows': True,
        'rowHeight': 50
    })
    
    # Add custom CSS for the grid
    st.markdown("""
    <style>
        .ag-theme-streamlit {
            --ag-border-radius: 8px;
            --ag-border-color: #e0e0e0;
            --ag-row-hover-color: #f5f5f5;
            --ag-header-background-color: #f8f9fa;
            --ag-odd-row-background-color: #ffffff;
            --ag-header-foreground-color: #495057;
            --ag-font-size: 14px;
            --ag-font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }
        .ag-header-cell {
            font-weight: 600 !important;
        }
        .ag-cell {
            display: flex;
            align-items: center;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Create a container for the grid with a key based on the current time
    container = st.container()
    
    # Display the AgGrid with buttons
    with container:
        # Add JavaScript to handle button clicks
        button_clicked = """
        <script>
        document.body.addEventListener('button_click', function(e) {
            const data = e.detail;
            const jsonData = JSON.stringify(data);
            const input = document.createElement('input');
            input.type = 'hidden';
            input.id = 'button_click_data';
            input.value = jsonData;
            document.body.appendChild(input);
            
            // Trigger a Streamlit component update
            const event = new Event('input', { bubbles: true });
            input.dispatchEvent(event);
        });
        </script>
        """
        
        # Ensure we're working with a copy of the display DataFrame
        display_df = display_df.copy()
        
        # Convert any datetime columns to strings to avoid serialization issues
        for col in display_df.select_dtypes(include=['datetime64']).columns:
            display_df[col] = display_df[col].astype(str)
            
        # Display the AgGrid component with the configured options
        grid_response = AgGrid(
            display_df,  # Pass the DataFrame directly
            gridOptions=grid_options,
            update_mode=GridUpdateMode.MODEL_CHANGED | GridUpdateMode.VALUE_CHANGED,
            fit_columns_on_grid_load=True,
            theme='streamlit',
            height=min(600, (len(display_df) + 1) * 50 + 50) if len(display_df) > 0 else 200,
            width='100%',
            reload_data=False,
            allow_unsafe_jscode=True,
            custom_css={
                ".ag-header-cell-label": {"justifyContent": "center"},
                ".ag-cell": {"display": "flex", "alignItems": "center", "justifyContent": "center"},
                ".ag-row-odd": {"backgroundColor": "#f9f9f9"},
                ".ag-row-hover": {"backgroundColor": "#f0f0f0 !important"}
            },
            columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
            enable_enterprise_modules=False,
            data_return_mode='AS_INPUT',
            try_to_convert_back_to_data_frame=True
        )
        
        # Add the JavaScript for handling button clicks
        st.components.v1.html(button_clicked, height=0)
        
        # Check if a button was clicked
        button_click_data = None
        try:
            button_click_data = st.session_state.get('button_click_data')
            if button_click_data:
                # Parse the JSON data
                data = json.loads(button_click_data)
                # Clear the state to prevent multiple triggers
                st.session_state.pop('button_click_data', None)
                
                # Return the action and ID
                return {
                    "action": data["action"].lower(),
                    "prediction_id": data["id"]
                }
        except Exception as e:
            st.error(f"Error processing button click: {str(e)}")
        
        # Return default response if no button was clicked
        return {"action": None, "prediction_id": None}
