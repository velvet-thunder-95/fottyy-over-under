import json
from datetime import datetime
from supabase import create_client, Client
import os
import streamlit as st

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_KEY')
)

def load_saved_filters():
    """Load filters from Supabase"""
    try:
        # Get filters for current user
        response = supabase.table('saved_filters') \
            .select('*') \
            .eq('user_id', st.session_state.user['id']) \
            .order('created_at', desc=True) \
            .execute()
        
        if response.data:
            return [{
                'name': filter['name'],
                'leagues': filter['leagues'],
                'confidence': filter['confidence'],
                'created': filter['created_at']
            } for filter in response.data]
        return []
    except Exception as e:
        st.error(f"Error loading filters: {str(e)}")
        return []

def save_filter(name, leagues, confidence_levels):
    """Save a new filter to Supabase"""
    try:
        data = {
            'name': name,
            'leagues': leagues,
            'confidence': confidence_levels,
            'user_id': st.session_state.user['id']
        }
        
        response = supabase.table('saved_filters').insert(data).execute()
        
        if response.data:
            return load_saved_filters()  # Reload all filters
        return []
    except Exception as e:
        st.error(f"Error saving filter: {str(e)}")
        return []

def delete_filter(filter_id):
    """Delete a filter from Supabase"""
    try:
        supabase.table('saved_filters') \
            .delete() \
            .eq('id', filter_id) \
            .eq('user_id', st.session_state.user['id']) \
            .execute()
        
        return load_saved_filters()  # Reload remaining filters
    except Exception as e:
        st.error(f"Error deleting filter: {str(e)}")
        return []
