import json
from datetime import datetime
from supabase import create_client, Client
import streamlit as st

# Supabase configuration from Streamlit secrets
SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["key"]

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def load_saved_filters():
    """Load filters from Supabase"""
    try:
        # Get filters for current user
        response = supabase.table('saved_filters') \
            .select('*') \
            .order('created_at', desc=True) \
            .execute()
        
        if response.data:
            return [{
                'id': filter['id'],
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
            'confidence': confidence_levels
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
            .execute()
        
        return load_saved_filters()  # Reload remaining filters
    except Exception as e:
        st.error(f"Error deleting filter: {str(e)}")
        return []

def load_history_saved_filters():
    """Load history page filters from Supabase"""
    try:
        response = supabase.table('history_saved_filters') \
            .select('*') \
            .order('created_at', desc=True) \
            .execute()
        if response.data:
            return [{
                'id': f['id'],
                'name': f['name'],
                'start_date': f['start_date'],
                'end_date': f['end_date'],
                'leagues': f['leagues'],
                'confidence': f['confidence'],
                'status': f['status'],
                'created': f['created_at']
            } for f in response.data]
        return []
    except Exception as e:
        st.error(f"Error loading history filters: {str(e)}")
        return []

def save_history_filter(name, start_date, end_date, leagues, confidence, status):
    """Save a new history page filter to Supabase"""
    try:
        data = {
            'name': name,
            'start_date': start_date,
            'end_date': end_date,
            'leagues': leagues,
            'confidence': confidence,
            'status': status
        }
        response = supabase.table('history_saved_filters').insert(data).execute()
        if response.data:
            return load_history_saved_filters()
        return []
    except Exception as e:
        st.error(f"Error saving history filter: {str(e)}")
        return []

def delete_history_filter(filter_id):
    """Delete a history page filter from Supabase"""
    try:
        supabase.table('history_saved_filters') \
            .delete() \
            .eq('id', filter_id) \
            .execute()
        return load_history_saved_filters()
    except Exception as e:
        st.error(f"Error deleting history filter: {str(e)}")
        return []
