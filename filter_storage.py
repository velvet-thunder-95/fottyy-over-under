import json
from datetime import datetime
from supabase import create_client, Client
import os
import streamlit as st

# Supabase configuration
SUPABASE_URL = "https://uaihjkawqvhrcozxvvpd.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVhaWhqa2F3cXZocmNvenh2dnBkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzkzMTA4MTUsImV4cCI6MjA1NDg4NjgxNX0.mM1QqSxDbJt8LChJYJDlvXGqHMM22ZvvvodkdtuSqsc"

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
