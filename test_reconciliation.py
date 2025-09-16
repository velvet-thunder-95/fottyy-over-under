#!/usr/bin/env python3
"""
Test the new reconciliation functionality
"""

import sys
import os

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from azure_sync import azure_sync
from supabase_db import SupabaseDB
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_reconciliation():
    """Test the reconciliation between Azure and Supabase"""
    print("=== Testing Azure-Supabase Reconciliation ===\n")
    
    try:
        # Test Azure connection
        if not azure_sync.get_connection():
            print("❌ Cannot connect to Azure")
            return
        print("✅ Azure connection successful")
        
        # Test Supabase connection
        supabase_db = SupabaseDB()
        test_result = supabase_db.supabase.table('predictions').select('count').limit(1).execute()
        print("✅ Supabase connection successful")
        
        # Get pending predictions from Azure
        print("\n--- Getting pending predictions from Azure ---")
        pending_predictions = azure_sync.get_pending_predictions()
        print(f"Found {len(pending_predictions)} pending predictions in Azure")
        
        if pending_predictions:
            print("Sample pending predictions:")
            for i, pred in enumerate(pending_predictions[:3]):  # Show first 3
                print(f"  {i+1}. {pred['home_team']} vs {pred['away_team']} on {pred['date']}")
        
        # Run reconciliation
        print("\n--- Running reconciliation ---")
        reconciled_count = azure_sync.reconcile_with_supabase(supabase_db)
        
        print(f"\n=== Reconciliation Results ===")
        print(f"Predictions reconciled: {reconciled_count}")
        
        if reconciled_count > 0:
            print("✅ Some predictions were out of sync and have been updated")
        else:
            print("✅ All predictions are already in sync")
            
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_reconciliation()