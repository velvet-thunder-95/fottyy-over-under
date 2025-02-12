import sqlite3
import pandas as pd
from supabase_db import SupabaseDB
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_data():
    """Migrate data from SQLite to Supabase"""
    try:
        # Connect to SQLite database
        sqlite_conn = sqlite3.connect('predictions.db')
        
        # Read all data from SQLite
        query = "SELECT * FROM predictions"
        df = pd.read_sql_query(query, sqlite_conn)
        
        # Close SQLite connection
        sqlite_conn.close()
        
        if df.empty:
            logger.info("No data to migrate")
            return
        
        # Initialize Supabase connection
        supabase_db = SupabaseDB()
        
        # Convert DataFrame records to list of dictionaries
        records = df.to_dict('records')
        
        # Migrate each record to Supabase
        success_count = 0
        error_count = 0
        
        for record in records:
            try:
                # Clean up the record
                # Convert numpy int64/float64 to Python native types
                for key, value in record.items():
                    if pd.isna(value):
                        record[key] = None
                    elif isinstance(value, (pd.Int64Dtype, pd.Float64Dtype)):
                        record[key] = float(value)
                
                # Add record to Supabase
                result = supabase_db.add_prediction(record)
                if result:
                    success_count += 1
                else:
                    error_count += 1
                    logger.error(f"Failed to migrate record: {record}")
            
            except Exception as e:
                error_count += 1
                logger.error(f"Error migrating record: {str(e)}")
        
        logger.info(f"Migration completed. Successfully migrated: {success_count}, Failed: {error_count}")
        
    except Exception as e:
        logger.error(f"Error during migration: {str(e)}")

if __name__ == "__main__":
    migrate_data()
