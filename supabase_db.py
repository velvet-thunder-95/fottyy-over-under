from supabase import create_client
import os
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase configuration
SUPABASE_URL = "https://uaihjkawqvhrcozxvvpd.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVhaWhqa2F3cXZocmNvenh2dnBkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzkzMTA4MTUsImV4cCI6MjA1NDg4NjgxNX0.mM1QqSxDbJt8LChJYJDlvXGqHMM22ZvvvodkdtuSqsc"

class SupabaseDB:
    def __init__(self):
        self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.init_database()

    def init_database(self):
        """Initialize the database tables if they don't exist"""
        try:
            # Try to create the table if it doesn't exist
            sql = '''
            CREATE TABLE IF NOT EXISTS public.predictions (
                id SERIAL PRIMARY KEY,
                match_id TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                league TEXT NOT NULL,
                date DATE NOT NULL,
                predicted_outcome TEXT NOT NULL,
                actual_outcome TEXT,
                confidence FLOAT NOT NULL,
                home_odds FLOAT NOT NULL,
                draw_odds FLOAT NOT NULL,
                away_odds FLOAT NOT NULL,
                bet_amount FLOAT NOT NULL,
                profit_loss FLOAT DEFAULT 0.0,
                status TEXT DEFAULT 'Pending',
                prediction_type TEXT,
                home_market_value FLOAT,
                away_market_value FLOAT,
                home_score FLOAT,
                away_score FLOAT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            '''
            
            # Execute raw SQL to create table using RPC
            logger.info("Attempting to create predictions table...")
            self.supabase.rpc('execute_sql', {'sql': sql}).execute()
            logger.info("Successfully created predictions table")
            
            # Enable RLS and create policies
            rls_sql = '''
            ALTER TABLE IF EXISTS public.predictions ENABLE ROW LEVEL SECURITY;
            '''
            self.supabase.rpc('execute_sql', {'sql': rls_sql}).execute()
            logger.info("Enabled RLS")
            
            # Create authenticated users policy
            auth_policy_sql = '''
            DO $$ 
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_policies 
                    WHERE tablename = 'predictions' AND policyname = 'Enable all operations for authenticated users'
                ) THEN
                    CREATE POLICY "Enable all operations for authenticated users" ON public.predictions
                        FOR ALL
                        TO authenticated
                        USING (true)
                        WITH CHECK (true);
                END IF;
            END $$;
            '''
            self.supabase.rpc('execute_sql', {'sql': auth_policy_sql}).execute()
            logger.info("Created authenticated users policy")
            
            # Create anonymous users policy
            anon_policy_sql = '''
            DO $$ 
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_policies 
                    WHERE tablename = 'predictions' AND policyname = 'Enable read-only access for anonymous users'
                ) THEN
                    CREATE POLICY "Enable read-only access for anonymous users" ON public.predictions
                        FOR SELECT
                        TO anon
                        USING (true);
                END IF;
            END $$;
            '''
            self.supabase.rpc('execute_sql', {'sql': anon_policy_sql}).execute()
            logger.info("Created anonymous users policy")
            
            # Verify table exists
            self.supabase.table('predictions').select("*").limit(1).execute()
            logger.info("Database connection successful")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

    def add_prediction(self, prediction_data):
        """Add a new prediction to the database"""
        try:
            # Clean up the data before insertion
            clean_data = {
                'match_id': str(prediction_data.get('match_id', '')),
                'home_team': prediction_data.get('home_team', ''),
                'away_team': prediction_data.get('away_team', ''),
                'league': prediction_data.get('league', ''),
                'date': prediction_data.get('date', ''),
                'predicted_outcome': prediction_data.get('predicted_outcome', ''),
                'actual_outcome': prediction_data.get('actual_outcome'),
                'confidence': float(prediction_data.get('confidence', 0.0)),
                'home_odds': float(prediction_data.get('home_odds', 0.0)),
                'draw_odds': float(prediction_data.get('draw_odds', 0.0)),
                'away_odds': float(prediction_data.get('away_odds', 0.0)),
                'bet_amount': float(prediction_data.get('bet_amount', 0.0)),
                'profit_loss': float(prediction_data.get('profit_loss', 0.0)),
                'status': prediction_data.get('status', 'Pending'),
                'prediction_type': prediction_data.get('prediction_type'),
                'home_market_value': float(prediction_data.get('home_market_value', 0.0)) if prediction_data.get('home_market_value') is not None else None,
                'away_market_value': float(prediction_data.get('away_market_value', 0.0)) if prediction_data.get('away_market_value') is not None else None,
                'home_score': float(prediction_data.get('home_score', 0.0)) if prediction_data.get('home_score') is not None else None,
                'away_score': float(prediction_data.get('away_score', 0.0)) if prediction_data.get('away_score') is not None else None
            }
            
            # Insert the prediction
            result = self.supabase.table('predictions').insert(clean_data).execute()
            logger.info(f"Added prediction for {clean_data['home_team']} vs {clean_data['away_team']}")
            return result.data
        except Exception as e:
            logger.error(f"Error adding prediction: {str(e)}")
            return None

    def get_predictions(self, start_date=None, end_date=None):
        """Get predictions within a date range"""
        try:
            query = self.supabase.table('predictions').select("*")
            
            if start_date:
                query = query.gte('date', start_date)
                logger.info(f"Added start date filter: {start_date}")
            if end_date:
                query = query.lt('date', end_date)
                logger.info(f"Added end date filter: {end_date}")

            result = query.order('date.desc').execute()
            
            # Convert to pandas DataFrame
            if result.data:
                return pd.DataFrame(result.data)
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error getting predictions: {str(e)}")
            return pd.DataFrame()

    def update_prediction(self, match_id, update_data):
        """Update an existing prediction"""
        try:
            result = self.supabase.table('predictions')\
                .update(update_data)\
                .filter('match_id', 'eq', match_id)\
                .execute()
            logger.info(f"Updated prediction for match ID: {match_id}")
            return result.data
        except Exception as e:
            logger.error(f"Error updating prediction: {str(e)}")
            return None

    def delete_prediction(self, match_id):
        """Delete a prediction"""
        try:
            result = self.supabase.table('predictions')\
                .delete()\
                .filter('match_id', 'eq', match_id)\
                .execute()
            logger.info(f"Deleted prediction for match ID: {match_id}")
            return result.data
        except Exception as e:
            logger.error(f"Error deleting prediction: {str(e)}")
            return None

    def get_prediction_by_match(self, home_team, away_team, date):
        """Get prediction for a specific match"""
        try:
            result = self.supabase.table('predictions')\
                .select("*")\
                .filter('home_team', 'eq', home_team)\
                .filter('away_team', 'eq', away_team)\
                .filter('date', 'eq', date)\
                .execute()
            
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            logger.error(f"Error getting prediction by match: {str(e)}")
            return None
