import psycopg2
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class AzureSync:
    def __init__(self):
        self.connection_params = {
            'host': 'soccer-prediction-model.postgres.database.azure.com',
            'user': 'soccer_prediction_team',
            'port': 5432,
            'database': 'soccer_predictions',
            'password': 'soccer_prediction_34509!',
            'sslmode': 'require'
        }
    
    def get_connection(self):
        """Get Azure PostgreSQL connection"""
        try:
            conn = psycopg2.connect(**self.connection_params)
            return conn
        except Exception as e:
            logger.error(f"Error connecting to Azure PostgreSQL: {str(e)}")
            return None
    
    def sync_prediction(self, prediction_data):
        """Sync a single prediction to Azure PostgreSQL"""
        try:
            conn = self.get_connection()
            if not conn:
                logger.error("Could not connect to Azure PostgreSQL")
                return False
            
            cursor = conn.cursor()
            
            # Transform data to match Azure table structure
            date_str = prediction_data.get('date', '')
            
            # Confidence level mapping (numeric to 1-3 scale)
            confidence_value = float(prediction_data.get('confidence', 0))

            if confidence_value >= 70:
                confidence_mapped = 3  # High
            elif confidence_value >= 50:
                confidence_mapped = 2  # Medium
            else:
                confidence_mapped = 1  # Low
            
            # Predicted outcome
            prediction_outcome = prediction_data.get('predicted_outcome', '')
            
            # Model probability
            model_probability = round(confidence_value / 100.0, 4)
            
            # Odds and implied probability
            home_odds = float(prediction_data.get('home_odds', 2.0))
            draw_odds = float(prediction_data.get('draw_odds', 3.0))
            away_odds = float(prediction_data.get('away_odds', 2.0))
            
            if prediction_outcome == 'HOME':
                implied_prob = 1.0 / home_odds
            elif prediction_outcome == 'AWAY':
                implied_prob = 1.0 / away_odds
            elif prediction_outcome == 'DRAW':
                implied_prob = 1.0 / draw_odds
            else:
                implied_prob = 0.5
            
            implied_probability = round(min(implied_prob, 0.9999), 4)
            
            # Delta
            delta = round(model_probability - implied_probability, 4)
            
            # Insert into Azure table
            insert_sql = """
            INSERT INTO soccer_predictions_data (
                "Date", "League", "Home Team", "Away Team",
                "Home Odds", "Draw Odds", "Away Odds",
                "Prediction", "Confidence", "Actual Outcome",
                "Result", "Profit/Loss", "Status",
                "Model_probability", "Implied_Probability", "Delta"
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s
            )
            """
            
            values = (
                date_str,
                prediction_data.get('league', '')[:100],
                prediction_data.get('home_team', '')[:100],
                prediction_data.get('away_team', '')[:100],
                home_odds,
                draw_odds,
                away_odds,
                prediction_outcome[:10],
                confidence_mapped,
                prediction_data.get('actual_outcome', '')[:10] if prediction_data.get('actual_outcome') else None,
                None,  # result will be calculated later
                float(prediction_data.get('profit_loss', 0.0)),
                'PENDING',
                model_probability,
                implied_probability,
                delta
            )
            
            cursor.execute(insert_sql, values)
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(
                f"Synced prediction: {prediction_data.get('home_team')} vs {prediction_data.get('away_team')} "
                f"- Model Prob: {model_probability}, Implied Prob: {implied_probability}, Delta: {delta}"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error syncing prediction to Azure: {str(e)}")
            return False

    def update_match_result(self, home_team, away_team, date, league, actual_outcome, profit_loss, predicted_outcome):
        """Update match result in Azure PostgreSQL"""
        try:
            conn = self.get_connection()
            if not conn:
                logger.error("Could not connect to Azure PostgreSQL")
                return False

            cursor = conn.cursor()

            result_value = "Won" if actual_outcome == predicted_outcome else "Lost"

            update_sql = """
            UPDATE soccer_predictions_data 
            SET "Actual Outcome" = %s,
                "Profit/Loss" = %s,
                "Status" = %s,
                "Result" = %s
            WHERE "Home Team" = %s 
            AND "Away Team" = %s 
            AND "Date" = %s 
            AND "League" = %s
            """
            
            values = (
                actual_outcome,
                float(profit_loss),
                'Completed',
                result_value,
                home_team,
                away_team,
                date,
                league
            )
            
            cursor.execute(update_sql, values)
            rows_updated = cursor.rowcount
            conn.commit()
            cursor.close()
            conn.close()
            
            if rows_updated > 0:
                logger.info(f"Azure: Updated {home_team} vs {away_team} -> {actual_outcome} ({result_value})")
                return True
            else:
                logger.warning(f"Azure: No match found for {home_team} vs {away_team} on {date}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating Azure: {str(e)}")
            return False

azure_sync = AzureSync()
