import requests
import json
from datetime import datetime
from supabase_db import SupabaseDB
import logging

class MatchAnalyzer:
    def __init__(self, api_key):
        """Initialize the match analyzer with API key"""
        self.api_key = api_key
        self.base_url = "https://api.football-data-api.com"
        print(f"Initialized MatchAnalyzer with API key: {api_key}")
        
    def get_match_details(self, match_id):
        """Fetch match details from the API."""
        # If match_id is a dict, it means we already have the match data
        if isinstance(match_id, dict):
            return match_id
            
        # Otherwise, fetch from API
        url = f"{self.base_url}/match"
        params = {
            "key": self.api_key,
            "match_id": str(match_id)
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data.get('success'):
                logging.error(f"API request failed for match {match_id}")
                return None
                
            match_data = data.get('data', {})
            if not match_data:
                logging.error(f"No match data found for match {match_id}")
                return None
                
            return {
                'status': match_data.get('status', 'unknown'),
                'home_team': match_data.get('home_name'),
                'away_team': match_data.get('away_name'),
                'home_score': match_data.get('homeGoalCount'),
                'away_score': match_data.get('awayGoalCount'),
                'winner': {
                    'id': match_data.get('winningTeam'),
                    'name': match_data.get('home_name') if match_data.get('winningTeam') == match_data.get('homeID') 
                           else match_data.get('away_name') if match_data.get('winningTeam') == match_data.get('awayID')
                           else None
                }
            }
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching match {match_id}: {str(e)}")
            return None

    def get_match_league(self, match_id):
        """Get league information for a match from the database"""
        db = SupabaseDB()
        result = db.supabase.table('predictions').select('league').eq('match_id', match_id).limit(1).execute()
        
        if not result.data:
            return None
            
        league_name = result.data[0]['league']
        # Map league name to ID using the LEAGUE_IDS dictionary from football_api.py
        league_id = None
        
        # Print league mapping for debugging
        print(f"\nLooking up league ID for: {league_name}")
        if league_name in LEAGUE_IDS:
            league_id = LEAGUE_IDS[league_name]
            print(f"Found league ID: {league_id}")
        else:
            print(f"League not found in mapping")
            
        return {
            'league_name': league_name,
            'league_id': league_id
        }

    def fetch_match_data(self, match_id):
        """Fetch match data from the predictions database"""
        db = SupabaseDB()
        result = db.supabase.table('predictions').select('date,league,home_team,away_team,predicted_outcome,actual_outcome,home_odds,draw_odds,away_odds,status').eq('match_id', match_id).limit(1).execute()
        
        if result.data:
            return result.data[0]
        return None

    def fetch_all_match_ids(self):
        """Fetch all match IDs from the predictions database"""
        db = SupabaseDB()
        result = db.supabase.table('predictions').select('match_id').not_('match_id', 'is', None).order('date.desc').execute()
        return [match['match_id'] for match in result.data]
    
    def analyze_match_result(self, match_id):
        """
        Analyze the result of a match and return a dictionary with the match details.
        Returns None if the match cannot be found or analyzed.
        """
        try:
            match_details = self.get_match_details(match_id)
            if not match_details:
                logging.warning(f"No details found for match {match_id}")
                return None

            home_score = match_details.get('home_score')
            away_score = match_details.get('away_score')
            match_status = match_details.get('status', 'unknown').lower()

            # Map API status to our internal status
            status_mapping = {
                'complete': 'Completed',
                'finished': 'Completed',
                'incomplete': 'Pending',
                'scheduled': 'SCHEDULED',  # Keep scheduled status
                'postponed': 'Postponed',
                'cancelled': 'Cancelled'
            }
            status = status_mapping.get(match_status, 'unknown')

            # For future matches, set status to SCHEDULED
            if status == 'unknown':
                match_time = match_details.get('match_time')
                if match_time:
                    match_datetime = datetime.strptime(match_time, '%Y-%m-%d %H:%M:%S')
                    current_time = datetime.now()
                    if match_datetime > current_time:
                        status = 'SCHEDULED'

            # Determine the actual outcome based on scores
            actual_outcome = None
            if home_score is not None and away_score is not None:
                if home_score > away_score:
                    actual_outcome = 'HOME'
                elif away_score > home_score:
                    actual_outcome = 'AWAY'
                else:
                    actual_outcome = 'DRAW'

            return {
                'match_id': match_id,
                'status': status,
                'home_score': home_score,
                'away_score': away_score,
                'actual_outcome': actual_outcome
            }

        except Exception as e:
            logging.error(f"Error analyzing match {match_id}: {str(e)}")
            return None

    def get_prediction(self, match_id):
        """Get prediction for a match from database"""
        try:
            print(f"Getting prediction for match ID: {match_id}")
            
            if match_id is None:
                print("Cannot get prediction: Missing match ID")
                return None
                
            # Convert match_id to string if it's not already
            match_id = str(match_id)
            
            # Connect to database
            connection = sqlite3.connect('predictions.db')
            cursor = connection.cursor()
            
            # Get prediction
            cursor.execute("""
                SELECT predicted_outcome
                FROM predictions 
                WHERE match_id = ?
            """, (match_id,))
            result = cursor.fetchone()
            
            if result:
                predicted_outcome = result[0]
                print(f"Found prediction for match {match_id}: {predicted_outcome}")
                return predicted_outcome
            else:
                print(f"No prediction found for match ID: {match_id}")
                return None
                
        except Exception as e:
            print(f"Error getting prediction: {str(e)}")
            return None
        finally:
            if 'connection' in locals():
                connection.close()

    def calculate_profit_loss(self, predicted_outcome, actual_outcome, odds):
        """Calculate profit/loss for a bet using real odds from API"""
        try:
            print(f"Calculating P/L - Predicted: {predicted_outcome}, Actual: {actual_outcome}")
            print(f"Odds data: {json.dumps(odds, indent=2)}")  # Debug log
            
            if not predicted_outcome or not actual_outcome:
                print("Missing outcome data")
                return -1.0
                
            # Get odds for predicted outcome
            if predicted_outcome == "HOME":
                bet_odds = odds.get("home_odds", 0)
            elif predicted_outcome == "AWAY":
                bet_odds = odds.get("away_odds", 0)
            else:  # DRAW
                bet_odds = odds.get("draw_odds", 0)
                
            # Convert odds to float
            try:
                bet_odds = float(bet_odds)
            except (ValueError, TypeError):
                print(f"Invalid odds value: {bet_odds}")
                return -1.0
                
            print(f"Bet odds for {predicted_outcome}: {bet_odds}")
            
            # Calculate profit/loss
            if predicted_outcome == actual_outcome:
                if bet_odds <= 0:
                    print(f"Invalid odds for {predicted_outcome}: {bet_odds}")
                    return -1.0
                    
                profit = bet_odds - 1.0
                print(f"Bet won! Odds: {bet_odds}, Profit: {profit}")
                return round(profit, 2)
            else:
                print(f"Bet lost. Predicted {predicted_outcome}, Actual {actual_outcome}")
                return -1.0
                
        except Exception as e:
            print(f"Error calculating profit/loss: {str(e)}")
            print(f"Inputs - Predicted: {predicted_outcome}, Actual: {actual_outcome}")
            print(f"Odds data: {odds}")
            return -1.0  # Return loss on error

    def print_match_analysis(self, result, prediction=None):
        """Print detailed match analysis including stats"""
        if not result:
            print("No result data to display")
            return
            
        print("\nMatch Result:")
        print(f"Date: {result['date']}")
        print(f"{result['home_team']} vs {result['away_team']}")
        print(f"Score: {result['score_line']}")
        print(f"Status: {result['status']}")
        
        if result['winner']:
            print(f"Winner: {result['winner']}")
            
        if result.get('stats'):
            print("\nMatch Statistics:")
            stats = result['stats']
            print(f"Possession: Home {stats['possession']['home']}% - {stats['possession']['away']}% Away")
            print(f"Shots: Home {stats['shots']['home']} ({stats['shots']['home_on_target']} on target) - Away {stats['shots']['away']} ({stats['shots']['away_on_target']} on target)")
            print(f"Corners: Home {stats['corners']['home']} - {stats['corners']['away']} Away")
            print(f"Expected Goals (xG): Home {stats['xg']['home']:.2f} - {stats['xg']['away']:.2f} Away")
            
        if prediction:
            print("\nBetting Result:")
            print(f"Predicted: {prediction['predicted_outcome']}")
            print(f"Actual: {result['winner']}")
            if prediction.get('profit_loss') is not None:
                print(f"Profit/Loss: {prediction['profit_loss']:.2f}")

    def determine_winner_from_db(self, match_id):
        """Get the actual winner for older matches from database"""
        connection = sqlite3.connect('predictions.db')
        cursor = connection.cursor()
        
        # Get the prediction details
        cursor.execute("""
            SELECT predicted_outcome, home_odds, away_odds
            FROM predictions 
            WHERE match_id = ?
        """, (match_id,))
        result = cursor.fetchone()
        connection.close()
        
        if not result:
            return None
            
        predicted_outcome, home_odds, away_odds = result
        
        # For older matches, we'll determine the winner based on the odds
        # The team with lower odds is more likely to win
        if home_odds < away_odds:
            return "HOME"
        else:
            return "AWAY"

    def get_match_statistics(self, match_data):
        """Get detailed match statistics"""
        # For older matches or matches without statistics
        if not match_data:
            return None
            
        try:
            return {
                "corners": {
                    "home": match_data.get("team_a_corners", -1),
                    "away": match_data.get("team_b_corners", -1)
                },
                "shots": {
                    "home": {
                        "total": match_data.get("team_a_shots", -1),
                        "on_target": match_data.get("team_a_shots_on_target", -1)
                    },
                    "away": {
                        "total": match_data.get("team_b_shots", -1),
                        "on_target": match_data.get("team_b_shots_on_target", -1)
                    }
                },
                "possession": {
                    "home": match_data.get("team_a_possession", -1),
                    "away": match_data.get("team_b_possession", -1)
                }
            }
        except Exception as e:
            print(f"Error getting match statistics: {str(e)}")
            return None

    def update_match_result(self, match_id, actual_outcome, profit_loss):
        """Update match result and profit/loss in database"""
        try:
            print(f"Updating match result - ID: {match_id}, Outcome: {actual_outcome}, P/L: {profit_loss}")  # Debug log
            
            if match_id is None:
                print("Cannot update result: Missing match ID")
                return
                
            # Convert match_id to string if it's not already
            match_id = str(match_id)
            
            # Validate actual_outcome
            valid_outcomes = ["HOME", "AWAY", "DRAW"]
            if actual_outcome not in valid_outcomes and actual_outcome is not None:
                print(f"Invalid outcome: {actual_outcome}")
                return
                
            # Validate profit_loss is numeric
            if profit_loss is not None:
                try:
                    profit_loss = float(profit_loss)
                except (ValueError, TypeError):
                    print(f"Invalid profit/loss value: {profit_loss}")
                    return
            
            # Connect to database
            connection = sqlite3.connect('predictions.db')
            cursor = connection.cursor()
            
            # First check if the match exists and get its current status
            cursor.execute("""
                SELECT id, status, actual_outcome, profit_loss
                FROM predictions 
                WHERE match_id = ?
            """, (match_id,))
            result = cursor.fetchone()
            
            if not result:
                print(f"No prediction found for match ID: {match_id}")
                connection.close()
                return
                
            pred_id, current_status, current_outcome, current_profit = result
            
            # Don't update if already completed with same outcome
            if current_status == 'complete' and current_outcome == actual_outcome:
                print(f"Match {match_id} already completed with same outcome: {current_outcome}")
                connection.close()
                return
            
            # Update the prediction
            cursor.execute("""
                UPDATE predictions 
                SET actual_outcome = ?,
                    profit_loss = ?,
                    status = 'complete'
                WHERE id = ?
            """, (actual_outcome, profit_loss, pred_id))
            
            if cursor.rowcount > 0:
                print(f"Successfully updated match {match_id}")
                print(f"New status: complete")
                print(f"New outcome: {actual_outcome}")
                print(f"New profit/loss: {profit_loss}")
                connection.commit()
            else:
                print(f"No rows updated for match {match_id}")
            
            connection.close()
            
        except Exception as e:
            print(f"Error updating match result: {str(e)}")
            print(f"Match ID: {match_id}")
            print(f"Actual outcome: {actual_outcome}")
            print(f"Profit/loss: {profit_loss}")
            if 'connection' in locals():
                connection.close()

    def get_prediction_details(self, match_id):
        """Get prediction details from database"""
        connection = sqlite3.connect('predictions.db')
        cursor = connection.cursor()
        cursor.execute("""
            SELECT predicted_outcome, home_odds, draw_odds, away_odds, bet_amount 
            FROM predictions 
            WHERE match_id = ?
        """, (match_id,))
        result = cursor.fetchone()
        connection.close()
        return result if result else None

    def create_features(self, match_data):
        """Create features for prediction model"""
        try:
            if not match_data:
                return None
            
            features = {}
            
            # Handle PPG (Points Per Game) features with default values
            features['home_ppg'] = float(match_data.get('home_ppg', 1.5))  # Default to average PPG
            features['away_ppg'] = float(match_data.get('away_ppg', 1.5))
            features['pre_match_home_ppg'] = float(match_data.get('pre_match_home_ppg', 1.5))
            features['pre_match_away_ppg'] = float(match_data.get('pre_match_away_ppg', 1.5))
            features['pre_match_teamA_overall_ppg'] = float(match_data.get('pre_match_teamA_overall_ppg', 1.5))
            features['pre_match_teamB_overall_ppg'] = float(match_data.get('pre_match_teamB_overall_ppg', 1.5))
            
            # Handle xG (Expected Goals) features with default values
            features['team_a_xg'] = float(match_data.get('team_a_xg', 1.2))  # Default to average xG
            features['team_b_xg'] = float(match_data.get('team_b_xg', 1.0))
            features['total_xg'] = float(match_data.get('total_xg', 2.2))
            features['team_a_xg_prematch'] = float(match_data.get('team_a_xg_prematch', 1.2))
            features['team_b_xg_prematch'] = float(match_data.get('team_b_xg_prematch', 1.0))
            features['total_xg_prematch'] = float(match_data.get('total_xg_prematch', 2.2))
            
            # Handle potential features
            features['btts_potential'] = int(match_data.get('btts_potential', 1))
            features['btts_fhg_potential'] = int(match_data.get('btts_fhg_potential', 0))
            features['btts_2hg_potential'] = int(match_data.get('btts_2hg_potential', 0))
            features['o45_potential'] = int(match_data.get('o45_potential', 0))
            features['o35_potential'] = int(match_data.get('o35_potential', 1))
            features['o25_potential'] = int(match_data.get('o25_potential', 1))
            features['o15_potential'] = int(match_data.get('o15_potential', 1))
            features['o05_potential'] = int(match_data.get('o05_potential', 1))
            
            # Calculate derived features with safety checks
            home_strength = features['pre_match_teamA_overall_ppg']
            away_strength = features['pre_match_teamB_overall_ppg']
            total_strength = home_strength + away_strength
            
            if total_strength == 0:
                # If no historical data, use home advantage bias
                features['ppg_ratio'] = 0.6  # 60% home advantage
            else:
                features['ppg_ratio'] = home_strength / total_strength
                
            home_xg = features['team_a_xg_prematch']
            away_xg = features['team_b_xg_prematch']
            total_xg = home_xg + away_xg
            
            if total_xg == 0:
                # If no xG data, use home advantage bias
                features['xg_ratio'] = 0.55  # 55% home advantage
            else:
                features['xg_ratio'] = home_xg / total_xg
                
            return features
            
        except Exception as e:
            print(f"Error creating features: {str(e)}")
            # Return default features when error occurs
            return {
                'home_ppg': 1.5,
                'away_ppg': 1.5,
                'pre_match_home_ppg': 1.5,
                'pre_match_away_ppg': 1.5,
                'pre_match_teamA_overall_ppg': 1.5,
                'pre_match_teamB_overall_ppg': 1.5,
                'team_a_xg': 1.2,
                'team_b_xg': 1.0,
                'total_xg': 2.2,
                'team_a_xg_prematch': 1.2,
                'team_b_xg_prematch': 1.0,
                'total_xg_prematch': 2.2,
                'btts_potential': 1,
                'btts_fhg_potential': 0,
                'btts_2hg_potential': 0,
                'o45_potential': 0,
                'o35_potential': 1,
                'o25_potential': 1,
                'o15_potential': 1,
                'o05_potential': 1,
                'ppg_ratio': 0.6,
                'xg_ratio': 0.55
            }

def main():
    """Main function to analyze matches"""
    API_KEY = "633379bdd5c4c3eb26919d8570866801e1c07f399197ba8c5311446b8ea77a49"
    analyzer = MatchAnalyzer(API_KEY)
    
    # Get all match IDs from database
    match_ids = analyzer.fetch_all_match_ids()
    
    # Initialize statistics
    matches_analyzed = 0
    correct_predictions = 0
    total_profit_loss = 0
    
    print("\n==================================================")
    for match_id in match_ids:
        print(f"Analyzing Match ID: {match_id}")
        print("==================================================")
        
        # Get prediction details
        pred_details = analyzer.get_prediction_details(match_id)
        if not pred_details:
            print("Failed to fetch prediction details")
            continue
            
        predicted_outcome, home_odds, draw_odds, away_odds, bet_amount = pred_details
        odds = {"home_odds": home_odds, "draw_odds": draw_odds, "away_odds": away_odds}
        
        # Get match data
        match_data = analyzer.get_match_details(match_id)
        
        # Analyze match result
        result = analyzer.analyze_match_result(match_id)
        if result:
            # Calculate profit/loss
            profit_loss = analyzer.calculate_profit_loss(
                predicted_outcome, 
                result['actual_outcome'], 
                odds
            )
            
            # Print analysis
            analyzer.print_match_analysis(
                result, 
                prediction={
                    "predicted_outcome": predicted_outcome, 
                    "profit_loss": profit_loss
                }
            )
            
            # Update statistics only if we have a result
            if profit_loss is not None:
                total_profit_loss += profit_loss
                matches_analyzed += 1
                if predicted_outcome == result['actual_outcome']:
                    correct_predictions += 1
        else:
            print("Failed to fetch match data")
        
        print("\n==================================================")
    
    # Print overall statistics
    if matches_analyzed > 0:
        success_rate = (correct_predictions / matches_analyzed) * 100
        print("\nOverall Statistics:")
        print(f"Matches Analyzed: {matches_analyzed}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Success Rate: {success_rate:.2f}%")
        print(f"Total Profit/Loss: {total_profit_loss:.2f}")

if __name__ == "__main__":
    main()
