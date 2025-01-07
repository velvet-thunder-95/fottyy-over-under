import requests
import json
from datetime import datetime
import sqlite3

class MatchAnalyzer:
    def __init__(self, api_key):
        """Initialize the match analyzer with API key"""
        self.api_key = api_key
        self.base_url = "https://api.football-data-api.com"
        print(f"Initialized MatchAnalyzer with API key: {api_key}")
        
    def get_match_details(self, match_id):
        """Fetch match details from API or local database"""
        # First get match info from our database
        match_info = self.fetch_match_data(match_id)
        if not match_info:
            print(f"No match found in database with ID {match_id}")
            return None
            
        match_date = datetime.strptime(match_info['match_date'], '%Y-%m-%d')
        current_date = datetime.now()
        
        # For future matches, mark as scheduled
        if match_date > current_date:
            print(f"Match {match_id} is scheduled for future")
            return None
            
        # For past matches, try to get data from API
        url = f"{self.base_url}/match"
        params = {
            "key": self.api_key,
            "match_id": str(match_id)  # Convert to string to be safe
        }
        
        try:
            print(f"Fetching match data from API for match {match_id}")
            print(f"API URL: {url}")
            print(f"API Params: {params}")
            
            response = requests.get(url, params=params)
            print(f"API Response Status: {response.status_code}")
            
            response.raise_for_status()
            data = response.json()
            
            if data.get("success") and data.get("data"):
                return data["data"]
            
            print(f"No API data available for match {match_id}")
            return None
            
        except Exception as e:
            print(f"Error fetching match details: {str(e)}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"API Error Response: {e.response.text}")
            return None

    def get_match_league(self, match_id):
        """Get league information for a match from the database"""
        connection = sqlite3.connect('predictions.db')
        cursor = connection.cursor()
        
        cursor.execute("""
            SELECT league 
            FROM predictions 
            WHERE match_id = ?
            LIMIT 1
        """, (match_id,))
        
        result = cursor.fetchone()
        connection.close()
        
        if not result:
            return None
            
        league_name = result[0]
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
        connection = sqlite3.connect('predictions.db')
        cursor = connection.cursor()
        cursor.execute("""
            SELECT date, league, home_team, away_team, 
                   predicted_outcome, actual_outcome,
                   home_odds, draw_odds, away_odds,
                   match_date, status
            FROM predictions 
            WHERE match_id = ?
            LIMIT 1
        """, (match_id,))
        
        result = cursor.fetchone()
        connection.close()
        
        if result:
            return {
                'date': result[0],
                'league': result[1],
                'home_team': result[2],
                'away_team': result[3],
                'predicted_outcome': result[4],
                'actual_outcome': result[5],
                'home_odds': result[6],
                'draw_odds': result[7],
                'away_odds': result[8],
                'match_date': result[9],
                'status': result[10]
            }
        return None

    def fetch_all_match_ids(self):
        """Fetch all match IDs from the predictions database"""
        connection = sqlite3.connect('predictions.db')
        cursor = connection.cursor()
        # Only get predictions with valid match_ids
        cursor.execute("""
            SELECT match_id, date
            FROM predictions 
            WHERE match_id IS NOT NULL
            ORDER BY date DESC, id DESC
        """)
        matches = cursor.fetchall()
        connection.close()
        return [match[0] for match in matches]
    
    def analyze_match_result(self, match_data):
        """Analyze the match result using Footystats API data"""
        if not match_data:
            print("No match data available")
            return None
            
        try:
            # Get basic match info
            home_team = match_data.get("home_name")
            away_team = match_data.get("away_name")
            status = match_data.get("status")
            
            # Get goals
            home_goals = match_data.get("homeGoalCount", 0)
            away_goals = match_data.get("awayGoalCount", 0)
            score = f"{home_goals} - {away_goals}"
            
            # Get winner based on winningTeam ID
            winner = None
            if status == "complete":
                winning_team_id = match_data.get("winningTeam")
                if winning_team_id == match_data.get("homeID"):
                    winner = "HOME"
                elif winning_team_id == match_data.get("awayID"):
                    winner = "AWAY"
                elif winning_team_id == -1:  # Draw
                    winner = "DRAW"
            
            # Get detailed stats
            stats = {
                "corners": {
                    "home": match_data.get("team_a_corners", 0),
                    "away": match_data.get("team_b_corners", 0),
                    "total": match_data.get("totalCornerCount", 0)
                },
                "shots": {
                    "home": match_data.get("team_a_shots", 0),
                    "away": match_data.get("team_b_shots", 0),
                    "home_on_target": match_data.get("team_a_shotsOnTarget", 0),
                    "away_on_target": match_data.get("team_b_shotsOnTarget", 0)
                },
                "possession": {
                    "home": match_data.get("team_a_possession", 0),
                    "away": match_data.get("team_b_possession", 0)
                },
                "cards": {
                    "home_yellow": match_data.get("team_a_yellow_cards", 0),
                    "away_yellow": match_data.get("team_b_yellow_cards", 0),
                    "home_red": match_data.get("team_a_red_cards", 0),
                    "away_red": match_data.get("team_b_red_cards", 0)
                },
                "xg": {
                    "home": match_data.get("team_a_xg", 0),
                    "away": match_data.get("team_b_xg", 0),
                    "total": match_data.get("total_xg", 0)
                }
            }
            
            # Get odds
            odds = {
                "home_odds": match_data.get("odds_ft_1", 0),
                "draw_odds": match_data.get("odds_ft_x", 0),
                "away_odds": match_data.get("odds_ft_2", 0),
                "btts_yes": match_data.get("odds_btts_yes", 0),
                "btts_no": match_data.get("odds_btts_no", 0),
                "over25": match_data.get("odds_ft_over25", 0),
                "under25": match_data.get("odds_ft_under25", 0)
            }
            
            return {
                "date": datetime.fromtimestamp(match_data.get("date_unix", 0)).strftime('%Y-%m-%d'),
                "home_team": home_team,
                "away_team": away_team,
                "score_line": score,
                "status": status,
                "winner": winner,
                "stats": stats,
                "odds": odds,
                "match_id": match_data.get("id")
            }
        except Exception as e:
            print(f"Error analyzing match result: {str(e)}")
            return None

    def calculate_profit_loss(self, predicted_outcome, actual_outcome, odds):
        """Calculate profit/loss for a bet using real odds from API"""
        if not actual_outcome:
            print("No actual outcome available")
            return None
            
        if predicted_outcome == actual_outcome:
            if predicted_outcome == "HOME":
                return round(float(odds["home_odds"]) - 1, 2)
            elif predicted_outcome == "AWAY":
                return round(float(odds["away_odds"]) - 1, 2)
            else:  # DRAW
                return round(float(odds["draw_odds"]) - 1, 2)
        else:
            return -1.0

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
        if actual_outcome is None or profit_loss is None:
            print("Cannot update result: Missing outcome or profit/loss")
            return
            
        connection = sqlite3.connect('predictions.db')
        cursor = connection.cursor()
        cursor.execute("""
            UPDATE predictions 
            SET actual_outcome = ?,
                profit_loss = ?,
                status = 'complete'
            WHERE match_id = ?
        """, (actual_outcome, profit_loss, match_id))
        connection.commit()
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
        result = analyzer.analyze_match_result(match_data)
        if result:
            # Calculate profit/loss
            profit_loss = analyzer.calculate_profit_loss(
                predicted_outcome, 
                result['winner'], 
                result.get('odds', odds)  # Use API odds if available, else DB odds
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
                if predicted_outcome == result['winner']:
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
