"""
Odds Fetcher - Fetches odds data from Supabase football_odds table

This module provides functions to fetch odds data from the Supabase football_odds table
and integrate it with the existing odds system in the application.
"""

import logging
from supabase_db import SupabaseDB
from unidecode import unidecode

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OddsFetcher:
    def __init__(self):
        """Initialize the OddsFetcher with a Supabase connection"""
        self.supabase = SupabaseDB().supabase
        logger.info("OddsFetcher initialized with Supabase connection")
    
    def normalize_team_name(self, team_name):
        """Normalize team name for comparison by removing accents and converting to lowercase"""
        if not team_name:
            return ""
        return unidecode(team_name).lower().strip()
    
    def get_odds_from_db(self, home_team, away_team, league_name=None):
        """
        Fetch odds from the football_odds table for a specific match
        
        Args:
            home_team (str): Home team name
            away_team (str): Away team name
            league_name (str, optional): League name to further filter results
            
        Returns:
            dict: Dictionary containing odds data if found, None otherwise
        """
        try:
            # Normalize team names for comparison
            normalized_home = self.normalize_team_name(home_team)
            normalized_away = self.normalize_team_name(away_team)
            
            logger.info(f"Fetching odds for {normalized_home} vs {normalized_away}")
            
            # Build the query
            query = self.supabase.table('football_odds')
            
            # Try to match by normalized team names
            # We'll fetch all records and do the matching in Python
            # as Supabase doesn't support complex string operations
            result = query.select('*').execute()
            
            if not result.data:
                logger.info("No odds data found in the database")
                return None
            
            # Find the match by comparing normalized team names
            for odds in result.data:
                db_home = self.normalize_team_name(odds.get('team1'))
                db_away = self.normalize_team_name(odds.get('team2'))
                db_league = self.normalize_team_name(odds.get('league_name'))
                
                # Check if team names match
                teams_match = (db_home == normalized_home and db_away == normalized_away)
                
                # If league name is provided, also check if it matches
                league_match = True
                if league_name:
                    normalized_league = self.normalize_team_name(league_name)
                    league_match = (normalized_league in db_league)
                
                if teams_match and league_match:
                    logger.info(f"Found odds for {home_team} vs {away_team}")
                    
                    # Convert odds strings to floats
                    return {
                        'home_odds': float(odds.get('home_odds', 0)),
                        'draw_odds': float(odds.get('draw_odds', 0)),
                        'away_odds': float(odds.get('away_odds', 0)),
                        'over25_odds': float(odds.get('over_odds', 0)),
                        'under25_odds': float(odds.get('under_odds', 0)),
                        'btts_yes_odds': float(odds.get('btts_yes', 0)),
                        'btts_no_odds': float(odds.get('btts_no', 0)),
                        'source': 'supabase'
                    }
            
            logger.info(f"No matching odds found for {home_team} vs {away_team}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching odds from database: {str(e)}")
            return None
    
    def get_leagues_with_odds(self):
        """
        Get a list of leagues that have odds data in the database
        
        Returns:
            list: List of league names with odds data
        """
        try:
            result = self.supabase.table('football_odds').select('league_name').execute()
            
            if not result.data:
                return []
            
            # Extract unique league names
            leagues = set()
            for odds in result.data:
                league = odds.get('league_name')
                if league:
                    leagues.add(league)
            
            return list(leagues)
            
        except Exception as e:
            logger.error(f"Error fetching leagues with odds: {str(e)}")
            return []
    
    def convert_odds_to_probabilities(self, odds_data):
        """
        Convert odds to probabilities
        
        Args:
            odds_data (dict): Dictionary containing odds data
            
        Returns:
            dict: Dictionary containing probabilities
        """
        if not odds_data:
            return None
        
        try:
            # Extract odds
            home_odds = odds_data.get('home_odds', 0)
            draw_odds = odds_data.get('draw_odds', 0)
            away_odds = odds_data.get('away_odds', 0)
            
            # Ensure odds are valid
            if home_odds <= 1 or draw_odds <= 1 or away_odds <= 1:
                logger.warning("Invalid odds values")
                return None
            
            # Convert to raw probabilities
            home_prob = 1 / home_odds
            draw_prob = 1 / draw_odds
            away_prob = 1 / away_odds
            
            # Calculate the overround (margin)
            total_prob = home_prob + draw_prob + away_prob
            
            # Normalize probabilities to remove the margin
            if total_prob > 0:
                home_prob = home_prob / total_prob
                draw_prob = draw_prob / total_prob
                away_prob = away_prob / total_prob
            
            return {
                'home_prob': home_prob,
                'draw_prob': draw_prob,
                'away_prob': away_prob,
                'source': 'supabase'
            }
            
        except Exception as e:
            logger.error(f"Error converting odds to probabilities: {str(e)}")
            return None


# Simple test function to verify the module works
def test_odds_fetcher():
    fetcher = OddsFetcher()
    
    # Test getting odds for a match
    odds = fetcher.get_odds_from_db("Brighton", "Liverpool")
    print(f"Odds for Brighton vs Liverpool: {odds}")
    
    # Test getting leagues with odds
    leagues = fetcher.get_leagues_with_odds()
    print(f"Leagues with odds: {leagues}")
    
    # Test converting odds to probabilities
    if odds:
        probs = fetcher.convert_odds_to_probabilities(odds)
        print(f"Probabilities: {probs}")

if __name__ == "__main__":
    test_odds_fetcher()
