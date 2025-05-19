

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
            
        # Remove common prefixes
        name = team_name
        prefixes = ['IF ', 'FC ', 'CD ', 'CA ', 'IFK ', 'UMF ']
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
        
        # Remove common suffixes that might cause mismatches
        suffixes = [' FC', ' IF', ' AIF', ' United', ' City', ' CF', ' UBK', ' Jrs.']
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        
        # Remove accents and convert to lowercase
        normalized = unidecode(name).lower().strip()
        
        # Log the normalization for debugging
        if name != team_name:
            logger.info(f"Normalized team name: '{team_name}' -> '{name}' -> '{normalized}'")
        else:
            logger.info(f"Normalized team name: '{team_name}' -> '{normalized}'")
            
        return normalized
    
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
            logger.info(f"Searching for match in {len(result.data)} database records")
            logger.info(f"Looking for: Home='{normalized_home}', Away='{normalized_away}', League='{league_name}'")
            
            for odds in result.data:
                db_home = self.normalize_team_name(odds.get('team1'))
                db_away = self.normalize_team_name(odds.get('team2'))
                db_league = self.normalize_team_name(odds.get('league_name'))
                
                logger.info(f"DB Record: Home='{db_home}', Away='{db_away}', League='{db_league}'")
                
                # Check if team names match - use more flexible matching with special cases for Swedish teams
                
                # Special cases for Swedish teams
                swedish_team_mappings = {
                    'djurgarden': ['djurgardens', 'djurgarden', 'djurgårdens', 'djurgården', 'djurgårdens if', 'djurgården', 'dju'],
                    'norrkoping': ['norrkoping', 'norrköping', 'ifk norrkoping', 'ifk norrköping', 'nor'],
                    'goteborg': ['goteborg', 'göteborg', 'ifk goteborg', 'ifk göteborg', 'ifk'],
                    'elfsborg': ['elfsborg', 'if elfsborg', 'elf'],
                    'mjallby': ['mjallby', 'mjällby', 'mjallby aif', 'mjällby aif', 'mjä'],
                    'brommapojkarna': ['brommapojkarna', 'if brommapojkarna', 'bro'],
                    'sirius': ['sirius', 'ik sirius', 'sir'],
                    'degerfors': ['degerfors', 'degerfors if', 'deg']
                }
                
                logger.info(f"Checking team matches for {normalized_home} vs {normalized_away}")
                
                # Check for special case matches
                home_match = False
                for key, variations in swedish_team_mappings.items():
                    if normalized_home in variations or any(var in normalized_home for var in variations):
                        if db_home in variations or any(var in db_home for var in variations):
                            home_match = True
                            logger.info(f"Special case match found for home team: {normalized_home} matches with {db_home} via {key}")
                            break
                
                away_match = False
                for key, variations in swedish_team_mappings.items():
                    if normalized_away in variations or any(var in normalized_away for var in variations):
                        if db_away in variations or any(var in db_away for var in variations):
                            away_match = True
                            logger.info(f"Special case match found for away team: {normalized_away} matches with {db_away} via {key}")
                            break
                
                # If no special case match, try standard matching
                if not home_match:
                    home_match = db_home == normalized_home or normalized_home in db_home or db_home in normalized_home
                    if home_match:
                        logger.info(f"Standard match found for home team: {normalized_home} matches with {db_home}")
                
                if not away_match:
                    away_match = db_away == normalized_away or normalized_away in db_away or db_away in normalized_away
                    if away_match:
                        logger.info(f"Standard match found for away team: {normalized_away} matches with {db_away}")
                
                teams_match = home_match and away_match
                
                logger.info(f"Team matching: home_match={home_match}, away_match={away_match}")
                
                # If league name is provided, also check if it matches
                league_match = True
                if league_name:
                    normalized_league = self.normalize_team_name(league_name)
                    
                    # Handle special cases for league names
                    if 'allsvenskan' in normalized_league and 'allsvenskan' in db_league:
                        logger.info(f"League match found: Both contain 'allsvenskan'")
                        league_match = True
                    elif 'sweden' in normalized_league and 'sweden' in db_league:
                        logger.info(f"League match found: Both contain 'sweden'")
                        league_match = True
                    # Extract country name for comparison
                    elif ' - ' in normalized_league and ',' in db_league:
                        # Extract country from "Country - League" format
                        country = normalized_league.split(' - ')[0].strip().lower()
                        # Extract country from "League, Country" format
                        db_country = db_league.split(',')[1].strip().lower() if ',' in db_league and len(db_league.split(',')) > 1 else ''
                        league_match = (country in db_country or db_country in country)
                    # More flexible league matching - check if one contains the other
                    elif normalized_league in db_league or db_league in normalized_league:
                        league_match = True
                    else:
                        league_match = False
                        
                    logger.info(f"League comparison: '{normalized_league}' vs '{db_league}' = {league_match}")
                    
                        # Special case for Swedish Allsvenskan
                    if not league_match and ('allsvenskan' in normalized_league or 'sweden' in normalized_league):
                        logger.info(f"Checking special case for Swedish Allsvenskan")
                        if 'allsvenskan' in db_league or 'sweden' in db_league:
                            logger.info(f"Special case match found for Swedish Allsvenskan")
                            league_match = True
                    
                    # Always match Swedish Allsvenskan if both teams are Swedish
                    if not league_match and home_match and away_match:
                        for key in swedish_team_mappings.keys():
                            if (key in normalized_home or key in db_home) and (key in normalized_away or key in db_away):
                                if 'allsvenskan' in db_league or 'sweden' in db_league:
                                    logger.info(f"Forced league match for Swedish teams in Allsvenskan")
                                    league_match = True
                                    break
                
                logger.info(f"Match result: teams_match={teams_match}, league_match={league_match}")
                
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
