# football_api.py

import requests
from datetime import datetime, timedelta
import json


# At the top of app.py
print("Loading app.py")
import sys
print(f"Python path: {sys.path}")



API_KEY = '633379bdd5c4c3eb26919d8570866801e1c07f399197ba8c5311446b8ea77a49'
# Base URL for the Football Data API
BASE_URL = "https://api.football-data-api.com"

# Define all league IDs we want to fetch
LEAGUE_IDS = {
    # England
    'England - Premier League': 12446,  # Current Premier League ID
    'England - Premier League (Alt)': 12325,  # Alternative Premier League ID
    'England - Championship': 12326,
    'England - League One': 12327,
    'England - League Two': 12422,
    'England - FA Cup': 12328,
    'England - EFL Cup': 12329,
    'England - Premier League 2 Division One': 13293,
    'England - FA Womens Super League': 13624,
    'England - FA Womens Championship': 13625,

    # European Competitions
    'UEFA Champions League': 12301,
    'UEFA Europa League': 12302,
    'UEFA Europa Conference League': 12303,
    'AFC Champions League': 12305,
    
    # International Competitions
    'UEFA Euro Qualifiers': 12308,
    'UEFA Nations League': 12309,
    'Africa Cup of Nations': 12310,
    'WC Qualification Europe': 12311,
    'International Friendlies': 12316,  # Added International Friendlies
    
    # Spain
    'Spain - La Liga': 12476,
    'Spain - LaLiga2': 12477,
    'Spain - Copa del Rey': 12478,
    
    # Italy
    'Italy - Serie A': 12447,
    'Italy - Serie B': 12448,
    'Italy - Coppa Italia': 12449,
    
    # Germany
    'Germany - Bundesliga': 12452,
    'Germany - 2. Bundesliga': 12453,
    'Germany 2. Bundesliga': 12528,
    'Germany - DFB-Pokal': 12454,
    'Germany - 3. Liga': 12623,  # Added German 3rd tier
    'Germany - Regional Leagues': 12451,  # Added German Regional Leagues
    
    # France
    'France - Ligue 1': 12377,
    'France - Ligue 2': 12378,
    'France - Coupe de France': 12379,
    'France - National': 12529,  # Added French 3rd tier
    'France - National 2': 12530,  # Added French 4th tier
    
    # Netherlands
    'Netherlands - Eredivisie': 12322,
    'Netherlands - Eerste Divisie': 13698,  # Added Dutch 2nd tier


    # Belgium
    'Belgium - Jupiler Pro League': 12338,
    'Belgium Pro League':12137,
    
    # Scotland
    'Scotland - Premiership': 12455,
    'Scotland - Championship': 12549,
    
    # Turkey
    'Turkey - Super Lig': 12928,
    'Turkey - 1. Lig': 12929,
    'Turkey Süper Lig':12641,


    # Russia
    'Russia - Premier League': 12431,
    
    # Portugal
    'Portugal - Primeira Liga': 12344,
    'Portugal Liga NOS': 12931,
    
    # Switzerland
    'Switzerland - Super League': 12482,
    
    # Austria
    'Austria - Bundesliga': 12419,
    
    # Greece
    'Greece - Super League 1': 12501,
    'Greece - Super League 2': 12502,
    'Greece Super League': 12734,
    
    # Czech Republic
    'Czech Republic - First League': 12426,
    
    # Poland
    'Poland - Ekstraklasa': 12412,
    
    # Denmark
    'Denmark - Superliga': 12339,
    
    # Norway
    'Norway - Eliteserien': 12392,
    
    # Sweden
    'Sweden - Allsvenskan': 12398,
    
    # Brazil
    'Brazil - Brasileirao': 12341,
    
    # Argentina
    'Argentina - Primera Division': 12363,
    
    # Chile
    'Chile - Primera Division': 12364,
    
    # Uruguay
    'Uruguay - Primera Division': 12365,
    
    # Colombia
    'Colombia - Primera A': 12366,
    
    # Mexico
    'Mexico - Liga MX': 12136,
    
    # USA
    'USA - MLS': 12384,
    
    # China
    'China - Super League': 12506,
    
    # Japan
    'Japan - J-League': 12411,
    
    # Australia
    'Australia - A-League': 13703,
    
    # Saudi Arabia
    'Saudi Arabia - Pro League': 12772,
    
    # Ukraine
    'Ukraine - Premier League': 12440,
    
    # Finland
    'Finland - Veikkausliiga': 12397,
    
    # Other Continental Competitions
    'South America - Copa Libertadores': 12304,
    'CONCACAF Champions League': 12306,
    
    # Romania
    'Romania - Liga I': 12467,  # Added Romanian top flight
    
    # Israel
    'Israel - Premier League': 12621,  # Added Israeli top flight
}

print("Loading football_api.py")
print(f"LEAGUE_IDS defined as: {LEAGUE_IDS}")


def get_matches(date_str):
    """Get matches for a specific date"""
    try:
        # Always use todays-matches endpoint with date parameter
        params = {
            'key': API_KEY,
            'date': date_str,
            'timezone': 'Asia/Kolkata'  # Using Indian timezone
        }
        
        url = f"{BASE_URL}/todays-matches"
        print(f"Fetching matches with params: {params}")
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data or 'data' not in data:
            print(f"No data found for date: {date_str}")
            return []
            
        matches = data['data']
        print(f"Found {len(matches)} matches for {date_str}")
        
        # Get the list of league IDs we're interested in
        league_ids = set(LEAGUE_IDS.values())
        
        # Filter matches by league IDs and required fields
        valid_matches = []
        for match in matches:
            print(f"Raw match data: {json.dumps(match, indent=2)}")  # Debug log
            missing_fields = [field for field in ['home_name', 'away_name', 'competition_id'] if field not in match]
            competition_id = match.get('competition_id')
            
            if missing_fields:
                print(f"Match missing required fields: {missing_fields}")
                continue
                
            if competition_id not in league_ids:
                print(f"Match competition_id {competition_id} not in tracked leagues")
                continue
            
            valid_matches.append(match)
        
        print(f"Found {len(valid_matches)} valid matches in selected leagues")
        if len(valid_matches) < len(matches):
            print(f"Filtered out {len(matches) - len(valid_matches)} matches")
        
        # Sort matches by competition and kickoff time
        valid_matches.sort(key=lambda x: (x.get('competition_id', ''), x.get('kickoff', '')))
        
        return valid_matches
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching matches: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return []
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return []


def get_match_result(match_id):
    """Get match result from database instead of API"""
    try:
        # Make API request to get match result
        url = f"{BASE_URL}/match/{match_id}"
        params = {
            "key": API_KEY
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get("success"):
            return None
            
        match_data = data.get("data", {})
        
        # Return formatted match result
        return {
            "status": "FINISHED" if match_data.get("status") == "complete" else match_data.get("status", "SCHEDULED"),
            "home_score": match_data.get("home_score", 0),
            "away_score": match_data.get("away_score", 0)
        }
        
    except Exception as e:
        print(f"Error getting match result: {str(e)}")
        return None


def get_results_by_date(date_str):
    """Get match results from database instead of API"""
    # This function should be implemented to get results from your local database
    return []


def get_team_stats(team_id, league_id=None):
    """Get team statistics from database instead of API"""
    # This function should be implemented to get team stats from your local database
    return None


def get_match_by_teams(home_team, away_team, date_str):
    """Find a match by teams from database instead of API"""
    try:
        # Convert date string to required format (YYYY-MM-DD)
        match_date = datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
        
        # Make API request to get matches for the date
        url = f"{BASE_URL}/matches"
        params = {
            "key": API_KEY,
            "date_from": match_date,
            "date_to": match_date
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get("success"):
            return None
            
        matches = data.get("data", [])
        
        # Find the specific match with matching teams
        for match in matches:
            if (match.get("home_team", "").lower() == home_team.lower() and 
                match.get("away_team", "").lower() == away_team.lower()):
                return {
                    "id": match.get("id"),
                    "home_team": match.get("home_team"),
                    "away_team": match.get("away_team"),
                    "date": match_date,
                    "status": match.get("status", "SCHEDULED")
                }
        
        return None
        
    except Exception as e:
        print(f"Error finding match by teams: {str(e)}")
        return None
