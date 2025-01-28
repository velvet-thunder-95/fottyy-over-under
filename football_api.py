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


    'England - Championship': 12451,
    'England - FA Cup': 13698,
    'England - EFL League One': 12446,
    'England - Premier League ': 12325,  
    'England - FA Womens Super League': 12827,      
    'England - FA Womens Championship': 12802,

    # Unknown leagues

    'Colombia - Categoria Primera A': 14086,
    'Croatia - Prva HNL':12121,

    # European Competitions
    'UEFA Champions League': 12321,
    'UEFA Europa League':12327,
    'UEFA Europa Conference League': 12278,
    'Europe - UEFA Youth League' :13497,

    # Asia uropean Competitions
    'AFC Champions League': 13356,

    
    # International Competitions
    'UEFA Euro Qualifiers': 12308,
    'UEFA Nations League': 13734,
    'Africa Cup of Nations': 12310,
    'WC Qualification Europe': 12311,
    

    # Spain
    'Spain - La Liga': 12316,
    'Spain - Segunda División': 12467,
    
    # Italy
    'Italy - Serie A': 12530,
    'Italy - Serie B': 12621,
    'Italy - Coppa Italia': 12579,

    # Germany
    'Germany - Bundesliga': 12529,
    'Germany 2. Bundesliga': 12528,
    'Germany - DFB-Pokal': 12057,
    'Germany - 3. Liga': 12623,
    
    # France
    'France - Ligue 1': 12337, 
    'France - Ligue 2': 12338,
    
    # Netherlands
    'Netherlands - Eredivisie': 12322,



    # Belgium
    'Belgium Pro League': 12137,
    
    # Scotland
    'Scotland - Premiership': 12455,
    'Scotland - Championship': 12456,
    
    # Turkey
    'Turkey - 1. Lig': 12928,
    'Turkey Süper Lig': 12641,


    # Russia
    'Russia - Premier League': 12335,
    
    # Portugal
    'Portugal - Primeira Liga': 12931,
    
    # Switzerland
    'Switzerland - Super League': 12326,
    
    # Austria
    'Austria - Bundesliga': 12472,
    
    # Greece
    'Greece - Super League 1': 12734,
    'Greece - Super League 2': 13694, 

    # Czech Republic
    'Czech Republic - First League': 12336, 
    
    # Poland
    'Poland - Ekstraklasa': 12120,
    
    # Denmark
    'Denmark - Superliga': 12132,
    
    # Sweden
    'Sweden - Allsvenskan': 13963,
    
    # Brazil
    'Brazil - Serie A': 11351,
    
    # Argentina
    'Argentina - Primera Division': 11212,
    
    # Chile
    'Chile - Primera Division': 12364,
    
    # Uruguay
    'Uruguay - Primera Division': 12365,
    
    # Colombia
    'Colombia - Primera A': 12366,
    
    # Mexico
    'Mexico - Liga MX': 12136,
    
    
    # USA
    'USA - MLS': 13973,
    
    # China
    'China - Super League': 12506,
    
    # Japan
    'Japan - J-League': 13960,   
    
    # Australia
    'Australia - A-League': 13703,
    
    # Saudi Arabia
    'Saudi Arabia - Pro League': 12772,
    
    # Ukraine
    'Ukraine - Premier League': 12483,
    
    # Finland
    'Finland - Veikkausliiga': 12397,
    
    # Other Continental Competitions
    'South America - Copa Libertadores': 13974,
    'CONCACAF Champions League': 13925,
    
    
    # Israel
    'Italy - Serie B': 12621,
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
            'timezone': 'Europe/Berlin'  # Using German timezone
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
        
        # If we get a 404, it's likely a future match
        if response.status_code == 404:
            return {
                "status": "SCHEDULED",
                "home_score": None,
                "away_score": None
            }
            
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get("success"):
            return None
            
        match_data = data.get("data", {})
        
        # Return formatted match result
        return {
            "status": "FINISHED" if match_data.get("status") == "complete" else match_data.get("status", "SCHEDULED"),
            "home_score": match_data.get("home_score", None),
            "away_score": match_data.get("away_score", None)
        }
    except requests.exceptions.RequestException as e:
        print(f"Error getting match result: {str(e)}")
        return {
            "status": "SCHEDULED",
            "home_score": None,
            "away_score": None
        }


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
        match_date = datetime.strptime(date_str, '%Y-%m-%d')
        
        # If the match is in the future, return a scheduled status
        if match_date > datetime.now():
            return {
                "id": None,
                "status": "SCHEDULED",
                "home_score": None,
                "away_score": None,
                "home_team": home_team,
                "away_team": away_team,
                "match_date": date_str
            }
        
        # Make API request to get matches for the date
        url = f"{BASE_URL}/matches"
        params = {
            "key": API_KEY,
            "date_from": date_str,
            "date_to": date_str
        }
        
        response = requests.get(url, params=params)
        
        # If we get a 404, return scheduled status
        if response.status_code == 404:
            return {
                "id": None,
                "status": "SCHEDULED",
                "home_score": None,
                "away_score": None,
                "home_team": home_team,
                "away_team": away_team,
                "match_date": date_str
            }
            
        response.raise_for_status()
        data = response.json()
        
        if not data.get("success"):
            return None
            
        matches = data.get("data", [])
        
        # Find the specific match with matching teams
        for match in matches:
            if (match.get("home_team") == home_team and 
                match.get("away_team") == away_team):
                return {
                    "id": match.get("id"),
                    "status": "FINISHED" if match.get("status") == "complete" else match.get("status", "SCHEDULED"),
                    "home_score": match.get("home_score"),
                    "away_score": match.get("away_score"),
                    "home_team": home_team,
                    "away_team": away_team,
                    "match_date": date_str
                }
        
        # If match not found, return scheduled status
        return {
            "id": None,
            "status": "SCHEDULED",
            "home_score": None,
            "away_score": None,
            "home_team": home_team,
            "away_team": away_team,
            "match_date": date_str
        }
        
    except requests.exceptions.RequestException as e:
        print(f"Error finding match by teams: {str(e)}")
        return {
            "id": None,
            "status": "SCHEDULED",
            "home_score": None,
            "away_score": None,
            "home_team": home_team,
            "away_team": away_team,
            "match_date": date_str
        }
