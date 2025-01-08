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
    'England - Premier League': 12325,
    'England - Championship': 12326,
    'England - League One': 12327,
    'England - League Two': 12422,
    'England - FA Cup': 12328,
    'England - EFL Cup': 12329,
    'England - Premier League 2 Division One': 13293,
    'England - Premier League' : 12446,
    'England - FA Womens Super League' : 13624,
    'England - FA Womens Championship' : 13625,
    
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
    'Germany - DFB-Pokal': 12454,
    
    # France
    'France - Ligue 1': 12377,
    'France - Ligue 2': 12378,
    'France - Coupe de France': 12379,
    
    # Netherlands
    'Netherlands - Eredivisie': 12335,
    
    # Belgium
    'Belgium - Jupiler Pro League': 12338,
    
    # Scotland
    'Scotland - Premiership': 12548,
    'Scotland - Championship': 12549,
    
    # Turkey
    'Turkey - Super Lig': 12928,
    'Turkey - 1. Lig': 12929,
    
    # Russia
    'Russia - Premier League': 12431,
    
    # Portugal
    'Portugal - Primeira Liga': 12344,
    
    # Switzerland
    'Switzerland - Super League': 12482,
    
    # Austria
    'Austria - Bundesliga': 12419,
    
    # Greece
    'Greece - Super League 1': 12501,
    'Greece - Super League 2': 12502,
    
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
    'Mexico - Liga MX': 12384,
    
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
    
    # European Competitions
    'Europe - Champions League': 12301,
    'Europe - Europa League': 12302,
    'Europe - Conference League': 12303,
    
    # Other Continental Competitions
    'South America - Copa Libertadores': 12304,
    'Asia - AFC Champions League': 12305,
    'North America - CONCACAF Champions': 12306
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
            if (all(key in match for key in ['home_name', 'away_name', 'competition_id']) and 
                match['competition_id'] in league_ids):
                valid_matches.append(match)
        
        print(f"Found {len(valid_matches)} valid matches in selected leagues")
        
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
    # This function should be implemented to get results from your local database
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
    # This function should be implemented to find matches from your local database
    return None
