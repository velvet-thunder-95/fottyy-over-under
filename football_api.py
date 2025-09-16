# football_api.py

import requests
from datetime import datetime, timedelta
import json
import streamlit as st

# At the top of app.py
print("Loading app.py")
import sys
print(f"Python path: {sys.path}")

# Get API key from Streamlit secrets
API_KEY = st.secrets["football_api"]["api_key"]

# Base URL for the Football Data API
BASE_URL = "https://api.football-data-api.com"

# Define all league IDs we want to fetch
LEAGUE_IDS = {


    #Ecuador
    'Ecuador - Primera Categoría Serie A': 11329,
    'Ecuador - Primera Categoría Serie B': 11329,

    # Uruguay leagues
    'Uruguay - Primera Division': 11209,

    # Ukraine leagues
    'Ukraine - Premier League': 14951,

    # Uganda leagues
    'Uganda - Premier League': 13320,

    # Honduras leagues
    'Honduras - Liga Nacional de Fútbol Profesional de Honduras': 12650,

    # Wales leagues
    'Wales - Welsh Premier League': 15226,

    # USA leagues
    'USA - MLS': 13973,

    # UAE leagues
    'UAE - Arabian Gulf League': 12584,

    # Azerbaijan leagues
    'Azerbaijan - Premyer Liqası': 12740,

    # Estonia leagues
    'Estonia - Meistriliiga': 11119,

    # Turkey leagues
    'Turkey - 1. Lig': 12928,
    'Turkey Süper Lig': 14972,
    'Turkey - Turkish Cup': 13549,

    # Switzerland leagues
    'Switzerland - Super League': 15047,
    'Switzerland - Challenge League': 14974,
    'Switzerland - Swiss Cup': 12688,
    'Switzerland - Nationalliga A Women': 12596,

    # Sweden leagues
    'Sweden - Allsvenskan': 13963,
    'Sweden - Superettan': 13975,

    # Spain leagues
    'Spain - La Liga': 12316,
    'Spain - Copa del Rey': 13624,
    'Spain - Segunda División': 12467,
    'Spain - Primera Division Women': 13336,

    # South Korea leagues
    'South Korea - K League 1': 26861,

    # Uzbekistan leagues
    'Uzbekistan - Uzbekistan Super League': 14186,

    # Venezuela leagues
    'Venezuela - Primera División': 11063,

    # Vietnam leagues
    'Vietnam - V.League 1': 13152,
    
    # South America leagues
    'South America - Copa Libertadores': 13974,
    'South America - Copa Sudamericana': 13965,

    # South Africa leagues
    'South Africa - Premier Soccer League': 13284,

    # Singapore leagues
    'Singapore - S.League': 11495,

    # Thailand leagues
    'Thailand - Thai League T1': 12475,

    # Slovenia leagues
    'Slovenia - Prva Liga': 12476,

    # Paraguay leagues
    'Paraguay - Division Profesional': 13927,

    # Peru leagues
    'Peru - Primera División': 11062,

    # Northern Ireland leagues
    'Northern Ireland - NIFL Premiership': 12417,

    # Morocco leagues
    'Morocco - Botola Pro': 13286,

    # Nigeria leagues
    'Nigeria - NPFL': 12762,

    # Panama leagues
    'Panama - LPF': 13685,

    # Slovakia leagues
    'Slovakia - Super Liga': 12944,
    'Slovakia - Slovakia Cup 2': 12687,
    'Slovakia - Slovakia Cup': 15168,

    # Cyprus leagues
    'Cyprus - First Division': 12541,

    # Lithuania leagues
    'Lithuania - A Lyga': 14291,

    # Malta leagues
    'Malta - Premier League': 12538,

    # Iran leagues
    'Iran - Persian Gulf Pro League': 12915,

    # Jordan leagues
    'Jordan - Jordanian Pro League': 15248,

    # Scotland leagues
    'Scotland - Premiership': 12455,
    'Scotland - Championship': 12456,

    # Saudi Arabia leagues
    'Saudi Arabia - Pro League': 12772,
    'Saudi Arabia - Professional League': 12772,

    # Romania leagues
    'Romania - Liga 1': 15015,

    # Qatar leagues
    'Qatar - Stars League': 12587,

    # Portugal leagues
    'Portugal - Primeira Liga': 12931,
    'Portugal - Liga NOS': 15115,
    'Portugal - Portuguese Super Cup': 12050,
    'Portugal - LigaPro': 12585,
    
    # Poland leagues
    'Poland - Ekstraklasa': 15031,

    # Norway leagues
    'Norway - Eliteserien': 13987,

    # Netherlands leaguea
    'Netherlands - Eredivisie': 12322,
    'Netherlands - KNVB Cup': 12799,

    # Mexico leagues
    'Mexico - Liga MX': 12136,
    
    # Japan
    'Japan - J-League': 13960,  
    'Japan - J1 League': 13960, 

    # Italy
    'Italy - Serie A': 12530,
    'Italy - Serie B': 12621,
    'Italy - Coppa Italia': 12579,
    'Italy - Serie A Women': 12586,

    # Israel
    'Italy - Serie B': 12621,

    # International Competitions
    'UEFA Euro Qualifiers': 12308,
    'UEFA Nations League': 13734,
    'Africa Cup of Nations': 12310,
    'WC Qualification Europe': 12311,

    # Iceland leagues
    'Iceland - Úrvalsdeild': 14017,

    # Hungary leagues
    'Hungary - NB I': 14963,

    # Greece
    'Greece - Super League 1': 12734,
    'Greece - Super League 2': 13694, 

    # Germany
    'Germany - Bundesliga': 12529,
    'Germany 2. Bundesliga': 14931,
    'Germany - DFB-Pokal': 12057,
    'Germany - 3. Liga': 14977,
    'Germany - Frauen Bundesliga': 12934,
    
    # France
    'France - Ligue 1': 12337, 
    'France - Ligue 2': 14954,
    'France - Coupe de France': 13729,
    'France - Feminine Division 1': 12789,

    # Finland
    'Finland - Veikkausliiga': 14089,

    # European Competitions
    'Europe - UEFA Champions League': 14924,
    'Europe - UEFA Europa League': 15002,
    'Europe - UEFA Europa Conference League': 12278,
    'Europe - World Championship':13964,
    'Europe - UEFA Youth League' :13497,
    'Europe - UEFA Womens Champions League': 12536,
    'Europe - UEFA Womens Nations League': 10812,

    # Georgia leagues
    'Georgia - Erovnuli Liga': 11233,

    # England
    'England - Championship': 14930,
    'England - FA Cup': 15238,
    'England - EFL League One': 12446,
    'England - EFL League Two': 12422,
    'England - Premier League ': 12325,  
    'England - FA Womens Super League': 12827,   
    'England - FA Womens Championship': 12802,

    # Czech Republic
    'Czech Republic - First League': 12336,
    'Czech Republic - Czech Cup': 12522,
    'Czech Republic - 1. Liga U19': 12978,
    'Czech Republic - FNL': 15222,

    # Colombian leagues
    'Colombia - Categoria Primera A': 14086,
    'Colombia - Categoria Primera B':14090,

    # Croatian leagues
    'Croatia - Prva HNL': 15053,
    'Croatia - Druga HNL':12471,

    # Argentine leagues
    'Argentine football league - Primera Nacional':14125,

    # Asia uropean Competitions
    'AFC Champions League': 13356,

    # Belgium
    'Belgium Pro League': 14937,
    'Belgium - First Division B': 15196,
    'Belgium - Belgian Cup': 13012,
    
    # Egypt leagues
    'Egypt - Egyptian Premier League': 15544,
    # Russia
    'Russia - Premier League': 12335,
    # Austria
    'Austria - Bundesliga': 12472,
    # Denmark
    'Denmark - Superliga': 15055,
    
    #Bulgaria
    'Bulgaria - First League': 13863,
    'Bulgaria - Second League': 15056,

    # Bolivia leagues
    'Bolivia - LFPB': 11072,
    
    # Brazil
    'Brazil - Serie A': 11351,
    'Brazil - Serie B': 11351,
    
    # Bosnia and Herzegovina
    'Bosnia and Herzegovina - Premier League of Bosnia': 12932,
    
    # Argentina
    'Argentina - Primera Division': 11212,
    
    # Algeria leagues
    'Algeria - Ligue 1': 12665,
    
    # Chile
    'Chile - Primera Division': 14116,
    
    # Armenia
    'Armenia - Armenian Premier League': 12638,

    # Uruguay
    'Uruguay - Primera Division': 12365,
    
    # Colombia leagues
    'Colombia - Primera A': 12366,
    
    # Indonesia leagues
    'Indonesia - Liga 1': 13046,
    
    # China's leagues
    'China - Super League': 12506,
    
    # Australia
    'Australia - A-League': 13703,
    
    # Other Continental Competitions
    'CONCACAF Champions League': 13925,

    # Unknown leagues
    'unknown - league 25 ': 15031,

    
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
    """Get match result from API"""
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
                "away_score": None,
                "winner": None
            }
            
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get("success"):
            print(f"API error for match {match_id}: {data.get('error', 'Unknown error')}")
            return None
            
        match_data = data.get("data", {})
        
        # Normalize status values
        raw_status = match_data.get("status", "").lower()
        if raw_status in ["complete", "finished", "completed"]:
            status = "FINISHED"
        elif raw_status in ["scheduled", "pending"]:
            status = "SCHEDULED"
        else:
            status = raw_status.upper()
        
        home_score = match_data.get("home_score")
        away_score = match_data.get("away_score")
        
        # Determine winner if match is finished
        winner = None
        if status == "FINISHED" and home_score is not None and away_score is not None:
            if home_score > away_score:
                winner = "HOME"
            elif away_score > home_score:
                winner = "AWAY"
            else:
                winner = "DRAW"
        
        # Return formatted match result
        return {
            "status": status,
            "home_score": home_score,
            "away_score": away_score,
            "winner": winner
        }
    except requests.exceptions.RequestException as e:
        print(f"Network error getting match result for {match_id}: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error getting match result for {match_id}: {str(e)}")
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
