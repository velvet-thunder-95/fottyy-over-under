import requests
import time
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API configuration
HEADERS = {
    "x-rapidapi-host": "transfermarket.p.rapidapi.com",
    "x-rapidapi-key": "9a7723d114mshe44a60d17ffc5e8p1d348djsncb88cc895980"
}
BASE_URL = "https://transfermarket.p.rapidapi.com"

# League IDs mapping with actual league names
LEAGUE_IDS = {
    # International Competitions
    "AFC Champions League": "AFC",
    "Europe - UEFA Champions League": "CL",
    "Europe - UEFA Europa League": "EL",
    "Europe - UEFA Europa Conference League": "UECL",
    "Europe - UEFA Womens Champions League": "UWCL",
    "Europe - UEFA Youth League": "UYL",
    "Europe - UEFA Womens Nations League": "UWNL",
    "UEFA Nations League": "UNL",
    "South America - Copa Libertadores": "LIBER",
    "South America - Copa Sudamericana": "SUDA",

    # England
    "England - Premier League": "GB1",
    "England - Championship": "GB2",
    "England - EFL League One": "GB3",
    "England - EFL League Two": "GB4",
    "England - FA Cup": "GBFA",
    "England - FA Womens Super League": "GBWS",

    # Germany
    "Germany - Bundesliga": "L1",
    "Germany 2. Bundesliga": "L2",
    "Germany - 3. Liga": "L3",
    "Germany - DFB-Pokal": "DFB",
    "Germany - Frauen Bundesliga": "L1F",

    # Spain
    "Spain - La Liga": "ES1",
    "Spain - Segunda División": "ES2",
    "Spain - Copa del Rey": "CDR",
    "Spain - Primera Division Women": "ESP1F",

    # Italy
    "Italy - Serie A": "IT1",
    "Italy - Serie B": "IT2",
    "Italy - Coppa Italia": "CIT",
    "Italy - Serie A Women": "IT1F",

    # France
    "France - Ligue 1": "FR1",
    "France - Ligue 2": "FR2",
    "France - Coupe de France": "FRC",
    "France - Feminine Division 1": "FR1F",

    # Other Major European Leagues
    "Netherlands - Eredivisie": "NL1",
    "Netherlands - KNVB Cup": "NLBK",
    "Portugal - Primeira Liga": "PO1",
    "Portugal - LigaPro": "PO2",
    "Portugal - Liga NOS": "PO1",
    "Belgium Pro League": "BE1",
    "Belgium - First Division B": "BE2",
    "Belgium - Belgian Cup": "BEC",
    "Austria - Bundesliga": "A1",
    "Switzerland - Super League": "C1",
    "Switzerland - Challenge League": "C2",
    "Switzerland - Swiss Cup": "SC",
    "Switzerland - Nationalliga A Women": "C1F",

    # Eastern European Leagues
    "Ukraine - Premier League": "UKR1",
    "Czech Republic - First League": "TS1",
    "Czech Republic - FNL": "TS2",
    "Czech Republic - Czech Cup": "TSC",
    "Croatia - Prva HNL": "KR1",
    "Poland - Ekstraklasa": "PL1",
    "Romania - Liga 1": "RO1",
    "Bulgaria - First League": "BU1",
    "Greece - Super League 1": "GR1",
    "Serbia - Super Liga": "SB1",
    "Slovakia - Super Liga": "SK1",
    "Slovakia - Slovakia Cup": "SKC",
    "Slovenia - Prva Liga": "SV1",

    # Northern European Leagues
    "Scotland - Premiership": "SC1",
    "Norway - Eliteserien": "NO1",
    "Sweden - Allsvenskan": "SE1",
    "Sweden - Superettan": "SE2",
    "Denmark - Superliga": "DK1",
    "Iceland - Úrvalsdeild": "IS1",
    "Estonia - Meistriliiga": "EST1",
    "Lithuania - A Lyga": "LT1",
    "Northern Ireland - NIFL Premiership": "NIR1",
    "Wales - Welsh Premier League": "WAL1",

    # South American Leagues
    "Argentina - Primera Division": "AR1N",
    "Argentine football league - Primera Nacional": "AR2",
    "Brazil - Serie A": "BRA1",
    "Brazil - Serie B": "BRA2",
    "Colombia - Categoria Primera A": "COL1",
    "Colombia - Categoria Primera B": "COL2",
    "Paraguay - Division Profesional": "PAR1",
    "Peru - Primera División": "PE1",
    "Uruguay - Primera Division": "UY1",
    "Venezuela - Primera División": "VE1",
    "Ecuador - Primera Categoría Serie A": "EC1",
    "Ecuador - Primera Categoría Serie B": "EC2",
    "Bolivia - LFPB": "BOL1",

    # North/Central American Leagues
    "USA - MLS": "MLS1",
    "Mexico - Liga MX": "MEX1",
    "Honduras - Liga Nacional de Fútbol Profesional de Honduras": "HON1",
    "Panama - LPF": "PAN1",

    # Asian Leagues
    "Japan - J1 League": "J1",
    "Japan - J-League": "J1",
    "South Korea - K League 1": "RSK1",
    "China - Super League": "CSL",
    "Saudi Arabia - Pro League": "SA1",
    "Saudi Arabia - Professional League": "SA1",
    "UAE - Arabian Gulf League": "UAE1",
    "Qatar - Stars League": "QAT1",
    "Iran - Persian Gulf Pro League": "IR1",
    "Uzbekistan - Uzbekistan Super League": "UZ1",
    "Vietnam - V.League 1": "V1",
    "Indonesia - Liga 1": "IN1",
    "Thailand - Thai League T1": "T1",
    "Singapore - S.League": "SIN1",

    # African Leagues
    "Algeria - Ligue 1": "ALG1",
    "Egypt - Egyptian Premier League": "EGY1",
    "Morocco - Botola Pro": "MAR1",
    "Nigeria - NPFL": "NGA1",
    "South Africa - Premier Soccer League": "RSA1",

    # Other Leagues
    "Australia - A-League": "AUS1",
    "Armenia - Armenian Premier League": "ARM1",
    "Azerbaijan - Premyer Liqası": "AZ1",
    "Bosnia and Herzegovina - Premier League of Bosnia": "BOS1",
    "Cyprus - First Division": "CYP1",
    "Georgia - Erovnuli Liga": "GEO1",
    "Hungary - NB I": "HUN1",
    "Jordan - Jordanian Pro League": "JOR1",
    "Malta - Premier League": "MAL1",
    "Turkey Süper Lig": "TR1",
    "Turkey - Turkish Cup": "TRC"
}

def search_league(league_name):
    """Search for a league and return its ID"""
    # Try direct match first
    if league_name in LEAGUE_IDS:
        return LEAGUE_IDS[league_name]
    
    # Try without country prefix
    if " - " in league_name:
        short_name = league_name.split(" - ")[1]
        for key, value in LEAGUE_IDS.items():
            if short_name.lower() in key.lower():
                return value
    
    # Try with different variations
    variations = [
        league_name,
        league_name.replace(" - ", " "),
        league_name.replace("&", "and"),
        league_name.split(" - ")[-1] if " - " in league_name else None,
        league_name.split(" - ")[0] if " - " in league_name else None,
        # Add more specific variations
        league_name.replace("Premier League", "").strip(),
        league_name.replace("First Division", "").strip(),
        league_name.replace("Liga", "").strip(),
        league_name.replace("League", "").strip(),
        league_name.replace("Division", "").strip(),
        league_name.replace("Championship", "").strip(),
        league_name.replace("Super", "").strip(),
        league_name.replace("Professional", "").strip(),
        league_name.replace("Pro", "").strip()
    ]
    
    # Try each variation
    for variation in variations:
        if variation:
            # Try exact match first
            if variation in LEAGUE_IDS:
                return LEAGUE_IDS[variation]
            
            # Try case-insensitive partial match
            for key, value in LEAGUE_IDS.items():
                if variation.lower() in key.lower():
                    return value
                # Try matching country code at start
                if key.startswith(variation.split(" ")[0]):
                    return value
    
    return None

def get_teams_by_league_id(league_id):
    """Get teams for a league using its ID"""
    url = f"{BASE_URL}/search"
    teams = []
    
    try:
        # Try different search variations
        search_variations = [
            league_id,  # Original league ID
            league_id.lower(),  # Lowercase
            league_id.upper(),  # Uppercase
            league_id.replace("1", ""), # Without number
            league_id.replace("2", ""), # Without number
            f"{league_id} league", # With league suffix
            f"{league_id} division" # With division suffix
        ]
        
        for query in search_variations:
            params = {
                "query": query,
                "domain": "com"
            }
            
            response = requests.get(url, headers=HEADERS, params=params)
            if response.status_code == 429:  # Rate limit hit
                logger.info("Rate limit hit, waiting 60 seconds...")
                time.sleep(60)
                response = requests.get(url, headers=HEADERS, params=params)
            elif response.status_code == 503:  # Service unavailable
                logger.info("Service temporarily unavailable, waiting 30 seconds...")
                time.sleep(30)
                continue
                
            response.raise_for_status()
            data = response.json()
            
            # Extract teams from the response
            if "clubs" in data:
                new_teams = [{
                    "name": team["name"],
                    "id": team.get("id"),
                    "country": team.get("area", {}).get("name"),
                    "league_id": league_id
                } for team in data["clubs"] if team.get("name")]
                
                # Add only unique teams
                for team in new_teams:
                    if not any(t["name"] == team["name"] for t in teams):
                        teams.append(team)
                        
            elif "teams" in data:
                new_teams = [{
                    "name": team["name"],
                    "id": team.get("id"),
                    "country": team.get("area", {}).get("name"),
                    "league_id": league_id
                } for team in data["teams"] if team.get("name")]
                
                # Add only unique teams
                for team in new_teams:
                    if not any(t["name"] == team["name"] for t in teams):
                        teams.append(team)
            
            # If we found enough teams, stop searching
            if len(teams) >= 10:
                break
                
            # Rate limiting between variations
            time.sleep(1)
                
        return teams
    except Exception as e:
        logger.error(f"Error getting teams for league ID {league_id}: {e}")
        return []

def main():
    """Main function to collect teams for all leagues"""
    # Read league names from file
    with open('league_names.txt', 'r') as f:
        league_names = [line.strip() for line in f if line.strip()]
    
    # Dictionary to store all leagues and their teams
    all_leagues = {}
    
    # Try to load existing data if any
    try:
        with open('all_leagues_teams.json', 'r', encoding='utf-8') as f:
            all_leagues = json.load(f)
            logger.info(f"Loaded {len(all_leagues)} existing leagues")
    except FileNotFoundError:
        logger.info("Starting fresh collection")
    
    # Process each league
    total_leagues = len(league_names)
    for idx, league_name in enumerate(league_names, 1):
        logger.info(f"\nProcessing league {idx}/{total_leagues}: {league_name}")
        
        # Skip if we've already processed this league
        if league_name in all_leagues:
            logger.info(f"Already have results for {league_name}, skipping...")
            continue
        
        # Search for league ID
        league_id = search_league(league_name)
        if not league_id:
            logger.warning(f"No league ID found for {league_name}")
            continue
            
        # Get teams for this league
        teams = get_teams_by_league_id(league_id)
        if teams:
            logger.info(f"Found {len(teams)} teams for {league_name}")
            all_leagues[league_name] = {
                "id": league_id,
                "teams": teams
            }
            
            # Save progress after each league
            with open('all_leagues_teams.json', 'w', encoding='utf-8') as f:
                json.dump(all_leagues, f, indent=2, ensure_ascii=False)
                logger.info(f"Progress saved: {idx}/{total_leagues} leagues processed")
        else:
            logger.warning(f"No teams found for {league_name}")
        
        # Rate limiting between leagues
        time.sleep(2)
    
    logger.info("\nFinished processing all leagues")
    logger.info(f"Total leagues processed: {len(all_leagues)}")
    logger.info("Results saved to all_leagues_teams.json")

if __name__ == '__main__':
    main()
