import os
import json
import time
import requests
import logging
from transfermarkt_api import TransfermarktAPI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# League IDs mapping (using Transfermarkt competition IDs)
LEAGUE_IDS = {
    # England
    'England - Premier League': 'GB1',
    'England - Championship': 'GB2',
    'England - League One': 'GB3',
    'England - FA Cup': 'GBFA',
    'England - FA Women\'s Super League': 'GBFS',
    'England - FA Women\'s Championship': 'GB2F',

    # European Competitions
    'UEFA Champions League': 'CL',
    'UEFA Europa League': 'EL',
    'UEFA Europa Conference League': 'UECL',
    'UEFA Youth League': 'UYL',
    'UEFA Nations League': 'NL',
    'UEFA Euro Qualifiers': 'EM',

    # Spain
    'Spain - La Liga': 'ES1',
    'Spain - Segunda División': 'ES2',
    'Spain - Copa del Rey': 'CDR',

    # Germany
    'Germany - Bundesliga': 'L1',
    'Germany - 2. Bundesliga': 'L2',
    'Germany - 3. Liga': 'L3',
    'Germany - DFB-Pokal': 'DFB',

    # Italy
    'Italy - Serie A': 'IT1',
    'Italy - Serie B': 'IT2',
    'Italy - Coppa Italia': 'CIT',

    # France
    'France - Ligue 1': 'FR1',
    'France - Ligue 2': 'FR2',
    'France - Coupe de France': 'FRC',

    # Netherlands
    'Netherlands - Eredivisie': 'NL1',
    'Netherlands - Eerste Divisie': 'NL2',
    'Netherlands - KNVB Beker': 'NLK',

    # Portugal
    'Portugal - Primeira Liga': 'PO1',
    'Portugal - Liga Portugal 2': 'PO2',
    'Portugal - Taça de Portugal': 'POK',

    # Belgium
    'Belgium - Pro League': 'BE1',
    'Belgium - Challenger Pro League': 'BE2',
    'Belgium - Beker van België': 'BEK',

    # Scotland
    'Scotland - Premiership': 'SC1',
    'Scotland - Championship': 'SC2',
    'Scotland - League One': 'SC3',
    'Scotland - League Two': 'SC4',

    # Turkey
    'Turkey - Süper Lig': 'TR1',
    'Turkey - 1. Lig': 'TR2',
    'Turkey - Turkish Cup': 'TRK',

    # Russia
    'Russia - Premier League': 'RU1',
    'Russia - First League': 'RU2',
    'Russia - Russian Cup': 'RUK',

    # Ukraine
    'Ukraine - Premier League': 'UKR1',
    'Ukraine - First League': 'UKR2',

    # Greece
    'Greece - Super League 1': 'GR1',
    'Greece - Super League 2': 'GR2',
    'Greece - Greek Cup': 'GRK',

    # Switzerland
    'Switzerland - Super League': 'C1',
    'Switzerland - Challenge League': 'C2',
    'Switzerland - Swiss Cup': 'CHP',

    # Austria
    'Austria - Bundesliga': 'A1',
    'Austria - 2. Liga': 'A2',
    'Austria - ÖFB Cup': 'OSC',

    # Croatia
    'Croatia - Prva HNL': 'KR1',
    'Croatia - Druga HNL': 'KR2',
    'Croatia - Croatian Cup': 'KRK',

    # Czech Republic
    'Czech Republic - First League': 'TS1',
    'Czech Republic - National Football League': 'TS2',
    'Czech Republic - Czech Cup': 'TSP',

    # Denmark
    'Denmark - Superliga': 'DK1',
    'Denmark - 1st Division': 'DK2',
    'Denmark - DBU Pokalen': 'DKP',

    # Norway
    'Norway - Eliteserien': 'NO1',
    'Norway - OBOS-ligaen': 'NO2',
    'Norway - Norwegian Cup': 'NOK',

    # Sweden
    'Sweden - Allsvenskan': 'SE1',
    'Sweden - Superettan': 'SE2',
    'Sweden - Svenska Cupen': 'SEC',

    # Poland
    'Poland - Ekstraklasa': 'PL1',
    'Poland - I liga': 'PL2',
    'Poland - Polish Cup': 'PLK',

    # Brazil
    'Brazil - Série A': 'BRA1',
    'Brazil - Série B': 'BRA2',
    'Brazil - Copa do Brasil': 'BRC',

    # Argentina
    'Argentina - Primera División': 'AR1N',
    'Argentina - Primera Nacional': 'AR2',
    'Argentina - Copa Argentina': 'ARC',

    # Colombia
    'Colombia - Primera A': 'COL1',
    'Colombia - Primera B': 'COL2',
    'Colombia - Copa Colombia': 'COLC',

    # Chile
    'Chile - Primera División': 'CLPD',
    'Chile - Primera B': 'CL2',
    'Chile - Copa Chile': 'CLC',

    # Mexico
    'Mexico - Liga MX': 'MEX1',
    'Mexico - Liga de Expansión': 'MEX2',
    'Mexico - Copa MX': 'MEXC',

    # USA & Canada
    'USA - MLS': 'MLS1',
    'USA - USL Championship': 'USL',
    'USA - US Open Cup': 'USC',

    # China
    'China - Super League': 'CSL',
    'China - China League One': 'CSL2',
    'China - FA Cup': 'CFA',

    # Japan
    'Japan - J1 League': 'JAP1',
    'Japan - J2 League': 'JAP2',
    'Japan - Emperor\'s Cup': 'JAPC',

    # South Korea
    'South Korea - K League 1': 'RSK1',
    'South Korea - K League 2': 'RSK2',
    'South Korea - FA Cup': 'RSKC',

    # Australia
    'Australia - A-League': 'AUS1',
    'Australia - NPL': 'AUS2',
    'Australia - FFA Cup': 'AUSC',

    # Saudi Arabia
    'Saudi Arabia - Pro League': 'SA1',
    'Saudi Arabia - First Division League': 'SA2',
    'Saudi Arabia - King Cup': 'SAK',

    # Qatar
    'Qatar - Stars League': 'QAT1',
    'Qatar - Second Division': 'QAT2',
    'Qatar - Emir Cup': 'QATC',

    # UAE
    'UAE - Pro League': 'UAE1',
    'UAE - First Division League': 'UAE2',
    'UAE - President\'s Cup': 'UAEC',

    # Continental Competitions
    'AFC Champions League': 'AFCL',
    'AFC Cup': 'AFCC',
    'CAF Champions League': 'CAFCL',
    'CAF Confederation Cup': 'CAFCC',
    'CONCACAF Champions League': 'COCL',
    'Copa Libertadores': 'LIBC',
    'Copa Sudamericana': 'SUDC',
}

class LogoExtractor:
    def __init__(self):
        self.base_url = "https://transfermarket.p.rapidapi.com"
        self.headers = {
            "x-rapidapi-host": "transfermarket.p.rapidapi.com",
            "x-rapidapi-key": "9a7723d114mshe44a60d17ffc5e8p1d348djsncb88cc895980"
        }
        self.logos_dir = os.path.join(os.path.dirname(__file__), 'team_logos')
        self.ensure_logos_directory()
        self.teams_data = {}

    def ensure_logos_directory(self):
        """Create logos directory if it doesn't exist"""
        if not os.path.exists(self.logos_dir):
            os.makedirs(self.logos_dir)
            logger.info(f"Created logos directory at {self.logos_dir}")

    def search_team(self, team_name):
        """Search for a team using the Transfermarkt API"""
        try:
            # Handle special cases for Asian teams
            team_name_map = {
                'Al Ain': 'Al-Ain FC',
                'Al Rayyan': 'Al-Rayyan SC',
                'Al Nassr': 'Al-Nassr FC',
                'Al Hilal': 'Al-Hilal SFC',
                'Al Ittihad': 'Al-Ittihad Club',
                'Al Duhail': 'Al-Duhail SC',
                'Al Fayha': 'Al-Fayha',
                'Al Sadd': 'Al-Sadd SC'
            }
            
            # Map team name if it exists in our mapping
            search_name = team_name_map.get(team_name, team_name)
            
            url = f"{self.base_url}/search"
            params = {
                "query": search_name,
                "types": "club",
                "domain": "de"
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'clubs' in data and data['clubs']:
                # Filter clubs to match the exact league we're looking for
                clubs = data['clubs']
                # Return the first club that's not a youth team
                for club in clubs:
                    if not any(youth_term in club['name'].lower() for youth_term in ['u21', 'u18', 'u19', 'youth', 'jugend']):
                        return club
            
            return None
            
        except Exception as e:
            logger.error(f"Error searching for team {team_name}: {str(e)}")
            return None

    def download_logo(self, url, team_name, league_name):
        """Download logo from URL and save to file"""
        try:
            # Clean team name for filename
            clean_name = f"{league_name}_{team_name}"
            clean_name = "".join(x for x in clean_name if x.isalnum() or x in (' ', '-', '_')).strip()
            filename = f"{clean_name}.png"
            filepath = os.path.join(self.logos_dir, filename)

            # Skip if already downloaded
            if os.path.exists(filepath):
                logger.info(f"Logo already exists for {team_name}")
                return filepath

            # Download logo
            response = requests.get(url)
            response.raise_for_status()

            # Save logo
            with open(filepath, 'wb') as f:
                f.write(response.content)

            logger.info(f"Downloaded logo for {team_name}")
            return filepath

        except Exception as e:
            logger.error(f"Error downloading logo for {team_name}: {str(e)}")
            return None

    def get_league_teams(self, league_id):
        """Get teams from a league"""
        try:
            # For testing, let's use some known teams for each league
            league_teams = {
                # Asian Champions League Teams
                'AFCL': [
                    'Al Ain', 'Al Rayyan', 'Al Nassr', 'Al Hilal', 'Al Ittihad', 'Persepolis', 
                    'Al Duhail', 'Nasaf', 'Navbahor', 'Al Fayha', 'Sepahan', 'Al Sadd',
                    'Foolad', 'Al Quwa Al Jawiya', 'Mumbai City', 'Pakhtakor',
                    'Jeonbuk Hyundai Motors', 'Bangkok United', 'Buriram United', 'Zhejiang',
                    'Shandong Taishan', 'Yokohama F. Marinos', 'Kawasaki Frontale', 'Ulsan Hyundai'
                ],
                
                # Saudi Pro League Teams
                'SA1': [
                    'Al Hilal', 'Al Nassr', 'Al Ittihad', 'Al Ahli', 'Al Taawoun', 'Al Fateh',
                    'Al Fayha', 'Al Riyadh', 'Al Khaleej', 'Al Raed', 'Al Tai', 'Damac',
                    'Al Wehda', 'Al Ettifaq', 'Abha', 'Al Hazem', 'Al Akhdoud', 'Al Shabab'
                ],
                
                # Qatar Stars League Teams
                'QAT1': [
                    'Al Sadd', 'Al Duhail', 'Al Rayyan', 'Al Gharafa', 'Al Wakrah', 'Qatar SC',
                    'Umm Salal', 'Al Markhiya', 'Al Arabi', 'Al Shamal', 'Al Ahli', 'Muaither'
                ],
                
                # UAE Pro League Teams
                'UAE1': [
                    'Al Ain', 'Shabab Al Ahli', 'Al Wasl', 'Al Wahda', 'Al Jazira', 'Sharjah',
                    'Baniyas', 'Al Nasr', 'Ajman', 'Khor Fakkan', 'Emirates Club', 'Hatta'
                ]
            }
            
            return league_teams.get(league_id, [])
            
        except Exception as e:
            logger.error(f"Error getting teams for league {league_id}: {str(e)}")
            return []

    def process_league(self, league_name, league_id):
        """Process all teams in a league"""
        logger.info(f"\nProcessing league: {league_name}")
        
        try:
            # Get all teams in the league
            teams = self.get_league_teams(league_id)
            
            if not teams:
                logger.warning(f"No teams found for league: {league_name}")
                return
                
            logger.info(f"Found {len(teams)} teams in {league_name}")
            
            # Process each team
            for team_name in teams:
                # Search for team to get logo URL
                team_data = self.search_team(team_name)
                if not team_data or 'logoImage' not in team_data:
                    logger.warning(f"No logo found for {team_name}")
                    continue
                
                logo_url = team_data['logoImage']
                if logo_url:
                    # Download logo
                    logo_path = self.download_logo(logo_url, team_name, league_name)
                    
                    # Store team data
                    if logo_path:
                        self.teams_data[team_name] = {
                            'league': league_name,
                            'logo_path': logo_path,
                            'logo_url': logo_url,
                            'team_id': team_data.get('id')
                        }
                
                # Rate limiting
                time.sleep(1)
            
            # Save progress after each league
            self.save_teams_data()
            
        except Exception as e:
            logger.error(f"Error processing league {league_name}: {str(e)}")

    def save_teams_data(self):
        """Save teams data to JSON file"""
        try:
            filepath = os.path.join(self.logos_dir, 'teams_data.json')
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.teams_data, f, indent=4, ensure_ascii=False)
            logger.info("Saved teams data to JSON")
        except Exception as e:
            logger.error(f"Error saving teams data: {str(e)}")

    def extract_all_logos(self):
        """Extract logos for all leagues"""
        total_teams = 0
        successful_downloads = 0
        
        # Only process Asian leagues
        asian_leagues = ['AFCL', 'SA1', 'QAT1', 'UAE1']
        
        for league_name, league_id in LEAGUE_IDS.items():
            # Skip non-Asian leagues
            if league_id not in asian_leagues:
                continue
                
            logger.info(f"\nStarting extraction for {league_name}")
            initial_count = len(self.teams_data)
            
            self.process_league(league_name, league_id)
            
            teams_processed = len(self.teams_data) - initial_count
            total_teams += teams_processed
            successful_downloads += teams_processed
            
            logger.info(f"Processed {teams_processed} teams in {league_name}")
            time.sleep(2)  # Rate limiting between leagues
        
        logger.info("\nLogo extraction completed!")
        logger.info(f"Total teams processed: {total_teams}")
        logger.info(f"Successful downloads: {successful_downloads}")
        logger.info(f"Logos saved in: {self.logos_dir}")

def main():
    extractor = LogoExtractor()
    extractor.extract_all_logos()

if __name__ == "__main__":
    main()
