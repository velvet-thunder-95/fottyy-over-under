import requests
import time
from difflib import get_close_matches
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransfermarktAPI:
    def __init__(self, max_workers=20):
        self.base_url = "https://transfermarket.p.rapidapi.com"
        self.headers = {
            "x-rapidapi-host": "transfermarket.p.rapidapi.com",
            "x-rapidapi-key": "9a7723d114mshe44a60d17ffc5e8p1d348djsncb88cc895980"
        }
        self.search_cache = {}  # Cache search results
        self.max_workers = max_workers  # Number of threads for parallel processing
        
        # Common abbreviations and their full names
        self.abbreviations = {
            # English Teams
            "celtic": "celtic glasgow",
            "celtic fc": "celtic glasgow",
            "rangers": "glasgow rangers",
            "rangers fc": "glasgow rangers",
            "arsenal": "fc arsenal",
            "chelsea": "fc chelsea",
            "liverpool": "fc liverpool",
            "manchester united": "manchester united",
            "manchester city": "manchester city",
            "newcastle": "newcastle united",
            "tottenham": "tottenham hotspur",
            "west ham": "west ham united",
            "aston villa": "aston villa",
            "leeds": "leeds united",
            "everton": "fc everton",
            "brighton": "brighton & hove albion",
            
            # Spanish Teams
            "real madrid": "real madrid",
            "barcelona": "fc barcelona",
            "atletico madrid": "atlético de madrid",
            "atletico": "atlético de madrid",
            "atlético": "atlético de madrid",
            "atleti": "atlético de madrid",
            "sevilla": "sevilla fc",
            "villarreal": "villarreal cf",
            "real betis": "real betis sevilla",
            "athletic bilbao": "athletic bilbao",
            "real sociedad": "real sociedad",
            "valencia": "fc valencia",
            "girona": "fc girona",
            
            # German Teams
            "bayern": "fc bayern münchen",
            "bayern munich": "fc bayern münchen",
            "bayern munchen": "fc bayern münchen",
            "dortmund": "borussia dortmund",
            "bvb": "borussia dortmund",
            "leverkusen": "bayer 04 leverkusen",
            "leipzig": "rb leipzig",
            "rb leipzig": "rasenballsport leipzig",
            "gladbach": "borussia mönchengladbach",
            "frankfurt": "eintracht frankfurt",
            "wolfsburg": "vfl wolfsburg",
            "stuttgart": "vfb stuttgart",
            "freiburg": "sc freiburg",
            
            # Italian Teams
            "milan": "ac milan",
            "inter": "inter mailand",
            "inter milan": "inter mailand",
            "internazionale": "inter mailand",
            "juventus": "juventus turin",
            "juve": "juventus turin",
            "napoli": "ssc napoli",
            "roma": "as rom",
            "lazio": "ss lazio",
            "atalanta": "atalanta bergamo",
            "fiorentina": "ac florenz",
            "bologna": "fc bologna",
            
            # French Teams
            "psg": "paris saint-germain",
            "paris": "paris saint-germain",
            "marseille": "olympique marseille",
            "om": "olympique marseille",
            "lyon": "olympique lyon",
            "ol": "olympique lyon",
            "lille": "losc lille",
            "monaco": "as monaco",
            "nice": "ogc nice",
            "rennes": "stade rennes",
            "lens": "rc lens",
            "brest": "stade brestois 29",
            "stade brestois": "stade brestois 29",
            
            # Portuguese Teams
            "benfica": "sl benfica",
            "sporting": "sporting cp",
            "sporting cp": "sporting lissabon",
            "sporting clube": "sporting lissabon",
            "porto": "fc porto",
            "braga": "sc braga",
            "guimaraes": "vitoria guimaraes",
            
            # Dutch Teams
            "ajax": "ajax amsterdam",
            "psv": "psv eindhoven",
            "feyenoord": "feyenoord rotterdam",
            "az": "az alkmaar",
            "twente": "fc twente",
            
            # Belgian Teams
            "club brugge": "fc brügge",
            "club bruges": "fc brügge",
            "brugge": "fc brügge",
            "anderlecht": "rsc anderlecht",
            "gent": "kaa gent",
            "genk": "krc genk",
            "standard": "standard lüttich",
            
            # Scottish Teams
            "celtic": "celtic glasgow",
            "rangers": "glasgow rangers",
            "aberdeen": "fc aberdeen",
            "hearts": "heart of midlothian",
            
            # Austrian Teams
            "salzburg": "red bull salzburg",
            "rb salzburg": "red bull salzburg",
            "rapid wien": "rapid vienna",
            "sturm graz": "sk sturm graz",
            "austria wien": "austria vienna",
            
            # Swiss Teams
            "basel": "fc basel",
            "young boys": "bsc young boys",
            "zurich": "fc zürich",
            "servette": "servette fc",
            
            # Turkish Teams
            "galatasaray": "galatasaray istanbul",
            "fenerbahce": "fenerbahce istanbul",
            "besiktas": "besiktas istanbul",
            "trabzonspor": "trabzonspor",
            
            # Greek Teams
            "olympiacos": "olympiakos piräus",
            "panathinaikos": "panathinaikos athen",
            "aek": "aek athen",
            "paok": "paok saloniki",
            
            # Serbian Teams
            "red star": "roter stern belgrad",
            "red star belgrade": "roter stern belgrad",
            "crvena zvezda": "roter stern belgrad",
            "partizan": "partizan belgrad",
            
            # Ukrainian Teams
            "shakhtar": "schachtar donezk",
            "shakhtar donetsk": "schachtar donezk",
            "dynamo kyiv": "dynamo kiew",
            "dynamo kiev": "dynamo kiew",
            
            # Czech Teams
            "slavia praha": "slavia prag",
            "slavia prague": "slavia prag",
            "sparta praha": "sparta prag",
            "sparta prague": "sparta prag",
            "viktoria plzen": "viktoria pilsen",
            
            # Polish Teams
            "legia": "legia warschau",
            "legia warsaw": "legia warschau",
            "lech poznan": "lech posen",
            
            # Croatian Teams
            "dinamo zagreb": "dinamo zagreb",
            "hajduk split": "hajduk split",
            "rijeka": "hnk rijeka",
        }
        
        # Set fuzzy matching thresholds
        self.exact_match_threshold = 0.95  # For considering as exact match
        self.fuzzy_match_threshold = 0.70  # Lower threshold for better matching
        
    def clean_team_name(self, team_name):
        """Clean team name by removing common prefixes/suffixes and standardizing format"""
        if not team_name:
            return ""
            
        # Convert to lowercase for consistent processing
        name = team_name.lower().strip()
        
        # Log original name
        logger.debug(f"Cleaning team name: {name}")
        
        # Check for abbreviations first (check original and cleaned versions)
        if name in self.abbreviations:
            name = self.abbreviations[name]
            logger.debug(f"Found abbreviation match: {name}")
        
        # Remove common prefixes/suffixes
        prefixes = ["fc ", "ac ", "afc ", "ss ", "ssc ", "as ", "rc ", "rcd ", "real ", "atletico ", "athletic ", "cd ", "sc ", "sv ", "vfb ", "bv ", "tsv ", "fk "]
        suffixes = [" fc", " ac", " cf", " afc", " sc", " bsc", " fk", " sk", " bv", " sv", " united", " city", " town"]
        
        original_name = name
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                logger.debug(f"Removed prefix '{prefix}' from {original_name}")
                break
                
        original_name = name
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                logger.debug(f"Removed suffix '{suffix}' from {original_name}")
                break
                
        # Handle special characters and standardize names
        special_chars = {
            "ü": "u", "é": "e", "á": "a", "ñ": "n", "ö": "o", "ó": "o",
            "ć": "c", "č": "c", "ş": "s", "ș": "s", "ž": "z", "ı": "i",
            "ä": "a", "à": "a", "â": "a", "ã": "a", "ě": "e", "è": "e",
            "ê": "e", "ë": "e", "í": "i", "ì": "i", "î": "i", "ï": "i",
            "ń": "n", "ň": "n", "ò": "o", "ô": "o", "õ": "o", "ř": "r",
            "ś": "s", "š": "s", "ť": "t", "ù": "u", "û": "u", "ů": "u",
            "ý": "y", "ź": "z", "ż": "z"
        }
        
        for char, replacement in special_chars.items():
            if char in name:
                name = name.replace(char, replacement)
                logger.debug(f"Replaced special character '{char}' with '{replacement}'")
        
        # Standard name replacements
        replacements = {
            "munich": "munchen",
            "belgrade": "belgrad",
            "glasgow": "glasgow",
            "lisbon": "lissabon",
            "bruges": "brugge",
            "warsaw": "warschau",
            "prague": "prag",
            "kiev": "kiew",
            "donetsk": "donezk",
            "athens": "athen",
            "rome": "rom",
            "turin": "turin",
            "milan": "mailand",
            "florence": "florenz",
            "vienna": "wien",
            "moscow": "moskau",
        }
        
        for old, new in replacements.items():
            name = name.replace(old, new)
        
        # Remove parentheses and their contents
        name = re.sub(r'\([^)]*\)', '', name)
        
        # Remove multiple spaces and standardize
        name = re.sub(r'\s+', ' ', name)
        
        logger.debug(f"Final cleaned name: {name}")
        return name.strip()
        
    def get_search_key(self, team_name):
        """Generate a consistent search key for a team name"""
        return self.clean_team_name(team_name)
        
    def search_team(self, team_name, domain="de"):
        """Search for a team by name with improved matching"""
        if not team_name:
            return None
            
        logger.info(f"Searching for team: {team_name}")
        
        # Try to find in abbreviations first
        if team_name.lower() in self.abbreviations:
            team_name = self.abbreviations[team_name.lower()]
            logger.debug(f"Using abbreviated name: {team_name}")
        
        # Clean the team name for searching
        search_key = self.get_search_key(team_name)
        logger.debug(f"Search key: {search_key}")
        
        # Check cache first
        cache_key = f"{search_key}:{domain}"
        if cache_key in self.search_cache:
            logger.info(f"Found in cache: {team_name}")
            return self.search_cache[cache_key]
            
        url = f"{self.base_url}/search"
        
        # Generate search variations
        search_variations = self._generate_search_variations(team_name)
        
        # Try each variation
        for query in search_variations:
            try:
                response = requests.get(url, headers=self.headers, params={"query": query, "domain": domain})
                response.raise_for_status()
                data = response.json()
                
                teams = data.get("clubs", [])
                if teams:
                    # Try exact match first
                    for team in teams:
                        if self._is_exact_match(team["name"], query):
                            logger.info(f"Found exact match: {team['name']}")
                            self.search_cache[cache_key] = team
                            return team
                    
                    # Try fuzzy matching
                    best_match = self._find_best_fuzzy_match(teams, query)
                    if best_match:
                        logger.info(f"Found fuzzy match: {best_match['name']}")
                        self.search_cache[cache_key] = best_match
                        return best_match
                
                time.sleep(0.1)  # Reduced delay for faster processing
                
            except Exception as e:
                logger.error(f"Error searching for {query}: {str(e)}")
                continue
        
        logger.warning(f"No match found for team: {team_name}")
        return None
        
    def _generate_search_variations(self, team_name):
        """Generate different variations of the team name for searching"""
        variations = [
            team_name,  # Original name
            self.clean_team_name(team_name),  # Cleaned name
            team_name.replace(" ", "-"),  # With hyphens
            team_name.replace("-", " "),  # Without hyphens
        ]
        
        # Add common prefixes/suffixes
        prefixes = ["fc", "ac", "as", "ss", "ssc", "sc", "sv", "vfb", "bv", "tsv", "fk"]
        if not any(team_name.lower().startswith(p) for p in prefixes):
            variations.extend([f"{p} {team_name}" for p in prefixes])
        
        # Add variations without prefixes/suffixes
        variations.extend([
            team_name.replace("fc", "").strip(),
            team_name.replace("ac", "").strip(),
            team_name.replace("as", "").strip(),
        ])
        
        # Add words from team name
        words = team_name.split()
        if len(words) > 1:
            variations.extend([
                words[-1],  # Last word
                words[0],   # First word
                " ".join(words[1:]),  # Without first word
                " ".join(words[:-1])  # Without last word
            ])
        
        # Remove duplicates and None values
        variations = [v for v in variations if v]
        return list(dict.fromkeys(variations))
        
    def _is_exact_match(self, name1, name2):
        """Check if two team names are exact matches"""
        clean1 = self.clean_team_name(name1)
        clean2 = self.clean_team_name(name2)
        similarity = self._calculate_similarity(clean1, clean2)
        return similarity >= self.exact_match_threshold
        
    def _find_best_fuzzy_match(self, teams, query):
        """Find the best fuzzy match from a list of teams"""
        query_clean = self.clean_team_name(query)
        best_match = None
        best_similarity = 0
        
        for team in teams:
            team_name_clean = self.clean_team_name(team["name"])
            similarity = self._calculate_similarity(team_name_clean, query_clean)
            
            if similarity > best_similarity and similarity >= self.fuzzy_match_threshold:
                best_similarity = similarity
                best_match = team
        
        return best_match

    def _calculate_similarity(self, str1, str2):
        """Calculate string similarity using difflib"""
        return sum(1 for a, b in zip(str1, str2) if a == b) / max(len(str1), len(str2))
        
    @lru_cache(maxsize=128)
    def get_team_squad(self, team_id, domain="de"):
        """Get all players in a team's squad with caching"""
        if not team_id:
            return []
            
        logger.debug(f"Fetching squad for team ID: {team_id}")
        
        url = f"{self.base_url}/clubs/get-squad"
        params = {
            "id": str(team_id),
            "domain": domain
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            squad = data.get("squad", [])
            logger.debug(f"Found {len(squad)} players in squad")
            
            time.sleep(0.1)  # Reduced delay for faster processing
            
            return squad
        except Exception as e:
            logger.error(f"Error fetching team squad: {str(e)}")
            return []

    def get_multiple_teams_market_value(self, teams, domain="de"):
        """Get market values for multiple teams in parallel"""
        logger.info(f"Getting market values for {len(teams)} teams")
        
        # Create a list to store all tasks
        tasks = []
        
        # Create ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all search tasks first
            search_futures = {
                executor.submit(self.search_team, team, domain): team
                for team in teams
            }
            
            # Process search results and submit squad tasks
            for future in as_completed(search_futures):
                team = search_futures[future]
                try:
                    team_data = future.result()
                    if team_data:
                        # Submit squad task
                        squad_future = executor.submit(self.get_team_squad, team_data["id"], domain)
                        tasks.append((team, squad_future))
                except Exception as e:
                    logger.error(f"Error searching for {team}: {str(e)}")
                    tasks.append((team, None))
            
            # Process squad results
            results = {}
            for team, squad_future in tasks:
                if squad_future:
                    try:
                        squad = squad_future.result()
                        total_value = sum(player.get("marketValue", {}).get("value", 0) for player in squad)
                        results[team] = total_value
                        logger.info(f"Total market value for {team}: €{total_value:,}")
                    except Exception as e:
                        logger.error(f"Error getting squad for {team}: {str(e)}")
                        results[team] = 0
                else:
                    results[team] = 0
            
            return results

    def get_both_teams_market_value(self, home_team, away_team, domain="de"):
        """Get market values for both teams in a match using parallel processing"""
        values = self.get_multiple_teams_market_value([home_team, away_team], domain)
        return {
            "home_market_value": values.get(home_team, 0),
            "away_market_value": values.get(away_team, 0)
        }
