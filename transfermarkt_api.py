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
        # Increase cache size and make it persistent across instances
        TransfermarktAPI.search_cache = getattr(TransfermarktAPI, 'search_cache', {})
        self.search_cache = TransfermarktAPI.search_cache
        self.max_workers = max_workers
        self.request_times = []  # Track API request times
        self.max_requests_per_second = 2  # Maximum requests per second (reduced from 5)
        self.min_delay = 0.5  # Minimum delay between requests in seconds
        
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
            "athletic club bilbao": "athletic bilbao",
            "athletic club": "athletic bilbao",
            "real sociedad": "real sociedad",
            "real sociedad san sebastian": "real sociedad",
            "sociedad": "real sociedad",
            "valencia": "fc valencia",
            "girona": "fc girona",
            
            # German Teams
            "bayern": "fc bayern münchen",
            "bayern munich": "fc bayern münchen",
            "bayern munchen": "fc bayern münchen",
            "dortmund": "borussia dortmund",
            "bvb": "borussia dortmund",
            "leverkusen": "bayer 04 leverkusen",
            "bayer leverkusen": "bayer 04 leverkusen",
            "leipzig": "rb leipzig",
            "greuther fürth": "spvgg greuther fürth",
            "greuther furth": "spvgg greuther fürth",
            "kaiserslautern": "1. fc kaiserslautern",
            "unterhaching": "spvgg unterhaching",
            
            # Italian Teams
            "milan": "ac milan",
            "inter": "inter mailand",
            "inter milan": "inter mailand",
            "internazionale": "inter mailand",
            "juventus": "juventus turin",
            "juve": "juventus turin",
            "napoli": "ssc napoli",
            "roma": "as rom",
            "lazio": "lazio rom",
            "ss lazio": "lazio rom",
            "atalanta": "atalanta bergamo",
            "cremonese": "us cremonese",
            "salernitana": "us salernitana 1919",
            "bari": "ssc bari",
            "bari 1908": "ssc bari",
            
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
            "ogc nice": "ogc nice",
            "ogc nizza": "ogc nice",
            "rennes": "stade rennes",
            "lens": "rc lens",
            "brest": "stade brestois 29",
            "stade brestois": "stade brestois 29",
            "bastia": "sc bastia",
            "sc bastia": "sporting club bastia",
            
            # Portuguese Teams
            "benfica": "sl benfica",
            "sporting": "sporting cp",
            "sporting cp": "sporting lissabon",
            "sporting clube": "sporting lissabon",
            "porto": "fc porto",
            "braga": "sc braga",
            "sporting braga": "sc braga",
            "guimaraes": "vitoria guimaraes",
            "farense": "sc farense",
            "estrela amadora": "cf estrela da amadora",
            
            # Dutch Teams
            "ajax": "ajax amsterdam",
            "psv": "psv eindhoven",
            "feyenoord": "feyenoord rotterdam",
            "az": "az alkmaar",
            "twente": "fc twente enschede",
            "fc twente": "fc twente enschede",
            
            # Belgian Teams
            "club brugge": "fc brügge",
            "club bruges": "fc brügge",
            "brugge": "fc brügge",
            "anderlecht": "rsc anderlecht",
            "gent": "kaa gent",
            "genk": "krc genk",
            "standard": "standard lüttich",
            "sint-truiden": "vv sint-truiden",
            "sint truiden": "vv sint-truiden",
            
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
            "galatasaray": "galatasaray sk",
            "galatasaray istanbul": "galatasaray sk",
            "fenerbahce": "fenerbahce sk",
            "fenerbahçe": "fenerbahce sk",
            "fenerbahce istanbul": "fenerbahce sk",
            "besiktas": "besiktas jk",
            "beşiktaş": "besiktas jk",
            "besiktas istanbul": "besiktas jk",
            "trabzonspor": "trabzonspor",
            "bodrumspor": "bb bodrumspor",
            "bb bodrumspor": "bandirmaboluspor",
            "rizespor": "caykur rizespor",
            
            # Greek Teams
            "paok": "paok thessaloniki",
            "paok saloniki": "paok thessaloniki",
            "olympiacos": "olympiakos piräus",
            "panathinaikos": "panathinaikos athen",
            "aek": "aek athen",
            "levadiakos": "levadiakos fc",
            "kallithea": "gps kallithea",
            "panaitolikos": "panetolikos gfs",
            
            # Additional Teams
            "rigas fs": "riga fs",
            "rīgas fs": "riga fs",
            "rigas futbola skola": "riga fs",
            "rīgas futbola skola": "riga fs",
            "qarabag": "qarabag agdam",
            "qarabağ": "qarabag agdam",
            "bodo/glimt": "bodo/glimt",
            "bodø/glimt": "bodo/glimt",
            "bodo glimt": "bodo/glimt",
            "fk bodo/glimt": "bodo/glimt",
            "fk bodø/glimt": "bodo/glimt",
            "fk bodo - glimt": "bodo/glimt",
            "ludogorets": "ludogorets",
            "ludogorets razgrad": "ludogorets",
            "ludogorets razgrad fc": "ludogorets",
            "pfc ludogorets": "ludogorets",
            "pfc ludogorets razgrad": "ludogorets",
            "elfsborg": "if elfsborg",
            "if elfsborg": "if elfsborg",
            "slavia praha": "sk slavia praha",
            "slavia prague": "sk slavia praha",
            "dynamo kyiv": "dynamo kiev",
            "dynamo kiev": "dynamo kiev",
            "dynamo kiew": "dynamo kiev",
            
            # Serbian Teams
            "red star": "roter stern belgrad",
            "red star belgrade": "roter stern belgrad",
            "crvena zvezda": "roter stern belgrad",
            "partizan": "partizan belgrad",
            
            # Ukrainian Teams
            "shakhtar": "schachtar donezk",
            "shakhtar donetsk": "schachtar donezk",
            
            # Czech Teams
            "slavia praha": "sk slavia praha",
            "slavia prague": "sk slavia praha",
            "sparta praha": "sparta prag",
            "sparta prague": "sparta prag",
            "viktoria plzen": "fc viktoria pilsen",
            "viktoria plzeň": "fc viktoria pilsen",
            "plzen": "fc viktoria pilsen",
            "slovacko": "1. fc slovacko",
            "slovácko": "1. fc slovacko",
            "jablonec": "fk jablonec",
            "mladá boleslav": "fk mlada boleslav",
            "mlada boleslav": "fk mlada boleslav",
            
            # Polish Teams
            "legia": "legia warschau",
            "legia warsaw": "legia warschau",
            "lech poznan": "lech posen",
            
            # Croatian Teams
            "dinamo zagreb": "gnk dinamo zagreb",
            "hajduk split": "hnk hajduk split",
            "rijeka": "hnk rijeka",
            "istra": "nk istra 1961",
            "istra 1961": "nk istra 1961",
            "gorica": "hnk gorica",
            "hnk gorica": "hnk gorica",
            "lokomotiva zagreb": "nk lokomotiva zagreb",
            "osijek": "nk osijek",
            
            # Saudi & UAE Teams
            "al orubah": "al-orobah fc",
            "al-orubah": "al-orobah fc",
            "al orobah": "al-orobah fc",
            "al ahli": "al-ahli saudi fc",
            "al-ahli": "al-ahli saudi fc",
            "al riyadh": "al-riyadh sc",
            "al-riyadh": "al-riyadh sc",
            "al wahda": "al-wahda fc",
            "al-wahda": "al-wahda fc",
            "al raed": "al-raed club",
            "al-raed": "al-raed club",
            "al nassr": "al-nassr fc",
            "al-nassr": "al-nassr fc",
            
            # Hungarian Teams
            "ferencvaros": "ferencvarosi tc",
            "ferencváros": "ferencvarosi tc",
            
            # Additional Missing Teams
            "dhamk": "damac fc",
            "al quadisiya": "al-qadisiyah",
            "cracovia kraków": "ks cracovia",
            "cracovia krakow": "ks cracovia",
            "slaven koprivnica": "nk slaven belupo",
            "šibenik": "hnk sibenik",
            "sibenik": "hnk sibenik",
            "varaždin": "nk varazdin",
            "varazdin": "nk varazdin",
            "monza": "ac monza",
            "torino": "fc torino",
            "sassuolo": "us sassuolo",
            "juve stabia": "ss juve stabia",
            "borussia m'gladbach": "borussia monchengladbach",
            "heidenheim": "1. fc heidenheim 1846",
            "rasenballsport leipzig": "rb leipzig",
            "union berlin": "1. fc union berlin",
            "sporting charleroi": "r charleroi sc",
            "beerschot-wilrijk": "k beerschot va",
            "české budějovice": "sk dynamo ceske budejovice",
            "ceske budejovice": "sk dynamo ceske budejovice",
            "dukla praha": "fk dukla praha",
            "wrexham": "wrexham afc",
            "santa clara": "cd santa clara",
            "kuban krasnodar": "kuban krasnodar",
            "kuban krasnodar fc": "kuban krasnodar",
            "kuban krasnodar krasnoyarsk": "kuban krasnodar",
            "kuban krasnodar krasnoyarsk fc": "kuban krasnodar",
            "1860 munchen": "tsv 1860 munchen",
            "gd estoril praia": "estoril praia",
            "estoril": "estoril praia",
            "avs": "avs futebol sad",
            "istanbul basaksehir": "istanbul basaksehir fk",
            "İstanbul başakşehir": "istanbul basaksehir fk",
            "ofi": "ofi kreta",
            "volos nfc": "volos nps",
            "volos": "volos nps",
            "lausanne sport": "fc lausanne-sport",
            "lausanne-sport": "fc lausanne-sport",
            "nec": "nec nijmegen",
            "guingamp": "ea guingamp",
            "schalke 04": "fc schalke 04",
            "schalke": "fc schalke 04",
            "monterrey": "cf monterrey",
            "pachuca": "cf pachuca",
            "atlas": "atlas fc",
            "necaxa": "club necaxa",
            "toluca": "deportivo toluca",
            "guadalajara": "cd guadalajara",
            
            # Additional Teams
            "hoffenheim": "tsg hoffenheim",
            "tsg hoffenheim": "tsg 1899 hoffenheim",
            "kaiserslautern": "1. fc kaiserslautern",
            "1. fc kaiserslautern": "1. fc kaiserslautern",
            "benfica": "sl benfica",
            "sl benfica": "benfica lissabon",
            "la equidad": "cd la equidad seguros",
            "levadiakos": "levadiakos fc",
            "levadiakos fc": "apo levadiakos",
            "kallithea": "gps kallithea",
            "gps kallithea": "kallithea fc",
            "sint-truiden": "vv sint-truiden",
            "sint truiden": "vv sint-truiden",
            "vv sint-truiden": "k. sint-truidense vv",
            
            # Add more mappings here
        }
        
        # Special direct mappings for problematic teams
        self.direct_mappings = {
            "bodo/glimt": {"id": "2619", "name": "FK Bodø/Glimt"},
            "ludogorets": {"id": "31614", "name": "Ludogorets Razgrad"},
            "riga fs": {"id": "35159", "name": "Riga FC"},
            "istra 1961": {"id": "4051", "name": "NK Istra 1961"},
            "gorica": {"id": "2947", "name": "HNK Gorica"},
            "bb bodrumspor": {"id": "24134", "name": "Bandırmaboluspor"},
            "elversberg": {"id": "4097", "name": "SV 07 Elversberg"},
            "darmstadt 98": {"id": "105", "name": "SV Darmstadt 98"},
            "nürnberg": {"id": "4", "name": "1. FC Nürnberg"},
            "bastia": {"id": "3444", "name": "SC Bastia"},
            "real sociedad": {"id": "681", "name": "Real Sociedad San Sebastián"},
            "hoffenheim": {"id": "533", "name": "TSG 1899 Hoffenheim"},
            "1. fc kaiserslautern": {"id": "2", "name": "1. FC Kaiserslautern"},
            "sl benfica": {"id": "294", "name": "Benfica Lissabon"},
            "la equidad": {"id": "6954", "name": "CD La Equidad Seguros"},
            "levadiakos fc": {"id": "2186", "name": "APO Levadiakos"},
            "gps kallithea": {"id": "5847", "name": "Kallithea FC"},
            "vv sint-truiden": {"id": "1773", "name": "K. Sint-Truidense VV"},
            "sporting cp": {"id": "336", "name": "Sporting CP"},
            "farense": {"id": "2420", "name": "SC Farense"},
            "estrela amadora": {"id": "15804", "name": "CF Estrela da Amadora"},
            "1. fc slovacko": {"id": "5226", "name": "1. FC Slovácko"},
            "sparta prag": {"id": "197", "name": "AC Sparta Praha"},
        }
        
        # Set fuzzy matching thresholds
        self.exact_match_threshold = 0.90  # Slightly reduced for better matching
        self.fuzzy_match_threshold = 0.65  # Slightly reduced for better matching
        
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
            "ý": "y", "ź": "z", "ż": "z", "ğ": "g", "ī": "i", "ū": "u",
            "ķ": "k", "ļ": "l", "ņ": "n", "ģ": "g"
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
        
    def _clean_special_chars(self, name):
        """Clean special characters from team names"""
        # Map of special characters to their replacements
        char_map = {
            'ā': 'a', 'ă': 'a', 'ą': 'a', 'ć': 'c', 'č': 'c',
            'ę': 'e', 'ě': 'e', 'ė': 'e', 'ğ': 'g', 'ı': 'i',
            'ī': 'i', 'ł': 'l', 'ń': 'n', 'ň': 'n', 'ō': 'o',
            'ő': 'o', 'ř': 'r', 'ś': 's', 'ş': 's', 'š': 's',
            'ť': 't', 'ū': 'u', 'ů': 'u', 'ű': 'u', 'ý': 'y',
            'ź': 'z', 'ż': 'z', 'ž': 'z', 'æ': 'ae', 'ø': 'o',
            'ß': 'ss', 'ğ': 'g', 'ș': 's', 'ț': 't', 'ğ': 'g',
            'ı': 'i', 'İ': 'i', 'ё': 'e', 'є': 'e', 'ї': 'i',
            'ґ': 'g', 'ң': 'n', 'ү': 'u', 'ұ': 'u', 'һ': 'h',
            'ә': 'a', 'ө': 'o', 'Ү': 'u', 'Ұ': 'u', 'Һ': 'h',
            'Ә': 'a', 'Ө': 'o'
        }
        
        # Convert to lowercase and replace special characters
        cleaned = name.lower()
        for special, normal in char_map.items():
            cleaned = cleaned.replace(special, normal)
        
        return cleaned

    def _is_youth_team(self, team_name):
        """Check if a team name indicates it's a youth team"""
        youth_indicators = ['u17', 'u18', 'u19', 'u20', 'u21', 'u23', 
                          'youth', 'jugend', 'junior', 'juvenil']
        name_lower = team_name.lower()
        return any(indicator in name_lower for indicator in youth_indicators)

    def search_team(self, team_name, domain="de"):
        """Search for a team by name with improved matching"""
        if not team_name:
            return None
            
        logger.info(f"Searching for team: {team_name}")
        
        # Try to find in abbreviations first (case-insensitive)
        team_lower = team_name.lower().strip()
        
        # Try direct match with abbreviations
        if team_lower in self.abbreviations:
            team_name = self.abbreviations[team_lower]
            logger.debug(f"Using abbreviated name: {team_name}")
        
        # Try fuzzy match with abbreviations if direct match fails
        if team_name.lower() not in self.abbreviations:
            matches = get_close_matches(team_lower, self.abbreviations.keys(), n=1, cutoff=0.8)
            if matches:
                team_name = self.abbreviations[matches[0]]
                logger.debug(f"Using fuzzy matched abbreviated name: {team_name}")
        
        # Clean the team name for searching
        search_key = self.get_search_key(team_name)
        logger.debug(f"Search key: {search_key}")
        
        # Check cache first
        cache_key = f"{search_key}:{domain}"
        if cache_key in self.search_cache:
            logger.info(f"Found in cache: {team_name}")
            return self.search_cache[cache_key]
        
        # Check direct mappings with both original and cleaned names
        clean_name = self.clean_team_name(team_name).lower()
        for key, value in self.direct_mappings.items():
            if clean_name == self.clean_team_name(key).lower():
                logger.info(f"Found direct mapping: {value['name']}")
                self.search_cache[cache_key] = value
                return value
                
        # Also check if the cleaned name matches any abbreviation's cleaned value
        for abbr_key, abbr_value in self.abbreviations.items():
            if clean_name == self.clean_team_name(abbr_value).lower():
                # Try to find in direct mappings using the full name
                if abbr_value.lower() in self.direct_mappings:
                    value = self.direct_mappings[abbr_value.lower()]
                    logger.info(f"Found direct mapping via abbreviation: {value['name']}")
                    self.search_cache[cache_key] = value
                    return value
        
        # Generate search variations
        search_variations = [
            self.clean_team_name(team_name),
            self._clean_special_chars(team_name)
        ]
        
        # Add common suffixes for certain teams
        base_name = self._clean_special_chars(team_name)
        if not any(x in base_name for x in ['fc', 'sc', 'fk', 'sk']):
            search_variations.extend([
                f"fc {base_name}",
                f"{base_name} fc",
                f"sk {base_name}",
                f"{base_name} sk"
            ])
        
        # Special handling for certain types of teams
        if any(x in team_lower for x in ["al-", "al "]):
            name = team_lower.replace("al-", "al ").replace("al ", "al-")
            search_variations.extend([name, f"{name} fc", f"{name} club"])
        
        # Try each variation
        tried_names = set()
        best_non_youth_match = None
        best_non_youth_similarity = 0
        
        for variation in search_variations:
            if variation in tried_names:
                continue
            tried_names.add(variation)
            
            try:
                url = f"{self.base_url}/search"
                response = self._make_api_request(url, {"query": variation, "domain": domain})
                response.raise_for_status()
                data = response.json()
                
                teams = data.get("clubs", [])
                if teams:
                    # Filter out youth teams unless specifically searching for one
                    if not self._is_youth_team(team_name):
                        teams = [t for t in teams if not self._is_youth_team(t["name"])]
                    
                    # Try exact match first
                    for team in teams:
                        team_clean = self._clean_special_chars(team["name"])
                        if self._is_exact_match(team_clean, variation):
                            logger.info(f"Found exact match: {team['name']}")
                            self.search_cache[cache_key] = team
                            return team
                    
                    # Try fuzzy matching with improved threshold
                    for team in teams:
                        if self._is_youth_team(team["name"]):
                            continue
                        
                        similarity = self._calculate_similarity(
                            self._clean_special_chars(team["name"]),
                            variation
                        )
                        
                        if similarity > best_non_youth_similarity:
                            best_non_youth_similarity = similarity
                            best_non_youth_match = team
                    
                    # Use the best non-youth match if it meets the threshold
                    if best_non_youth_match and best_non_youth_similarity >= self.fuzzy_match_threshold:
                        logger.info(f"Found fuzzy match: {best_non_youth_match['name']}")
                        self.search_cache[cache_key] = best_non_youth_match
                        return best_non_youth_match
                    
                    # For specific cases, use first non-youth result if reasonable match
                    if any(x in team_lower for x in ["al-", "al ", "fc", "sporting", "dynamo", "viktoria", "bodo", "riga", "ludogorets"]):
                        for team in teams:
                            if not self._is_youth_team(team["name"]):
                                similarity = self._calculate_similarity(
                                    self._clean_special_chars(team["name"]),
                                    variation
                                )
                                if similarity > 0.4:
                                    logger.info(f"Using first non-youth result: {team['name']}")
                                    self.search_cache[cache_key] = team
                                    return team
                
            except Exception as e:
                logger.error(f"Error searching for {variation}: {str(e)}")
                continue
        
        # If we found a decent non-youth match earlier, use it
        if best_non_youth_match and best_non_youth_similarity > 0.4:
            logger.info(f"Using best non-youth match: {best_non_youth_match['name']}")
            self.search_cache[cache_key] = best_non_youth_match
            return best_non_youth_match
        
        logger.warning(f"No match found for team: {team_name}")
        self.search_cache[cache_key] = None  # Cache negative results too
        return None

    def get_multiple_teams_market_value(self, teams, domain="de"):
        """Get market values for multiple teams in parallel with batching"""
        logger.info(f"Getting market values for {len(teams)} teams")
        
        results = {}
        search_tasks = []
        
        # Process teams in batches to avoid overwhelming the API
        batch_size = min(10, self.max_workers)  # Process up to 10 teams at once
        team_batches = [teams[i:i + batch_size] for i in range(0, len(teams), batch_size)]
        
        for batch in team_batches:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit batch of search tasks
                for team in batch:
                    if not team:
                        results[team] = 0
                        continue
                    
                    # Check cache first
                    search_key = self.get_search_key(team)
                    cache_key = f"{search_key}:{domain}"
                    
                    if cache_key in self.search_cache:
                        team_data = self.search_cache[cache_key]
                        if team_data:
                            # Submit squad task
                            future = executor.submit(self.get_team_squad, team_data["id"], domain)
                            search_tasks.append((team, future))
                        else:
                            results[team] = 0
                    else:
                        # Submit search task
                        future = executor.submit(self.search_team, team, domain)
                        search_tasks.append((team, future))
                
                # Process batch results
                for team, future in search_tasks:
                    try:
                        result = future.result()
                        if isinstance(result, dict) and "id" in result:  # Search result
                            squad_future = executor.submit(self.get_team_squad, result["id"], domain)
                            squad = squad_future.result()
                            total_value = sum(player.get("marketValue", {}).get("value", 0) for player in squad)
                            results[team] = total_value
                            logger.info(f"Total market value for {team}: €{total_value:,}")
                        elif isinstance(result, list):  # Squad result
                            total_value = sum(player.get("marketValue", {}).get("value", 0) for player in result)
                            results[team] = total_value
                            logger.info(f"Total market value for {team}: €{total_value:,}")
                        else:
                            results[team] = 0
                    except Exception as e:
                        logger.error(f"Error processing {team}: {str(e)}")
                        results[team] = 0
            
            search_tasks = []  # Clear tasks for next batch
        
        return results

    def get_both_teams_market_value(self, home_team, away_team, domain="de"):
        """Get market values for both teams in a match using parallel processing"""
        values = self.get_multiple_teams_market_value([home_team, away_team], domain)
        return {
            "home_market_value": values.get(home_team, 0),
            "away_market_value": values.get(away_team, 0)
        }

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
        clean1 = self._clean_special_chars(name1)
        clean2 = self._clean_special_chars(name2)
        similarity = self._calculate_similarity(clean1, clean2)
        return similarity >= self.exact_match_threshold
        
    def _find_best_fuzzy_match(self, teams, query):
        """Find the best fuzzy match from a list of teams"""
        query_clean = self._clean_special_chars(query)
        best_match = None
        best_similarity = 0
        
        for team in teams:
            team_name_clean = self._clean_special_chars(team["name"])
            similarity = self._calculate_similarity(team_name_clean, query_clean)
            
            if similarity > best_similarity and similarity >= self.fuzzy_match_threshold:
                best_similarity = similarity
                best_match = team
        
        return best_match

    def _calculate_similarity(self, str1, str2):
        """Calculate string similarity using difflib"""
        return sum(1 for a, b in zip(str1, str2) if a == b) / max(len(str1), len(str2))
        
    def _rate_limit(self):
        """Implement rate limiting for API requests"""
        current_time = time.time()
        
        # Remove old requests from tracking
        self.request_times = [t for t in self.request_times if current_time - t < 1.0]
        
        # If we've made too many requests recently, wait
        if len(self.request_times) >= self.max_requests_per_second:
            sleep_time = max(
                1.0 - (current_time - self.request_times[0]),  # Wait until a second has passed
                self.min_delay  # Minimum delay between requests
            )
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.request_times = self.request_times[1:]  # Remove oldest request
        
        # Ensure minimum delay between requests
        if self.request_times and (current_time - self.request_times[-1]) < self.min_delay:
            time.sleep(self.min_delay)
        
        self.request_times.append(current_time)

    def _make_api_request(self, url, params):
        """Make an API request with rate limiting and retries"""
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                self._rate_limit()  # Apply rate limiting
                response = requests.get(url, headers=self.headers, params=params)
                
                if response.status_code == 429:  # Too Many Requests
                    if attempt < max_retries - 1:  # Don't sleep on last attempt
                        sleep_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Rate limit hit, waiting {sleep_time} seconds...")
                        time.sleep(sleep_time)
                        continue
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    sleep_time = retry_delay * (2 ** attempt)
                    logger.warning(f"Request failed, retrying in {sleep_time} seconds... Error: {str(e)}")
                    time.sleep(sleep_time)
                else:
                    raise
        
        raise Exception("Max retries exceeded")
        
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
            response = self._make_api_request(url, params)
            response.raise_for_status()
            data = response.json()
            
            squad = data.get("squad", [])
            logger.debug(f"Found {len(squad)} players in squad")
            
            time.sleep(0.1)  # Reduced delay for faster processing
            
            return squad
        except Exception as e:
            logger.error(f"Error fetching team squad: {str(e)}")
            return []

    def get_team_market_value(self, team_name):
        """Get market value for a team"""
        try:
            logger.info(f"Getting market value for team: {team_name}")
            if not team_name:
                logger.warning("Team name is empty")
                return None
                
            # Clean and normalize team name
            cleaned_name = self.clean_team_name(team_name)
            logger.info(f"Cleaned team name: {cleaned_name} (original: {team_name})")
            
            # Check direct mappings first
            if cleaned_name in self.direct_mappings:
                logger.info(f"Found direct mapping for {cleaned_name}")
                team_id = self.direct_mappings[cleaned_name]["id"]
                logger.info(f"Team ID from direct mapping: {team_id}")
                
                # Make API request to get market value
                url = f"{self.base_url}/clubs/{team_id}"
                logger.info(f"Making API request to: {url}")
                response = requests.get(url, headers=self.headers)
                logger.info(f"API response status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    logger.debug(f"API response data: {data}")
                    market_value = data.get("marketValue", {}).get("value")
                    if market_value:
                        value = f"€{market_value}m"
                        logger.info(f"Found market value from direct mapping: {value}")
                        return value
                    else:
                        logger.warning(f"No market value found in response for {cleaned_name}")
            else:
                logger.info(f"No direct mapping found for {cleaned_name}, trying search")
            
            # If not in direct mappings, try to search
            search_url = f"{self.base_url}/search"
            params = {"query": cleaned_name}
            logger.info(f"Making search API request to: {search_url} with params: {params}")
            response = requests.get(search_url, headers=self.headers, params=params)
            logger.info(f"Search API response status: {response.status_code}")
            
            if response.status_code == 200:
                results = response.json()
                logger.debug(f"Search API response: {results}")
                if results and "clubs" in results and results["clubs"]:
                    team_id = results["clubs"][0]["id"]
                    logger.info(f"Found team ID from search: {team_id}")
                    
                    # Make API request to get market value
                    url = f"{self.base_url}/clubs/{team_id}"
                    logger.info(f"Making API request to: {url}")
                    response = requests.get(url, headers=self.headers)
                    logger.info(f"API response status: {response.status_code}")
                    
                    if response.status_code == 200:
                        data = response.json()
                        logger.debug(f"API response data: {data}")
                        market_value = data.get("marketValue", {}).get("value")
                        if market_value:
                            value = f"€{market_value}m"
                            logger.info(f"Found market value from search: {value}")
                            return value
                        else:
                            logger.warning(f"No market value found in response for {cleaned_name}")
                else:
                    logger.warning(f"No clubs found in search results for {cleaned_name}")
            else:
                logger.error(f"Search API request failed with status {response.status_code}")
            
            logger.warning(f"Could not find market value for {team_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting market value for {team_name}: {str(e)}")
            return None

    def get_both_teams_market_value(self, home_team, away_team):
        """Get market values for both teams"""
        logger.info(f"Getting market values for: {home_team} vs {away_team}")
        home_value = self.get_team_market_value(home_team)
        logger.info(f"Home team ({home_team}) market value: {home_value}")
        away_value = self.get_team_market_value(away_team)
        logger.info(f"Away team ({away_team}) market value: {away_value}")
        return home_value or 'N/A', away_value or 'N/A'
