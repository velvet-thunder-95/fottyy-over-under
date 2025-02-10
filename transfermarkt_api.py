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
            # Youth Teams
            "stuttgart u19": {"name": "VfB Stuttgart U19"},
            "liverpool u19": {"name": "Liverpool FC U19"},
            "atalanta u19": {"name": "Atalanta BC U19"},
            "benfica u19": {"name": "SL Benfica U19"},
            "az u19": {"name": "AZ Alkmaar U19"},
            
            # English Teams
            "celtic": "celtic glasgow",
            "celtic fc": "celtic glasgow",
            "rangers": "glasgow rangers",
            "rangers fc": "glasgow rangers",
            "arsenal": "fc arsenal",
            "chelsea": "fc chelsea",
            "liverpool": "fc liverpool",
            "manchester united": "manchester united",
            "man utd": "manchester united",
            "man united": "manchester united",
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
            "bayern münchen": "fc bayern münchen",
            "werder": "sv werder bremen",
            "werder bremen": "sv werder bremen",
            "dortmund": "borussia dortmund",
            "bvb": "borussia dortmund",
            "leverkusen": "bayer 04 leverkusen",
            "bayer leverkusen": "bayer 04 leverkusen",
            "leipzig": "rb leipzig",
            "greuther fürth": "spvgg greuther fürth",
            "greuther furth": "spvgg greuther fürth",
            "kaiserslautern": "1. fc kaiserslautern",
            "unterhaching": "spvgg unterhaching",
            "eintracht": "eintracht frankfurt",
            "eintracht frankfurt": {"id": "24", "name": "Eintracht Frankfurt"},
            "frankfurt": "eintracht frankfurt",
            
            # Italian Teams
            "milan": "ac milan",
            "inter": "inter mailand",
            "inter milan": "inter mailand",
            "internazionale": "inter mailand",
            "fc internazionale": "inter mailand",
            "juventus": "juventus turin",
            "juve": "juventus turin",
            "napoli": "ssc napoli",
            "roma": "as rom",
            "lazio": "lazio rom",
            "ss lazio": "lazio rom",
            "atalanta": "atalanta bergamo",
            "fiorentina": "acf fiorentina",
            "acf fiorentina": "acf fiorentina",
            "torino": "fc torino",
            "fc torino": "fc torino",
            "cremonese": "us cremonese",
            "salernitana": "us salernitana 1919",
            "brescia": "brescia calcio",
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
            "grenoble": "grenoble foot 38",
            "grenoble foot": "grenoble foot 38",
            "grenoble foot 38": "grenoble foot 38",
            "red star": "red star fc",
            "laval": "stade lavallois",
            "stade laval": "stade lavallois",
            "annecy": "fc annecy",
            
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
            "kaa gent": "kaa gent",
            "genk": "krc genk",
            "standard": "standard lüttich",
            "sint-truiden": "vv sint-truiden",
            "sint truiden": "vv sint-truiden",
            "mechelen": "kv mechelen",
            "kv mechelen": "kv mechelen",
            
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
            "sturm": "sk sturm graz",
            "austria wien": "fk austria wien",
            "austria vienna": "fk austria wien",
            
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
            "atromitos": {"id": "2182", "name": "Atromitos Athen"},
            "atromitos athens": {"id": "2182", "name": "Atromitos Athen"},
            "levadiakos": {"id": "2186", "name": "APO Levadiakos"},
            "levadiakos fc": {"id": "2186", "name": "APO Levadiakos"},
            "lamia": {"id": "7955", "name": "PAS Lamia 1964"},
            "pae lamia": {"id": "7955", "name": "PAS Lamia 1964"},
            "kallithea": {"id": "5847", "name": "Kallithea FC"},
            "gps kallithea": {"id": "5847", "name": "Kallithea FC"},
            "panaitolikos": "panetolikos gfs",
            
            # Additional Teams
            "rigas fs": "riga fc",
            "rīgas fs": "riga fc",
            "rigas futbola skola": "riga fc",
            "rīgas futbola skola": "riga fc",
            "qarabag": "qarabag agdam",
            "qarabağ": "qarabag agdam",
            "bodo/glimt": "bodo/glimt",
            "bodø/glimt": "bodo/glimt",
            "bodo glimt": "bodo/glimt",
            "fk bodo/glimt": "bodo/glimt",
            "fk bodø/glimt": "bodo/glimt",
            "fk bodo - glimt": "bodo/glimt",
            "ludogorets": "ludogorets razgrad",
            "ludogorets razgrad": "ludogorets razgrad",
            "ludogorets razgrad fc": "ludogorets razgrad",
            "pfc ludogorets": "ludogorets razgrad",
            "pfc ludogorets razgrad": "ludogorets razgrad",
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
            
            # Colombian Teams
            "union magdalena": "unión magdalena",
            "unión magdalena": "union magdalena",
            
            # Greek Teams
            "lamia": "pae lamia",
            "pae lamia": "pae lamia 1964",
            
            # Turkish Teams
            "gazisehir gaziantep": "gaziantep fk",
            "gazişehir gaziantep": "gaziantep fk",
            "gaziantep": "gaziantep fk",
            "bb bodrumspor": {"id": "24134", "name": "Bandırmaboluspor"},
            "bodrumspor": {"id": "24134", "name": "Bandırmaboluspor"},
            "bandirmaboluspor": {"id": "24134", "name": "Bandırmaboluspor"},
            
            # Spanish Teams
            "racing ferrol": "racing club de ferrol",
            "racing de ferrol": "racing club de ferrol",
            "racing club ferrol": "racing club de ferrol",
            
            # English Teams
            "sunderland": "sunderland afc",
            "oxford": "oxford united",
            "oxford utd": "oxford united",
            
            # Swiss Teams
            "grasshopper": "grasshopper club zürich",
            "grasshoppers": "grasshopper club zürich",
            "gc zürich": "grasshopper club zürich",
            
            # Iraqi Teams
            "al shorta": "al-shorta sc",
            "al-shorta": "al-shorta sc",
            "police club": "al-shorta sc",
            
            # Add more mappings here
            "bodo/glimt": {"id": "2619", "name": "FK Bodø/Glimt"},
            "ludogorets razgrad": {"id": "31614", "name": "Ludogorets Razgrad"},
            "riga fc": {"id": "35159", "name": "Riga FC"},
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
            
            # Add new direct mappings
            "union magdalena": {"id": "2442", "name": "Unión Magdalena"},
            "pae lamia": {"id": "7955", "name": "PAS Lamia 1964"},
            "gaziantep fk": {"id": "18562", "name": "Gaziantep FK"},
            "racing club de ferrol": {"id": "2859", "name": "Racing de Ferrol"},
            "sunderland afc": {"id": "289", "name": "Sunderland AFC"},
            "oxford united": {"id": "1072", "name": "Oxford United"},
            "grasshopper club zürich": {"id": "405", "name": "Grasshopper Club Zürich"},
            "al-shorta sc": {"id": "24162", "name": "Al-Shorta SC"},
            
            # Saudi Teams with Fixed IDs
            "al quadisiya": {"id": "18541", "name": "Al-Qadisiyah"},
            "al-quadisiya": {"id": "18541", "name": "Al-Qadisiyah"},
            "al qadisiyah": {"id": "18541", "name": "Al-Qadisiyah"},
            "al raed": {"id": "18549", "name": "Al-Raed"},
            "al-raed": {"id": "18549", "name": "Al-Raed"},
            "al shabab": {"id": "7801", "name": "Al-Shabab"},
            "al-shabab": {"id": "7801", "name": "Al-Shabab"},
            
            # English Teams
            "brighton": {"id": "1237", "name": "Brighton & Hove Albion"},
            "brighton & hove": {"id": "1237", "name": "Brighton & Hove Albion"},
            "brighton & hove albion": {"id": "1237", "name": "Brighton & Hove Albion"},
            "ipswich": {"id": "677", "name": "Ipswich Town"},
            "ipswich town": {"id": "677", "name": "Ipswich Town"},
            
            # German Teams with Updated IDs
            "jahn regensburg": "ssv jahn regensburg",
            "ssv jahn regensburg": "ssv jahn regensburg",
            "sandhausen": "sv sandhausen",
            "sv sandhausen": "sv sandhausen",
            "werder": "werder bremen",
            "werder bremen": "werder bremen",
            "sv werder bremen": "werder bremen",
            "stuttgart": "vfb stuttgart",
            "vfb stuttgart": "vfb stuttgart",
            "stuttgart ii": "vfb stuttgart ii",
            "vfb stuttgart ii": "vfb stuttgart ii",
            "union berlin": "1. fc union berlin",
            "1. fc union berlin": "1. fc union berlin",
            "magdeburg": "1. fc magdeburg",
            "1. fc magdeburg": "1. fc magdeburg",
            "ulm": "ssv ulm 1846",
            "ssv ulm": "ssv ulm 1846",
            "ssv ulm 1846": "ssv ulm 1846",
            "saarbrucken": "1. fc saarbrücken",
            "1. fc saarbrucken": "1. fc saarbrücken",
            "waldhof": "sv waldhof mannheim",
            "waldhof mannheim": "sv waldhof mannheim",
            "sv waldhof mannheim": "sv waldhof mannheim",
            "wehen": "sv wehen wiesbaden",
            "wehen wiesbaden": "sv wehen wiesbaden",
            "sv wehen wiesbaden": "sv wehen wiesbaden",
            
            # French Teams with Updated IDs
            "troyes": "ES Troyes AC",
            "es troyes": "ES Troyes AC",
            "es troyes ac": "ES Troyes AC",
            "laval": "Stade Lavallois",
            "stade laval": "Stade Lavallois",
            "stade lavallois": "Stade Lavallois",
            
            # Austrian Teams with Updated IDs
            "austria wien": "FK Austria Wien",
            "austria vienna": "FK Austria Wien",
            "fk austria wien": "FK Austria Wien",
            "rheindorf altach": "SCR Altach",
            "scr altach": "SCR Altach",

            # Italian Teams
            "torino": "FC Torino",
            "fc torino": "FC Torino",
            "fiorentina": "ACF Fiorentina",
            "acf fiorentina": "ACF Fiorentina",

            # Spanish Teams
            "mirandes": "CD Mirandés",
            "cd mirandes": "CD Mirandés",
            "eibar": "SD Eibar",
            "sd eibar": "SD Eibar",

            # Colombian Teams with Updated IDs
            "santa fe": "Independiente Santa Fe",
            "independiente santa fe": "Independiente Santa Fe",
            
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
            "grenoble": "grenoble foot 38",
            "grenoble foot": "grenoble foot 38",
            "grenoble foot 38": "grenoble foot 38",
            "red star": "red star fc",
            "laval": "stade lavallois",
            "stade laval": "stade lavallois",
            "annecy": "fc annecy",
            "dunkerque": {"id": "3725", "name": "USL Dunkerque"},
            "usl dunkerque": {"id": "3725", "name": "USL Dunkerque"},
            
            # Asian Teams
            "vissel kobe": {"id": "3958", "name": "Vissel Kobe"},
            "shanghai sipg": {"id": "7037", "name": "Shanghai Port FC"},
            "central coast mariners": {"id": "3419", "name": "Central Coast Mariners"},
            "johor darul ta'zim": {"id": "20012", "name": "Johor Darul Ta'zim FC"},
            "johor darul tazim": {"id": "20012", "name": "Johor Darul Ta'zim FC"},
            
            # Existing mappings continue...
        }
        
        # Set fuzzy matching thresholds
        self.exact_match_threshold = 0.90  # Slightly reduced for better matching
        self.fuzzy_match_threshold = 0.65  # Slightly reduced for better matching
        
    def clean_team_name(self, team_name, domain="de"):
        """Clean and standardize team name"""
        if not team_name:
            return None
            
        team_name = team_name.lower().strip()
        
        # Check if team name is in direct mappings
        if team_name in self.abbreviations:
            return self.abbreviations[team_name]
            
        return team_name

    def _clean_special_chars(self, text):
        """Remove special characters and standardize text"""
        if not text:
            return ""
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        # Replace multiple spaces with single space
        text = ' '.join(text.split())
        return text

    def _is_exact_match(self, name1, name2):
        """Check if two team names are exact matches"""
        if not name1 or not name2:
            return False
        clean1 = self._clean_special_chars(name1)
        clean2 = self._clean_special_chars(name2)
        return clean1 == clean2

    def search_team(self, team_name, domain="de"):
        """Search for a team and return its details"""
        logger.info(f"Searching for team: {team_name}")
        
        try:
            # Clean and standardize the team name
            team_name = self.clean_team_name(team_name, domain)
            if isinstance(team_name, dict):
                # If it's a direct mapping, return it
                return team_name
            
            # Check cache
            cache_key = f"{team_name.lower()}_{domain}"
            if cache_key in self.search_cache:
                return self.search_cache[cache_key]
            
            # Search API
            url = f"{self.base_url}/search"
            params = {"query": team_name, "domain": domain}
            
            data = self._make_api_request(url, params)
            if not data:
                logger.warning(f"No results found for team: {team_name}")
                return None
            
            # Look for exact match first in clubs array
            clubs = data.get("clubs", [])
            if not clubs:
                logger.warning(f"No clubs found for team: {team_name}")
                return None
            
            # Try exact match first
            for club in clubs:
                if self._is_exact_match(club["name"], team_name):
                    result = {"name": club["name"]}
                    self.search_cache[cache_key] = result
                    return result
            
            # If no exact match found, return None
            logger.warning(f"No exact match found for: {team_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error searching for team {team_name}: {str(e)}")
            return None

    def get_team_market_value(self, team_name, domain="de"):
        """Get market value for a team"""
        logger.info(f"Getting market value for team: {team_name}")
        
        team_info = self.search_team(team_name, domain)
        if not team_info:
            return None
            
        # If we found the team, return a default market value (since we're not using actual market values)
        return {"name": team_info["name"], "market_value": "10.00m €"}

    def get_both_teams_market_value(self, home_team, away_team, domain="de"):
        """Get market values for both teams"""
        logger.info(f"Getting market values for match: {home_team} vs {away_team}")
        
        home_value = self.get_team_market_value(home_team, domain)
        away_value = self.get_team_market_value(away_team, domain)
        
        logger.info(f"Market values - Home: {home_value}, Away: {away_value}")
        return home_value, away_value

    def _make_api_request(self, url, params=None, max_retries=3, initial_delay=1.0):
        """Make an API request with rate limiting and exponential backoff retry"""
        current_retry = 0
        delay = initial_delay

        while current_retry <= max_retries:
            try:
                # Rate limiting
                self._rate_limit()
                
                # Make the request
                response = requests.get(url, headers=self.headers, params=params)
                
                # Handle different status codes
                if response.status_code == 200:
                    return response.json()  # Return full response
                elif response.status_code == 503:
                    # Service Unavailable - retry with exponential backoff
                    if current_retry < max_retries:
                        logger.warning(f"Service Unavailable (503), retrying in {delay} seconds...")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                        current_retry += 1
                        continue
                elif response.status_code == 429:
                    # Rate limit exceeded - wait longer
                    wait_time = float(response.headers.get('Retry-After', delay))
                    logger.warning(f"Rate limit exceeded, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    current_retry += 1
                    continue
                else:
                    response.raise_for_status()
                    
            except requests.exceptions.RequestException as e:
                if current_retry < max_retries:
                    logger.warning(f"Request failed, retrying in {delay} seconds... Error: {str(e)}")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                    current_retry += 1
                    continue
                else:
                    logger.error(f"Max retries ({max_retries}) exceeded. Last error: {str(e)}")
                    return None
        
        logger.error(f"Failed to get successful response after {max_retries} retries")
        return None

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
