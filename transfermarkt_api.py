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
        self.max_requests_per_second = 5  # Maximum requests per second
        self.min_delay = 0.2  # Minimum delay between requests in seconds
        
        # Common abbreviations and their full names with standardized formats
        self.abbreviations = {
            # English Teams
            "manchester city": "manchester city",
            "man city": "manchester city",
            "arsenal": "arsenal fc",
            "liverpool": "liverpool fc",
            "manchester united": "manchester united",
            "man utd": "manchester united",
            "chelsea": "chelsea fc",
            "tottenham": "tottenham hotspur",
            "newcastle": "newcastle united",
            "brighton": "brighton & hove albion",
            
            # German Teams
            "bayern munich": "fc bayern munchen",
            "bayern": "fc bayern munchen",
            "dortmund": "borussia dortmund",
            "bvb": "borussia dortmund",
            "leipzig": "rb leipzig",
            "leverkusen": "bayer leverkusen",
            
            # Spanish Teams
            "real madrid": "real madrid cf",  # Important: use CF suffix
            "barcelona": "fc barcelona",
            "atletico madrid": "atletico de madrid",  # Use ASCII version
            "atletico": "atletico de madrid",
            "sevilla": "sevilla fc",
            "valencia": "valencia cf",
            
            # Italian Teams
            "juventus": "juventus",  # Just Juventus, no Turin
            "juve": "juventus",
            "inter milan": "inter",  # Just Inter
            "inter": "inter",
            "ac milan": "ac milan",
            "milan": "ac milan",
            "napoli": "ssc napoli",
            
            # French Teams
            "psg": "paris saint-germain",
            "paris": "paris saint-germain",
            "lyon": "olympique lyon",  # Not lyonnais
            "marseille": "olympique marseille",
            "monaco": "as monaco",
            "lille": "lille osc",
            "feyenoord": "feyenoord rotterdam",
            "az": "az alkmaar",
            
            # Scottish Teams
            "celtic": "celtic glasgow",
            "rangers": "rangers fc",
            "hearts": "heart of midlothian",
            "hibs": "hibernian fc",
            
            # Greek Teams
            "olympiakos": "olympiakos piraeus",
            "panathinaikos": "panathinaikos fc",
            "aek": "aek athens",
            "paok": "paok thessaloniki",
            
            # Turkish Teams
            "galatasaray": "galatasaray sk",
            "fenerbahce": "fenerbahçe sk",
            "besiktas": "beşiktaş jk",
            "basaksehir": "istanbul başakşehir",
            
            # Common prefixes/suffixes to standardize
            
            # Common prefixes/suffixes to standardize
            "fc": "",
            "cf": "",
            "ac": "",
            "as": "",
            "sv": "",
            "sc": "",
            "rc": "",
            "afc": "",
            "fk": "",
            "fsv": "",
            "vfb": "",
            "vfl": "",
            "tsg": "",
            "spvgg": "",
            
            # Common team name variations
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
            "levadiakos": "levadiakos fc",
            "kallithea": "gps kallithea",
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
            "independiente santa fe": "Independiente Santa Fe"
        }
        
        # Set fuzzy matching thresholds
        self.exact_match_threshold = 0.90  # Slightly reduced for better matching
        self.fuzzy_match_threshold = 0.65  # Slightly reduced for better matching
        
    def get_search_domain(self, team_name):
        """Get appropriate domain for team search based on team name"""
        team_lower = team_name.lower()
        
        # Common team keywords to domain mapping
        domain_keywords = {
            "gb": [
                "united", "city", "arsenal", "chelsea", "liverpool", "tottenham", "everton", "leeds", "newcastle",
                "man", "manchester", "forest", "villa", "wolves", "albion", "rovers", "wednesday", "county",
                "town", "rangers", "celtic", "hearts", "dundee", "aberdeen", "motherwell", "hibernian"
            ],
            "de": [
                "bayern", "dortmund", "leipzig", "leverkusen", "gladbach", "frankfurt", "münchen", "munchen",
                "hoffenheim", "bvb", "schalke", "werder", "hertha", "köln", "mainz", "wolfsburg", "hannover",
                "nurnberg", "stuttgart", "bochum", "freiburg", "augsburg", "hamburg"
            ],
            "es": [
                "barcelona", "madrid", "sevilla", "valencia", "bilbao", "sociedad", "real", "atletico",
                "villarreal", "betis", "espanyol", "getafe", "celta", "mallorca", "cadiz", "osasuna",
                "granada", "levante", "alaves", "valladolid"
            ],
            "it": [
                "milan", "inter", "juventus", "juve", "roma", "napoli", "lazio", "turin", "torino",
                "fiorentina", "atalanta", "sassuolo", "sampdoria", "genoa", "cagliari", "verona",
                "udinese", "bologna", "empoli", "venezia"
            ],
            "fr": [
                "paris", "lyon", "marseille", "monaco", "lille", "rennes", "psg", "nice", "montpellier",
                "strasbourg", "lens", "nantes", "bordeaux", "saint-etienne", "toulouse", "lorient"
            ],
            "pt": [
                "porto", "benfica", "sporting", "braga", "guimaraes", "boavista", "maritimo", "setubal",
                "belenenses", "vitoria"
            ],
            "nl": [
                "ajax", "psv", "feyenoord", "az", "utrecht", "vitesse", "twente", "groningen", "heerenveen",
                "willem"
            ],
            "tr": [
                "galatasaray", "fenerbahce", "besiktas", "trabzonspor", "basaksehir", "antalyaspor",
                "konyaspor", "alanyaspor", "gaziantep", "sivasspor"
            ],
            "gr": [
                "olympiakos", "panathinaikos", "aek", "paok", "aris", "ofi", "atromitos", "asteras",
                "larissa", "panionios"
            ]
        }
        
        # First try exact matches in team name variations
        for domain, keywords in domain_keywords.items():
            if any(keyword == team_lower for keyword in keywords):
                return domain
                
        # Then try partial matches
        for domain, keywords in domain_keywords.items():
            if any(keyword in team_lower for keyword in keywords):
                return domain
        
        # For teams with multiple possible domains, check specific patterns
        if "celtic" in team_lower or "rangers" in team_lower:
            return "gb"  # Scottish teams
        elif "ajax" in team_lower or "psv" in team_lower:
            return "nl"  # Dutch teams
        elif "porto" in team_lower or "benfica" in team_lower:
            return "pt"  # Portuguese teams
        elif "olympiakos" in team_lower or "paok" in team_lower:
            return "gr"  # Greek teams
        elif "galatasaray" in team_lower or "fenerbahce" in team_lower:
            return "tr"  # Turkish teams
                
        return "de"  # default to German domain

    def clean_team_name(self, team_name, domain="de"):
        """Clean team name by removing common prefixes/suffixes and standardizing format"""
        if not team_name:
            return team_name
            
        if isinstance(team_name, dict):
            return team_name
            
        # Convert to lowercase for consistent processing
        cleaned = team_name.lower().strip()
        
        # First check for direct matches in abbreviations
        if cleaned in self.abbreviations:
            return self.abbreviations[cleaned]
        
        # Special handling for German teams
        if domain == "de":
            # Handle Bayern Munich variations
            if any(name in cleaned for name in ["bayern", "munich", "münchen"]):
                return "fc bayern münchen"
            # Handle Dortmund variations
            if any(name in cleaned for name in ["dortmund", "bvb", "borussia"]):
                return "borussia dortmund"
                
        # Special handling for Spanish teams
        elif domain == "es":
            # Handle Real Madrid variations
            if "real madrid" in cleaned:
                return "real madrid cf"  # Add CF to get correct market value
            # Handle Barcelona variations
            if "barcelona" in cleaned:
                return "fc barcelona"
            # Handle Atletico Madrid variations
            if any(name in cleaned for name in ["atletico", "atlético", "atleti"]):
                return "atletico de madrid"  # Use standard ASCII characters
                
        # Special handling for Italian teams
        elif domain == "it":
            # Handle Juventus variations
            if any(name in cleaned for name in ["juventus", "juve"]):
                return "juventus"  # Remove 'turin' as it's not used in API
            # Handle Inter variations
            if "inter" in cleaned:
                return "inter"  # API uses just 'inter', not 'inter milan'
        
        # Special handling for numbered clubs
        if "1860 munich" in cleaned:
            return "tsv 1860 münchen"
        if "1899 hoffenheim" in cleaned:
            return "tsg hoffenheim"
        
        # Special handling for French teams
        elif domain == "fr":
            # Handle Lyon variations
            if "lyon" in cleaned:
                return "olympique lyon"  # Not lyonnais
            # Handle PSG variations
            if any(name in cleaned for name in ["psg", "paris"]):
                return "paris saint-germain"
            # Handle Monaco variations
            if "monaco" in cleaned:
                return "as monaco"
                
        # Standard replacements for special characters
        replacements = {
            'ue': 'u',  # Use standard ASCII instead of umlauts
            'ae': 'a',
            'oe': 'o',
            'ss': 'ss'
        }
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        
        # Remove common prefixes if they're standalone words
        prefixes_to_remove = ["fc", "ac", "as", "sv", "vfb", "vfl"]
        for prefix in prefixes_to_remove:
            cleaned = re.sub(f"^{prefix}\s+|\s+{prefix}$", "", cleaned)
        
        return cleaned

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

    def validate_market_value(self, value, team_name):
        """Validate market value is within reasonable range"""
        if value is None or value == 0:
            return None
        
        # Reasonable ranges in euros
        MIN_VALUE = 1_000_000  # 1M
        MAX_VALUE = 2_000_000_000  # 2B
        
        if not MIN_VALUE <= value <= MAX_VALUE:
            logger.warning(f"Suspicious market value {value} for {team_name}")
            return None
        
        return value

    def _generate_search_variations(self, team_name, domain="de"):
        """Generate different variations of the team name for searching"""
        variations = [
            team_name,  # Original name
            self.clean_team_name(team_name, domain),  # Cleaned name
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
                words[0],  # First word
                words[-1],  # Last word
                " ".join(words[:-1]),  # All but last word
                " ".join(words[1:]),  # All but first word
            ])
        
        # Remove duplicates and empty strings
        variations = list(set(filter(None, variations)))
        
        logger.debug(f"Generated variations for {team_name}: {variations}")
        return variations

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
        """Implement minimal rate limiting for API requests"""
        current_time = time.time()
        
        # Keep only recent requests in tracking
        self.request_times = [t for t in self.request_times if current_time - t < 1.0]
        
        # If we've made too many requests recently, just wait minimum delay
        if len(self.request_times) >= self.max_requests_per_second:
            time.sleep(self.min_delay)
            self.request_times = self.request_times[1:]  # Remove oldest request
        
        self.request_times.append(current_time)

    def _make_api_request(self, url, params):
        """Make an API request with minimal rate limiting and no retries"""
        try:
            self._rate_limit()  # Apply rate limiting
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 429:  # Too Many Requests
                logger.warning("Rate limit hit, returning None")
                return None
            
            response.raise_for_status()
            return response.json()  # Return JSON data instead of response object
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return None
        
    @lru_cache(maxsize=128)
    def get_team_squad(self, team_id, domain="de"):
        """Get all players in a team's squad with caching and validation"""
        if not team_id:
            return []
            
        logger.debug(f"Fetching squad for team ID: {team_id}")
        
        url = f"{self.base_url}/clubs/get-squad"
        params = {
            "id": str(team_id),
            "domain": domain
        }
        
        try:
            data = self._make_api_request(url, params)
            if not data:
                return []
            
            squad = data.get("squad", [])
            
            # Validate squad size
            if len(squad) < 15:
                logger.warning(f"Squad size suspiciously small ({len(squad)}) for team {team_id}")
                # Try to get cached value if available
                cached_key = f"squad_{team_id}_{domain}"
                if hasattr(self, cached_key):
                    cached_squad = getattr(self, cached_key)
                    if len(cached_squad) > len(squad):
                        return cached_squad
            
            # Cache valid squad
            if len(squad) >= 15:
                setattr(self, f"squad_{team_id}_{domain}", squad)
            
            logger.debug(f"Found {len(squad)} players in squad")
            time.sleep(0.1)  # Reduced delay for faster processing
            
            return squad
            
        except Exception as e:
            logger.error(f"Error fetching team squad: {str(e)}")
            return []

    def get_search_key(self, team_name):
        """Generate a standardized search key for a team name"""
        if not team_name:
            return ""
        # Remove special characters and convert to lowercase
        key = re.sub(r'[^a-zA-Z0-9\s]', '', team_name.lower())
        # Replace multiple spaces with single space and strip
        key = ' '.join(key.split())
        return key

    def search_team(self, team_name, domain="de"):
        """Search for a team and return its ID"""
        logger.info(f"Searching for team: {team_name}")
        
        try:
            # Clean and standardize the team name
            team_name = self.clean_team_name(team_name, domain)
            if isinstance(team_name, dict):
                # If team_name is a dict (from abbreviations), return it directly
                return team_name
                
            search_key = self.get_search_key(team_name)
            
            # Check cache first
            cache_key = f"{search_key}:{domain}"
            if cache_key in self.search_cache:
                logger.info(f"Found {team_name} in cache")
                return self.search_cache[cache_key]
            
            # Check if we have a direct mapping for this team
            if team_name.lower() in self.abbreviations:
                mapped_name = self.abbreviations[team_name.lower()]
                # If mapped_name is a string, we need to search for it
                if isinstance(mapped_name, str):
                    team_name = mapped_name
                    search_key = self.get_search_key(team_name)
                else:
                    # If mapped_name is a dict with id and name, cache and return it
                    self.search_cache[cache_key] = mapped_name
                    return mapped_name
            
            # Generate search variations
            search_variations = self._generate_search_variations(team_name, domain)
            
            # Try each variation
            for variation in search_variations:
                url = f"{self.base_url}/search"
                params = {"query": variation, "domain": domain}
                
                try:
                    data = self._make_api_request(url, params)
                    if not data:
                        continue
                    
                    # Look for exact match first in clubs array
                    clubs = data.get("clubs", [])
                    if not clubs:
                        # Try teams array if clubs is empty
                        clubs = data.get("teams", [])
                    
                    if clubs:
                        # Try exact match first
                        for club in clubs:
                            if club.get("name", "").lower() == variation.lower():
                                result = {"id": str(club["id"]), "name": club["name"]}
                                self.search_cache[cache_key] = result
                                return result
                        
                        # If no exact match, use fuzzy matching
                        best_match = None
                        highest_similarity = 0
                        
                        for club in clubs:
                            similarity = self._calculate_similarity(club.get("name", "").lower(), variation.lower())
                            if similarity > highest_similarity and similarity > 0.8:  # 80% similarity threshold
                                highest_similarity = similarity
                                best_match = club
                        
                        if best_match:
                            result = {"id": str(best_match["id"]), "name": best_match["name"]}
                            self.search_cache[cache_key] = result
                            return result
                        
                        # If no good match found, try next variation
                        continue
                
                except Exception as e:
                    logger.error(f"Error searching for variation {variation}: {str(e)}")
                    continue
            
            logger.warning(f"No results found for team: {team_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error searching for team {team_name}: {str(e)}")
            return None

    def get_team_market_value(self, team_name, search_domain=None):
        """Get market value for a team with validation"""
        logger.info(f"Getting market value for team: {team_name}")
        
        try:
            # Use appropriate domain based on team name if not provided
            if search_domain is None:
                search_domain = self.get_search_domain(team_name)
            
            # Clean and standardize team name
            cleaned_name = self.clean_team_name(team_name)
            
            # Search for the team first
            search_result = self.search_team(cleaned_name, search_domain)
            if not search_result:
                # Try alternate domains if first search fails
                alternate_domains = ["de", "gb", "es", "it"] if search_domain != "de" else ["gb", "es", "it", "fr"]
                for alt_domain in alternate_domains:
                    search_result = self.search_team(cleaned_name, alt_domain)
                    if search_result:
                        search_domain = alt_domain
                        break
                        
                if not search_result:
                    logger.warning(f"No search results found for team: {team_name}")
                    return None
            
            # Get team ID from search result
            team_id = search_result.get('id')
            if not team_id:
                logger.warning(f"No team ID found in search result for {team_name}")
                return None
            
            # Get squad data to calculate total market value
            squad = self.get_team_squad(team_id, search_domain)
            if not squad:
                logger.warning(f"No squad data found for team ID {team_id}")
                return None
            
            # Validate squad size
            if len(squad) < 15:
                logger.warning(f"Squad size suspiciously small ({len(squad)}) for team {team_id}")
                # Try alternate domain for squad data
                for alt_domain in ["de", "gb", "es", "it"]:
                    if alt_domain != search_domain:
                        alt_squad = self.get_team_squad(team_id, alt_domain)
                        if alt_squad and len(alt_squad) >= 15:
                            squad = alt_squad
                            break
            
            # Calculate total market value from squad with validation
            valid_values = []
            for player in squad:
                market_value = player.get('marketValue', {}).get('value', 0)
                # Validate individual player values
                if isinstance(market_value, (int, float)) and market_value > 0:
                    valid_values.append(market_value)
            
            # Validate total value
            if not valid_values:
                logger.warning(f"No valid market values found for {team_name}")
                return None
            
            total_value = sum(valid_values)
            
            # Validate total value is reasonable
            total_value = self.validate_market_value(total_value, team_name)
            if total_value is None:
                return None
            
            logger.info(f"Total market value for {team_name}: {total_value}")
            return {
                'market_value': total_value,
                'currency': 'EUR',
                'last_updated': None
            }
            
        except Exception as e:
            logger.error(f"Error getting market value for {team_name}: {str(e)}")
            return {
                'market_value': 0,
                'currency': 'EUR',
                'last_updated': None
            }


    def get_both_teams_market_value(self, home_team, away_team, domain="de"):
        """Get market values for both teams in a match"""
        logger.info(f"Getting market values for match: {home_team} vs {away_team}")
        
        try:
            home_value = self.get_team_market_value(home_team, domain)
            away_value = self.get_team_market_value(away_team, domain)
            
            logger.info(f"Market values - Home: {home_value}, Away: {away_value}")
            return home_value, away_value
            
        except Exception as e:
            logger.error(f"Error getting market values: {str(e)}")
            return None, None

    def get_market_values_batch(self, teams, domain="de"):
        """Get market values for multiple teams in a single batch"""
        logger.info(f"Getting market values for {len(teams)} teams in batch")
        
        try:
            # Use the existing get_multiple_teams_market_value method
            market_values = self.get_multiple_teams_market_value(teams, domain)
            
            # Format the results as expected by the odds generator
            return {
                team: {
                    'market_value': value,
                    'currency': 'EUR',
                    'last_updated': None  # We don't track this currently
                } for team, value in market_values.items()
            }
            
        except Exception as e:
            logger.error(f"Error getting market values batch: {str(e)}")
            return {team: {'market_value': 0, 'currency': 'EUR', 'last_updated': None} for team in teams}
