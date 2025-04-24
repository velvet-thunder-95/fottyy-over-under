import requests
import time
from difflib import get_close_matches
import re
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from thefuzz import fuzz

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
        
        # Set fuzzy matching thresholds
        self.exact_match_threshold = 0.90  # Slightly reduced for better matching
        self.fuzzy_match_threshold = 0.75  # Adjusted for better matching
        
        # Load unified team mappings
        try:
            with open('unified_leagues_teams_20250420_203041_20250420_203043.json', 'r') as f:
                self.unified_data = json.load(f)
            logger.info("Loaded unified team mappings successfully")
        except Exception as e:
            logger.warning(f"Could not load unified team mappings: {e}")
            self.unified_data = {}
        
        # Keep our existing team mappings
        self.TEAM_MAPPINGS = {
            # La Liga
            'Villarreal': 'Villarreal CF',
            'Athletic Club Bilbao': 'Athletic Bilbao',
            'Deportivo Alavés': 'Deportivo Alaves',
            'CA Osasuna': 'Osasuna',
            
            # Premier League
            'Brighton & Hove Albion': 'Brighton',
            'Wolverhampton Wanderers': 'Wolves',
            
            # Bundesliga
            'Borussia M\'gladbach': 'Borussia Mönchengladbach',
            'Bayern München': 'Bayern Munich',
            'Köln': 'FC Köln',
            
            # Serie A
            'Inter Milan': 'Inter',
            
            # Ligue 1
            'Olympique Marseille': 'Marseille',
            'Olympique Lyonnais': 'Lyon',
            'PSG': 'Paris Saint-Germain',
            
            # Other Leagues
            'RSC Anderlecht': 'Anderlecht',
            'Union Saint-Gilloise': 'Royale Union Saint-Gilloise',
            'Ferencváros': 'Ferencvarosi TC',
            'Viktoria Plzeň': 'Viktoria Plzen',
            'Qarabağ': 'FK Qarabag',
            'Neftçi': 'Neftchi Baku',
            
            # Italian Teams with Fixed IDs - Part 1
            'Cagliari': 'Cagliari Calcio',
            'Fiorentina': 'ACF Fiorentina',
            'Torino': 'Torino FC',
            'FC Torino': 'Torino FC',
            'Udinese': 'Udinese Calcio',
            'Genoa': 'Genoa CFC',
            'Lazio': 'SS Lazio',
            'Parma': 'Parma Calcio 1913',
            'Juventus': 'Juventus FC',
            'Brescia': 'Brescia Calcio',
            'Reggiana': 'AC Reggiana 1919',
            'Cittadella': 'AS Cittadella',
            'Salernitana': 'US Salernitana 1919',
            'Inter': 'Inter Milan',
            'Inter Milan': 'Inter Milan',
            'Milan': 'AC Milan',
            'AC Milan': 'AC Milan',
            'Roma': 'AS Roma',
            'Napoli': 'SSC Napoli',
            'Atalanta': 'Atalanta BC',
            
            # Italian Teams with Fixed IDs - Part 2
            'Bologna': 'Bologna FC 1909',
            'Empoli': 'Empoli FC',
            'Lecce': 'US Lecce',
            'Monza': 'AC Monza',
            'Sassuolo': 'US Sassuolo',
            'Verona': 'Hellas Verona',
            'Hellas Verona': 'Hellas Verona',
            'Frosinone': 'Frosinone Calcio',
            'Sampdoria': 'UC Sampdoria',
            'Spezia': 'Spezia Calcio',
            'Venezia': 'Venezia FC',
            'Cremonese': 'US Cremonese',
            'Palermo': 'US Città di Palermo',
            'Bari': 'SSC Bari',
            'Como': 'Como 1907',
            'Pisa': 'AC Pisa 1909',
            'Ternana': 'Ternana Calcio',
            'Catanzaro': 'US Catanzaro 1929',
            'Modena': 'Modena FC',
            
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
            "olympiacos": "olympiakos piraeus",
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
            "bodo/glimt": "FK Bodø/Glimt",
            "ludogorets razgrad": "Ludogorets Razgrad",
            "riga fc": "Riga FC",
            "istra 1961": "NK Istra 1961",
            "gorica": "HNK Gorica",
            "bb bodrumspor": "Bandırmaboluspor",
            "elversberg": "SV 07 Elversberg",
            "darmstadt 98": "SV Darmstadt 98",
            "nürnberg": "1. FC Nürnberg",
            "bastia": "SC Bastia",
            "real sociedad": "Real Sociedad San Sebastián",
            "hoffenheim": "TSG 1899 Hoffenheim",
            "1. fc kaiserslautern": "1. FC Kaiserslautern",
            "sl benfica": "Benfica Lissabon",
            "la equidad": "CD La Equidad Seguros",
            "levadiakos fc": "APO Levadiakos",
            "gps kallithea": "Kallithea FC",
            "vv sint-truiden": "K. Sint-Truidense VV",
            "sporting cp": "Sporting CP",
            "farense": "SC Farense",
            "estrela amadora": "CF Estrela da Amadora",
            
            # Add new direct mappings
            "union magdalena": "Unión Magdalena",
            "pae lamia": "PAS Lamia 1964",
            "gaziantep fk": "Gaziantep FK",
            "racing club de ferrol": "Racing de Ferrol",
            "sunderland afc": "Sunderland AFC",
            "oxford united": "Oxford United",
            "grasshopper club zürich": "Grasshopper Club Zürich",
            "al-shorta sc": "Al-Shorta SC",
            
            # Saudi Teams with Fixed IDs
            "al quadisiya": "Al-Qadisiyah",
            "al-quadisiya": "Al-Qadisiyah",
            "al qadisiyah": "Al-Qadisiyah",
            "al raed": "Al-Raed",
            "al-raed": "Al-Raed",
            "al shabab": "Al-Shabab",
            "al-shabab": "Al-Shabab",
            
            # English Teams
            "brighton": "Brighton & Hove Albion",
            "brighton & hove": "Brighton & Hove Albion",
            "brighton & hove albion": "Brighton & Hove Albion",
            "ipswich": "Ipswich Town",
            "ipswich town": "Ipswich Town",
            
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
            "1. fc saarbruck": "1. fc saarbrücken",
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
            
            # Indonesian Teams
            'Madura United': 'Madura United FC',
            'Persib': 'Persib Bandung',
            'Persija': 'Persija Jakarta',
            'PSM': 'PSM Makassar',
            'Bali United': 'Bali United FC',
            'Arema': 'Arema FC',
            'PSIS': 'PSIS Semarang',
            'Persebaya': 'Persebaya Surabaya',
            'Bhayangkara': 'Bhayangkara FC',
            'Borneo': 'Borneo FC',
            
            # South American Teams - Copa Sudamericana
            'Racing': 'Racing Club',
            'Racing Club': 'Racing Club',
            'Defensa': 'Defensa y Justicia',
            'Estudiantes': 'Estudiantes de La Plata',
            'Independiente': 'CA Independiente',
            'San Lorenzo': 'CA San Lorenzo de Almagro',
            'Lanús': 'Club Atlético Lanús',
            'Vélez': 'Vélez Sarsfield',
            'Huracán': 'CA Huracán',
            'Banfield': 'CA Banfield',
            'LDU Quito': 'LDU de Quito',
            'Barcelona SC': 'Barcelona Sporting Club',
            'Emelec': 'CS Emelec',
            'Nacional': 'Club Nacional',
            'Peñarol': 'CA Peñarol',
            'Cerro Porteño': 'Club Cerro Porteño',
            'Olimpia': 'Club Olimpia',
            'Sporting Cristal': 'Club Sporting Cristal',
            'Universitario': 'Universitario de Deportes',
            'Atlético Nacional': 'Atlético Nacional',
            'Millonarios': 'Millonarios FC',
            'Santa Fe': 'Independiente Santa Fe'
        }
        
        # Regional domain mappings
        self.DOMAIN_MAPPINGS = {
            'id': ['Madura United', 'Persib', 'Persija', 'PSM', 'Bali United', 'Arema', 'PSIS', 'Persebaya', 'Bhayangkara', 'Borneo'],
            'ar': ['Racing', 'Defensa', 'Estudiantes', 'Independiente', 'San Lorenzo', 'Lanús', 'Vélez', 'Huracán', 'Banfield'],
            'ec': ['LDU Quito', 'Barcelona SC', 'Emelec'],
            'uy': ['Nacional', 'Peñarol'],
            'py': ['Cerro Porteño', 'Olimpia'],
            'pe': ['Sporting Cristal', 'Universitario'],
            'co': ['Atlético Nacional', 'Millonarios', 'Santa Fe']
        }
        
        # Danish Teams
        self.DANISH_TEAMS = {
            'brøndby': 'Brøndby IF',
            'københavn': 'FC København',
            'agf': 'AGF Aarhus',
            'randers': 'Randers FC',
            'lyngby': 'Lyngby BK',
            'viborg': 'Viborg FF',
            'silkeborg': 'Silkeborg IF',
            'vejle': 'Vejle BK',
            'midtjylland': 'FC Midtjylland',
            'nordsjælland': 'FC Nordsjælland',
            'odense': 'OB Odense',
            'aarhus': 'AGF Aarhus',
            'horsens': 'AC Horsens',
            'aalborg': 'AaB Aalborg',
            'sønderjyske': 'SønderjyskE'
        }
        
        # Swedish Teams
        self.SWEDISH_TEAMS = {
            'djurgården': 'Djurgårdens IF',
            'gais': 'GAIS Göteborg',
            'malmö': 'Malmö FF',
            'aik': 'AIK Solna',
            'hammarby': 'Hammarby IF',
            'göteborg': 'IFK Göteborg',
            'elfsborg': 'IF Elfsborg',
            'häcken': 'BK Häcken',
            'kalmar': 'Kalmar FF',
            'norrköping': 'IFK Norrköping',
            'sirius': 'IK Sirius',
            'degerfors': 'Degerfors IF',
            'mjällby': 'Mjällby AIF',
            'varbergs': 'Varbergs BoIS',
            'värnamo': 'IFK Värnamo'
        }
        
        # Norwegian Teams
        self.NORWEGIAN_TEAMS = {
            'rosenborg': 'Rosenborg BK',
            'vålerenga': 'Vålerenga Fotball',
            'molde': 'Molde FK',
            'bodø/glimt': 'FK Bodø/Glimt',
            'brann': 'SK Brann',
            'lillestrøm': 'Lillestrøm SK',
            'viking': 'Viking FK',
            'strømsgodset': 'Strømsgodset IF',
            'odd': 'Odds BK',
            'sarpsborg': 'Sarpsborg 08',
            'haugesund': 'FK Haugesund',
            'tromsø': 'Tromsø IL',
            'aalesund': 'Aalesunds FK',
            'stabæk': 'Stabæk Fotball',
            'ham-kam': 'Hamarkameratene'
        }
        
        # Italian Teams
        self.ITALIAN_TEAMS = {
            'palermo': 'US Città di Palermo',
            'carrarese': 'Carrarese Calcio',
            'frosinone': 'Frosinone Calcio',
            'spezia': 'Spezia Calcio',
            'cosenza': 'Cosenza Calcio',
            'mantova': 'Mantova 1911',
            'juve stabia': 'SS Juve Stabia',
            'sampdoria': 'UC Sampdoria',
            'lecco': 'Calcio Lecco 1912',
            'sudtirol': 'FC Südtirol',
            'perugia': 'AC Perugia Calcio',
            'pescara': 'Delfino Pescara 1936',
            'avellino': 'US Avellino 1912',
            'triestina': 'US Triestina Calcio 1918',
            'virtus entella': 'Virtus Entella',
            'vicenza': 'LR Vicenza',
            'pro vercelli': 'FC Pro Vercelli 1892',
            'arezzo': 'SS Arezzo',
            'novara': 'Novara Calcio',
            'alessandria': 'US Alessandria Calcio 1912',
            'cesena': 'Cesena FC',
            'genoa': 'Genoa CFC',
            'lazio': 'SS Lazio',
            'cagliari': 'Cagliari Calcio',
            'fiorentina': 'ACF Fiorentina',
            'parma': 'Parma Calcio 1913',
            'juventus': 'Juventus FC',
            'cittadella': 'AS Cittadella',
            'salernitana': 'US Salernitana 1919',
            'sassuolo': 'US Sassuolo',
            'cremonese': 'US Cremonese',
            'pisa': 'AC Pisa 1909',
            'catanzaro': 'US Catanzaro 1929',
            'modena': 'Modena FC',
            'milan': 'AC Milan'
        }
        
        # English Teams
        self.ENGLISH_TEAMS = {
            'barnsley': 'Barnsley FC',
            'peterborough united': 'Peterborough United',
            'blackpool': 'Blackpool FC',
            'wrexham': 'Wrexham AFC',
            'bristol rovers': 'Bristol Rovers',
            'stevenage': 'Stevenage FC',
            'burton albion': 'Burton Albion',
            'cambridge united': 'Cambridge United',
            'leyton orient': 'Leyton Orient FC',
            'crawley town': 'Crawley Town FC',
            'exeter city': 'Exeter City',
            'lincoln city': 'Lincoln City FC',
            'bolton wanderers': 'Bolton Wanderers',
            'mansfield town': 'Mansfield Town FC',
            'reading': 'Reading FC',
            'northampton town': 'Northampton Town',
            'shrewsbury town': 'Shrewsbury Town FC',
            'stockport county': 'Stockport County FC',
            'huddersfield town': 'Huddersfield Town',
            'wigan athletic': 'Wigan Athletic',
            'rotherham united': 'Rotherham United',
            'wycombe wanderers': 'Wycombe Wanderers',
            'charlton athletic': 'Charlton Athletic FC'
        }
        
        # Add all team mappings to the abbreviations dictionary
        self.abbreviations = {}
        for team_dict in [self.ITALIAN_TEAMS, self.ENGLISH_TEAMS, self.DANISH_TEAMS, self.SWEDISH_TEAMS, self.NORWEGIAN_TEAMS]:
            self.abbreviations.update(team_dict)

    def clean_team_name(self, team_name, domain="de"):
        """Clean team name by removing common prefixes/suffixes and standardizing format"""
        if not team_name:
            return ""
            
        if isinstance(team_name, dict):
            return team_name
            
        # Convert to lowercase for consistent processing
        name = team_name.lower().strip()
        
        # Check for abbreviations first
        if name in self.abbreviations:
            name = self.abbreviations[name]
            logger.debug(f"Found abbreviation match: {name}")
            
        # Remove special characters and extra spaces
        name = re.sub(r'[^\w\s-]', '', name)
        name = ' '.join(name.split())
        
        # Convert to title case for consistent formatting
        name = name.title()
        
        logger.debug(f"Cleaned team name: {name}")
        return name

    def search_team(self, team_name, domain="de"):
        """Search for a team and return its ID"""
        if not team_name:
            return None
            
        # First check all team mappings for exact matches
        normalized = self.normalize_team_name(team_name).lower()
        
        # Get the exact Transfermarkt name if available
        exact_name = None
        if normalized in self.ITALIAN_TEAMS:
            exact_name = self.ITALIAN_TEAMS[normalized]
        elif normalized in self.ENGLISH_TEAMS:
            exact_name = self.ENGLISH_TEAMS[normalized]
        elif normalized in self.DANISH_TEAMS:
            exact_name = self.DANISH_TEAMS[normalized]
        elif normalized in self.SWEDISH_TEAMS:
            exact_name = self.SWEDISH_TEAMS[normalized]
        elif normalized in self.NORWEGIAN_TEAMS:
            exact_name = self.NORWEGIAN_TEAMS[normalized]
            
        if exact_name:
            logging.info(f"Found exact Transfermarkt name for {team_name}: {exact_name}")
            # Use the exact name for searching
            team_name = exact_name
            
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
            
        # Generate search variations
        variations = self._generate_search_variations(team_name, domain)
        
        # Try each variation
        for variation in variations:
            try:
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
                        
                        # If no exact match, return first result
                        club = clubs[0]
                        result = {"id": str(club["id"]), "name": club["name"]}
                        self.search_cache[cache_key] = result
                        return result
                
                except Exception as e:
                    logger.error(f"Error searching for variation {variation}: {str(e)}")
                    continue
            
            except Exception as e:
                logger.warning(f"Error searching for team {variation}: {str(e)}")
                continue
                
        logger.warning(f"No search results found for team: {team_name}")
        return None

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

    def normalize_team_name(self, team_name):
        """Normalize team name by removing special characters and converting to lowercase"""
        if not team_name:
            return ""
        # Remove special characters and convert to lowercase
        normalized = re.sub(r'[^a-zA-Z0-9\s]', '', team_name.lower())
        # Replace multiple spaces with single space and strip
        normalized = ' '.join(normalized.split())
        return normalized

    def get_search_domain(self, team_name):
        """Determine the appropriate search domain based on team name"""
        normalized_name = self.normalize_team_name(team_name)
        team_lower = team_name.lower()
        
        # Check in unified data first
        if self.unified_data:
            for league_data in self.unified_data.values():
                if team_lower in map(str.lower, league_data.get('teams', [])):
                    return league_data.get('domain', 'de')
        
        # Check if team is in any of our domain-specific lists
        for domain, teams in self.DOMAIN_MAPPINGS.items():
            if any(self.normalize_team_name(team) in normalized_name or normalized_name in self.normalize_team_name(team) for team in teams):
                logger.info(f"Found domain {domain} for team {team_name}")
                return domain
        
        # Swedish teams
        swedish_keywords = ['if', 'aik', 'malmö', 'malmo', 'göteborg', 'goteborg', 'hammarby', 'djurgården', 'djurgarden']
        if any(keyword in team_lower for keyword in swedish_keywords):
            return 'se'
            
        # Norwegian teams
        norwegian_keywords = ['rosenborg', 'molde', 'bodø', 'bodo', 'brann', 'viking', 'vålerenga', 'valerenga']
        if any(keyword in team_lower for keyword in norwegian_keywords):
            return 'no'
            
        # Danish teams
        danish_keywords = ['københavn', 'kobenhavn', 'midtjylland', 'brøndby', 'brondby', 'aarhus', 'randers']
        if any(keyword in team_lower for keyword in danish_keywords):
            return 'dk'
        
        # Default domain mappings for major leagues
        if any(term in normalized_name for term in ['united', 'city', 'arsenal', 'chelsea', 'liverpool']):
            return 'gb1'  # Premier League
        elif any(term in normalized_name for term in ['real', 'barcelona', 'atletico', 'sevilla']):
            return 'es1'  # La Liga
        elif any(term in normalized_name for term in ['bayern', 'dortmund', 'leipzig']):
            return 'de1'  # Bundesliga
        elif any(term in normalized_name for term in ['milan', 'inter', 'juventus', 'roma']):
            return 'it1'  # Serie A
        elif any(term in normalized_name for term in ['paris', 'lyon', 'marseille']):
            return 'fr1'  # Ligue 1
        
        # Default to German domain if no specific match
        return 'de'

    def get_search_key(self, team_name):
        """Generate a standardized search key for a team name"""
        if not team_name:
            return ""
        # Remove special characters and convert to lowercase
        key = re.sub(r'[^a-zA-Z0-9\s]', '', team_name.lower())
        # Replace multiple spaces with single space and strip
        key = ' '.join(key.split())
        return key

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
                return response.json()  # Return JSON data instead of response object
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    sleep_time = retry_delay * (2 ** attempt)
                    logger.warning(f"Request failed, retrying in {sleep_time} seconds... Error: {str(e)}")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Request failed after {max_retries} attempts: {str(e)}")
                    raise

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
            data = self._make_api_request(url, params)
            if not data:
                return []
            
            squad = data.get("squad", [])
            logger.debug(f"Found {len(squad)} players in squad")
            
            time.sleep(0.1)  # Reduced delay for faster processing
            
            return squad
            
        except Exception as e:
            logger.error(f"Error fetching team squad: {str(e)}")
            return []

    def get_market_values_batch(self, teams, domain="de"):
        """Get market values for multiple teams in a single batch"""
        logger.info(f"Getting market values for {len(teams)} teams in batch")
        
        try:
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_team = {
                    executor.submit(self.get_team_market_value, team, domain): team
                    for team in teams
                }
                
                # Collect results as they complete
                results = {}
                for future in as_completed(future_to_team):
                    team = future_to_team[future]
                    try:
                        market_value = future.result()
                        results[team] = {
                            'market_value': market_value if market_value is not None else 0,
                            'currency': 'EUR',
                            'last_updated': None
                        }
                    except Exception as e:
                        logger.error(f"Error getting market value for {team}: {str(e)}")
                        results[team] = {
                            'market_value': 0,
                            'currency': 'EUR',
                            'last_updated': None
                        }
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting market values batch: {str(e)}")
            return {team: {'market_value': 0, 'currency': 'EUR', 'last_updated': None} for team in teams}

    def get_team_market_value(self, team_name, domain="de"):
        """Get market value for a team"""
        logger.info(f"Getting market value for team: {team_name}")
        
        try:
            # Search for the team first
            search_result = self.search_team(team_name, domain)
            if not search_result:
                logger.warning(f"No search results found for team: {team_name}")
                return None
                
            # Get team ID from search result
            team_id = search_result.get('id')
            if not team_id:
                logger.warning(f"No team ID found in search result for {team_name}")
                return None
                
            # Get squad data to calculate total market value
            squad = self.get_team_squad(team_id, domain)
            if not squad:
                logger.warning(f"No squad data found for team ID {team_id}")
                return None
                
            # Calculate total market value from squad
            total_value = sum(player.get('marketValue', {}).get('value', 0) for player in squad)
            
            logger.info(f"Total market value for {team_name}: {total_value}")
            return total_value
            
        except Exception as e:
            logger.error(f"Error getting market value for {team_name}: {str(e)}")
            return None

    def get_both_teams_market_value(self, home_team, away_team, domain="de"):
        """Get market values for both teams in a match"""
        logger.info(f"Getting market values for match: {home_team} vs {away_team}")
        
        try:
            home_value = self.get_team_market_value(home_team, domain)
            away_value = self.get_team_market_value(away_team, domain)
            
            logger.info(f"Market values - Home: {home_value}, Away: {away_value}")
            return {
                "home_market_value": home_value if home_value is not None else 0,
                "away_market_value": away_value if away_value is not None else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting market values: {str(e)}")
            return {
                "home_market_value": 0,
                "away_market_value": 0
            }
