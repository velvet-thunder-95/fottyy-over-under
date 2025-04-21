import requests
import time
from difflib import get_close_matches
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import json
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
        
        # Load unified team mappings
        try:
            with open('unified_leagues_teams_20250420_203041_20250420_203043.json', 'r') as f:
                self.unified_data = json.load(f)
            logger.info("Loaded unified team mappings successfully")
        except Exception as e:
            logger.warning(f"Could not load unified team mappings: {e}")
            self.unified_data = {}
        
        # Team name mappings from games_export.csv
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
            "independiente santa fe": "Independiente Santa Fe"
        }
        
        # English Teams Direct IDs
        self.ENGLISH_TEAMS = {
            'crewe alexandra': {'id': '127', 'name': 'Crewe Alexandra'},
            'milton keynes dons': {'id': '1020', 'name': 'Milton Keynes Dons'},
            'doncaster rovers': {'id': '142', 'name': 'Doncaster Rovers'},
            'colchester united': {'id': '148', 'name': 'Colchester United'},
            'gillingham': {'id': '149', 'name': 'Gillingham FC'},
            'afc wimbledon': {'id': '4777', 'name': 'AFC Wimbledon'},
            'harrogate town': {'id': '7968', 'name': 'Harrogate Town AFC'},
            'fleetwood town': {'id': '4782', 'name': 'Fleetwood Town'},
            'morecambe': {'id': '1076', 'name': 'Morecambe FC'},
            'salford city': {'id': '7370', 'name': 'Salford City'},
            'cesena': {'id': '4086', 'name': 'Cesena FC'}
        }
        
        # Italian Teams Direct IDs
        self.ITALIAN_TEAMS = {
            'palermo': {'id': '458', 'name': 'US Città di Palermo'},
            'carrarese': {'id': '6578', 'name': 'Carrarese Calcio'},
            'frosinone': {'id': '2821', 'name': 'Frosinone Calcio'},
            'spezia': {'id': '3522', 'name': 'Spezia Calcio'},
            'cosenza': {'id': '4088', 'name': 'Cosenza Calcio'},
            'mantova': {'id': '4087', 'name': 'Mantova 1911'},
            'juve stabia': {'id': '4083', 'name': 'SS Juve Stabia'},
            'sampdoria': {'id': '1038', 'name': 'UC Sampdoria'},
            'lecco': {'id': '6505', 'name': 'Calcio Lecco 1912'},
            'sudtirol': {'id': '15107', 'name': 'FC Südtirol'},
            'perugia': {'id': '4087', 'name': 'AC Perugia Calcio'},
            'pescara': {'id': '2834', 'name': 'Delfino Pescara 1936'},
            'avellino': {'id': '4078', 'name': 'US Avellino 1912'},
            'triestina': {'id': '4084', 'name': 'US Triestina Calcio 1918'},
            'virtus entella': {'id': '6865', 'name': 'Virtus Entella'},
            'vicenza': {'id': '4088', 'name': 'LR Vicenza'},
            'pro vercelli': {'id': '4089', 'name': 'FC Pro Vercelli 1892'},
            'arezzo': {'id': '4090', 'name': 'SS Arezzo'},
            'novara': {'id': '2885', 'name': 'Novara Calcio'},
            'alessandria': {'id': '4091', 'name': 'US Alessandria Calcio 1912'}
        }
        
        # Set fuzzy matching thresholds
        self.exact_match_threshold = 0.90  # Slightly reduced for better matching
        self.fuzzy_match_threshold = 0.65  # Slightly reduced for better matching
        
        # Women's team mappings with their IDs
        self.womens_teams = {
            "barcelona w": {"id": "131355", "name": "FC Barcelona Women"},
            "barcelona women": {"id": "131355", "name": "FC Barcelona Women"},
            "chelsea w": {"id": "131345", "name": "Chelsea FC Women"},
            "chelsea women": {"id": "131345", "name": "Chelsea FC Women"},
            "arsenal w": {"id": "131346", "name": "Arsenal WFC"},
            "arsenal women": {"id": "131346", "name": "Arsenal WFC"},
            "manchester city w": {"id": "131350", "name": "Manchester City WFC"},
            "manchester city women": {"id": "131350", "name": "Manchester City WFC"},
            "lyon w": {"id": "130428", "name": "Olympique Lyon Women"},
            "lyon women": {"id": "130428", "name": "Olympique Lyon Women"},
            "wolfsburg w": {"id": "131343", "name": "VfL Wolfsburg Women"},
            "wolfsburg women": {"id": "131343", "name": "VfL Wolfsburg Women"},
            "psg w": {"id": "130432", "name": "Paris Saint-Germain Women"},
            "psg women": {"id": "130432", "name": "Paris Saint-Germain Women"},
            "bayern munich w": {"id": "131351", "name": "Bayern Munich Women"},
            "bayern women": {"id": "131351", "name": "Bayern Munich Women"}
        }
        
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
            ],
            "cz": [
                "slavia praha", "sparta praha", "viktoria plzen", "banik ostrava", "sigma olomouc",
                "slovan liberec", "zbrojovka brno", "jihlava", "Příbram", "karviná", "mladá boleslav",
                "slovácko", "jablonec", "bohemians 1905", "dynamo č. budějovice", "teplice"
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

    def normalize_team_name(self, name):
        """Normalize team name for comparison."""
        if not name:
            return ""
            
        # First check if it's a women's team
        normalized = name.lower().strip()
        is_womens_team = any(suffix in normalized for suffix in [" w", " women", " ladies", " fem"])
        if is_womens_team:
            # Remove women's team suffixes for matching
            for suffix in [" w", " women", " ladies", " fem"]:
                normalized = normalized.replace(suffix, "")
            normalized = normalized.strip()
            # Add back "w" for consistent women's team lookup
            normalized = normalized + " w"
            if normalized in self.womens_teams:
                return self.womens_teams[normalized]["name"]
            
        # Check direct mappings
        if name in self.TEAM_MAPPINGS:
            return self.TEAM_MAPPINGS[name]
            
        # Remove special characters but keep spaces and letters with accents
        normalized = re.sub(r'[^\w\s\u00C0-\u017F]', '', name)
        normalized = normalized.lower().strip()
        
        # Remove common suffixes and prefixes
        normalized = re.sub(r'\s+(fc|cf|ac|sc|fk|if)\s*$', '', normalized)
        normalized = re.sub(r'^(fc|cf|ac|sc|fk|if)\s+', '', normalized)
        
        # Check unified mappings
        for league in self.unified_data.values():
            if isinstance(league, dict) and "teams" in league:
                for team in league["teams"]:
                    if team["name"].lower() == normalized and team["transfermarkt_name"]:
                        logger.info(f"Found exact Transfermarkt name for {name}: {team['transfermarkt_name']}")
                        return team["transfermarkt_name"]
        
        return normalized.strip()
        
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

    def get_both_teams_market_value(self, home_team, away_team):
        """Get market values for both teams."""
        try:
            home_value = self.get_team_market_value(home_team)
            away_value = self.get_team_market_value(away_team)
            
            # Extract market values from dictionaries
            if isinstance(home_value, dict) and 'market_value' in home_value:
                home_value = home_value.get('market_value')
            if isinstance(away_value, dict) and 'market_value' in away_value:
                away_value = away_value.get('market_value')
                
            return home_value, away_value
        except Exception as e:
            logger.error(f"Error getting market values: {str(e)}")
            return None, None

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
            self.normalize_team_name(team_name),  # Normalized name
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
        key = re.sub(r'[^\w\s\u00C0-\u017F]', '', team_name.lower())
        # Replace multiple spaces with single space and strip
        key = ' '.join(key.split())
        return key

    def search_team(self, team_name, domain="de"):
        """Search for a team and return its ID."""
        if not team_name:
            return None
            
        # First check Italian and English teams
        normalized = self.normalize_team_name(team_name).lower()
        if normalized in self.ITALIAN_TEAMS:
            return self.ITALIAN_TEAMS[normalized]
        if normalized in self.ENGLISH_TEAMS:
            return self.ENGLISH_TEAMS[normalized]
            
        # Generate variations of the team name
        variations = self._generate_search_variations(team_name, domain)
        
        # Try each variation
        for variation in variations:
            try:
                # Check if we have a direct mapping
                if variation in self.TEAM_MAPPINGS:
                    variation = self.TEAM_MAPPINGS[variation]
                
                # Make the search request
                url = f"https://transfermarkt-api.vercel.app/teams/search/{variation}"
                response = requests.get(url, headers=self.headers)
                
                if response.status_code == 200:
                    data = response.json()
                    if data and isinstance(data, list) and len(data) > 0:
                        # Find best match using fuzzy matching
                        best_match = None
                        highest_ratio = 0
                        
                        for team in data:
                            ratio = fuzz.ratio(variation.lower(), team['name'].lower())
                            if ratio > highest_ratio and ratio >= self.fuzzy_match_threshold:
                                highest_ratio = ratio
                                best_match = team
                                
                        if best_match:
                            return {"id": best_match["id"], "name": best_match["name"]}
                            
            except Exception as e:
                logger.warning(f"Error searching for team {variation}: {str(e)}")
                continue
                
        logger.warning(f"No search results found for team: {team_name}")
        return None

    def get_team_market_value(self, team_name, search_domain=None):
        """Get market value for a team with validation"""
        logger.info(f"Getting market value for team: {team_name}")
        
        try:
            # Use appropriate domain based on team name if not provided
            if search_domain is None:
                search_domain = self.get_search_domain(team_name)
            
            # Clean and standardize team name
            cleaned_name = self.normalize_team_name(team_name)
            
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
            
            # Make the market value request
            url = f"https://transfermarkt-api.vercel.app/teams/{team_id}/market-value"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, dict):
                    market_value = data.get('marketValue', 0)
                    if isinstance(market_value, (int, float)):
                        logger.info(f"Total market value for {team_name}: {market_value}")
                        return {
                            'market_value': market_value,
                            'currency': 'EUR',
                            'last_updated': None
                        }
            
            logger.warning(f"No valid market value found for team: {team_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting market value for {team_name}: {str(e)}")
            return None
