# coding: utf-8
import os, re, sys, time, argparse, subprocess, time
import glob
import requests
from collections import defaultdict
from pprint import pprint

sys.path.append(os.getcwd())
from common import recDotDict
from scripts.data import util_sqlite3 

ROOT = 'https://jp1.api.riotgames.com'
URL = recDotDict({
    'summoner' : {
        'by_name': '/lol/summoner/v4/summoners/by-name/%s',
    },
    'match' : {
        'by_account': '/lol/match/v4/matchlists/by-account/%s',
        'by_matchId': '/lol/match/v4/matches/%s',
    },
})



def get_summoner_by_name(name: str) -> dict:
    '''
    <Return>
    - a dictionary as follows: 
    {'accountId': 'cofQ-rbC5Z8T0rhY6lYv1XNuyVAAUTkwdf3rA1PM_ZMbmis',
    'id': 'MeO3jVSaEnqWa9ZXKTIUQfj7Kfe9YoEy1e1WjJnPh4Q2tA',
    'name': 'letra418',
    'profileIconId': 4614,
    'puuid': 'KkUl5yJXRNJ4Y49hCZOzkokL59_g6Q_Y8dYgSCNWkSX0PhNT9d5NZX0AcRKsl5h_Ri5KldtEinYoOQ',
    'revisionDate': 1613323120000,
    'summonerLevel': 246}

    '''
    url = ROOT + URL.summoner.by_name % name
    res = requests.get(url, headers=headers).json()
    return res

def get_matches_by_accountId(accountId: str) -> list:
    '''
    <Return>
      - a list of dicts as follows:
             {'champion': 25,
              'gameId': 277171495,
              'lane': 'NONE',
              'platformId': 'JP1',
              'queue': 450,
              'role': 'DUO_SUPPORT',
              'season': 13,
              'timestamp': 1613055807593}
    '''
    url = ROOT + URL.match.by_account % accountId
    res = requests.get(url, headers=headers).json()
    return res['matches']

def get_detailed_match_by_matchId(matchId: str) -> dict:
    url = ROOT + URL.match.by_matchId % matchId
    res = requests.get(url, headers=headers).json()
    return res



def filter_match(match) -> bool:
    return True


def get_matches_by_summoner(summoner: dict, known_matchIds: set) -> dict:
    if 'accountId' not in summoner:
        print(summoner)
        exit(1)
    matches = get_matches_by_accountId(summoner['accountId'])
    match_details = []
    for match_overview in matches:
        if match_overview['gameId'] not in known_matchIds:
            match_detail = get_detailed_match_by_matchId(match_overview['gameId'])
            if filter_match(match_detail):
                match_details.append(match_detail)

    return match_details



def connect2db(db_path):
    import sqlite3
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

def main(args):
    conn, cur = connect2db(args.sqlite_path)
    exit(1)
    root_sns = [l.rstrip() for l in open(args.root_summoners)]
    summoners = [get_summoner_by_name(sn) for sn in root_sns]

    match_details = set()
    for summoner in summoners:
        match_details += get_matches_by_summoner(summoner, known_matchIds)
        time.sleep(args.sleep_time_per_crawl)
    print(match_details)
    exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser( 
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('models_root', type=str)
    parser.add_argument('--apikey', default='configs/apikey', type=str)
    parser.add_argument('--root-summoners', 
                        default='configs/root_summoners.txt', 
                        type=str)
    parser.add_argument('--sqlite-path', default='datasets/data.sqlite3', 
                        type=str)
    parser.add_argument('--sleep-time-per-crawl', default=0.1, 
                        type=float)
    args = parser.parse_args()

    global headers
    apikey = open(args.apikey).readline().rstrip()
    headers = {
        "X-Riot-Token": apikey,
    }

    main(args)
