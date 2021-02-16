# coding: utf-8
import pymongo
import os, re, sys, time, argparse, subprocess, time, datetime, random
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
    'league' : {
        'by_summoner': '/lol/league/v4/entries/by-summoner/%s'
    },
})

num_requests = 0


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
    headers = {
        "X-Riot-Token": apikey,
    }
    res = requests.get(url, headers=headers).json()
    global num_requests
    num_requests += 1

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
    headers = {
        "X-Riot-Token": apikey,
    }
    res = requests.get(url, headers=headers, params={'queue':420}).json()
    global num_requests
    num_requests += 1
    return res['matches']

def get_detailed_match_by_matchId(matchId: str) -> dict:
    url = ROOT + URL.match.by_matchId % matchId
    headers = {
        "X-Riot-Token": apikey,
    }
    res = requests.get(url, headers=headers).json()
    global num_requests
    num_requests += 1
    return res

def get_league_by_summonerId(summonerId: str) -> dict:
    url = ROOT + URL.league.by_summoner % summonerId
    headers = {
        "X-Riot-Token": apikey,
    }
    res = requests.get(url, headers=headers).json()
    global num_requests
    num_requests += 1
    return res

def get_league_by_summoner(summoner: dict) -> dict:
    url = ROOT + URL.league.by_summoner % summoner['summonerId']
    headers = {
        "X-Riot-Token": apikey,
    }
    res = requests.get(url, headers=headers).json()
    global num_requests
    num_requests += 1
    for r in res:
        if r['queueType'] == 'RANKED_SOLO_5x5':
            return r


def get_matches_by_summoner(summoner: dict, known_matchIds: set) -> dict:
    if 'accountId' not in summoner:
        print(summoner, file=sys.stderr)
        exit(1)

    accountId = summoner['currentAccountId'] if 'currentAccountId' in summoner else summoner['accountId']
    matches = get_matches_by_accountId(accountId)
    match_details = []

    _num_req = 0
    for match_overview in reversed(matches):
        if match_overview['gameId'] not in known_matchIds:
            match_detail = get_detailed_match_by_matchId(match_overview['gameId'])
            # try:
            #     if filter_match(match_detail):
            #         match_details.append(match_detail)
            # except:
            #     print(match_detail, file=sys.stderr)
            #     exit(1)
            match_details.append(match_detail)
            _num_req += 1
            # time.sleep(1/20) # 20 requests per sec
            time.sleep(120/100) # 100 requests per 2mins
        if args.max_req_per_summoner and _num_req >= args.max_req_per_summoner:
            break
    return match_details

def filter_match(match) -> bool:
    if match['gameType'] != 'MATCHED_GAME':
        return False
    if match['gameMode'] != 'CLASSIC':
        return False
    return True


def update_summoners(db, summoners: list):
    coll = db.summoners
    for summoner in summoners:
        summoner['_id'] = summoner['summonerId']
        summoner['update_timestamp'] = int(time.time())

    # coll.update_many({}, summoners, upsert=True) # update_manyは同じキーを持つレコードを複数update?

    for summoner in summoners:
        key = {'summonerId': summoner['summonerId']}
        coll.update(key, summoner, upsert=True)

def update_matches(db, matches: list):
    coll = db.matches
    for match in matches:
        match['_id'] = match['gameId']
    coll.insert_many(matches)

def update_leagues(db, leagues: list):
    coll = db.leagues
    for league in leagues:
        league['_id'] = league['summonerId']

    for league in leagues:
        key = {'_id': league['_id']}
        coll.update(key, league)


def find_summoners_from_db(db):
    return list(db.summoners.find())

def find_matchIds_from_db(db):
    # return set(db.matches.find({}, {'gameId':1, '_id':0}))
    res = db.matches.find({}, {'gameId':1, '_id':0})
    return set([r['gameId'] for r in res])

def crawl_histories(db, target_summoners: list, 
                    known_summonerIds: set, known_matchIds: set):
    next_summoners = []
    new_matchIds = set

    for summoner in target_summoners:
        try:
            match_details = get_matches_by_summoner(summoner, known_matchIds)
        except Exception as e:
            print(e) 
            return []

        for match_detail in match_details:
            participants = [p['player'] for p in match_detail['participantIdentities']]
            try:
                match_detail['participantIdentities'] = [p['player']['summonerId'] for p in match_detail['participantIdentities']]
            except Exception as e:
                print(match_detail['participantIdentities'], file=sys.stderr)
                print(e)
                exit(1)
            

            # leagues = [get_league_by_summoner(p) for p in participants]
            # match_detail['participantLeagues'] = ["%s %s %d" % (l['tier'], l['rank'], l['leaguePoints']) for l in leagues]
            # update_leagues(db, leagues)

            for p in participants:
                if p['summonerId'] not in known_summonerIds:
                    next_summoners.append(p) 
                    known_summonerIds.add(p['summonerId'])

            update_summoners(db, participants)
        if len(match_details) > 0:
            update_matches(db, match_details)

        for m in match_details:
            known_matchIds.add(m['gameId'])
    return next_summoners


def main(args):
    with pymongo.MongoClient('localhost', 27017) as client:
        db = client.lol
        if db.summoners.count() == 0: # for cold starting
            root_sns = [l.rstrip() for l in open(args.root_summoners)]
            known_summoners = [get_summoner_by_name(sn) for sn in root_sns]
            known_summonerIds = set()

        else:
            known_summoners = find_summoners_from_db(db)
            random.shuffle(known_summoners)
            known_summonerIds = set([s['summonerId'] for s in known_summoners])


        if db.matches.count() == 0:
            known_matchIds = set()
        else:
            known_matchIds = find_matchIds_from_db(db)

        next_summoners = known_summoners

        i = 0
        global num_requests
        while True:
            print('Loop %d begins: #known summoners, #summoner queues, #matches = (%d, %d, %d)' % (i, len(known_summonerIds), len(next_summoners), len(known_matchIds)), flush=True)
            t = time.time()

            next_summoner = next_summoners.pop(0)
            try:
                next_summoners += crawl_histories(db, [next_summoner], 
                                                  known_summonerIds,
                                                  known_matchIds)
            except Exception as e:
                print(e)
                continue
            if len(next_summoners) > 100000:
                next_summoners = next_summoners[:100000]
            # except Exception as e:
            #     print(e)
            print('Loop %d ends: #requests=%d, elapsed time=%.3f' % \
                  (i, num_requests, time.time() - t), flush=True)
            num_requests = 0
            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser( 
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('models_root', type=str)
    parser.add_argument('--apikey', default='configs/apikey', type=str, help=' ')
    parser.add_argument('--root-summoners', 
                        default='configs/root_summoners.txt', 
                        type=str, help=' ')
    parser.add_argument('--sqlite-path', default='datasets/data.sqlite3', 
                        type=str, help=' ')
    parser.add_argument('--sleep-time-per-crawl', default=0.1, 
                        type=float, help=' ')
    parser.add_argument('--max-req-per-summoner', default=30, 
                        type=int, help=' ')
    # parser.add_argument('--crawl-known-summoners', default=False, 
    #                     action='store_true', type=bool,
    #                     help='to update match histories of known summoners.')

    args = parser.parse_args()

    global apikey
    apikey = open(args.apikey).readline().rstrip()
    headers = {
        "X-Riot-Token": apikey,
    }

    main(args)

'''
Rate Limits
20 requests every 1 seconds
100 requests every 2 minutes
'''
