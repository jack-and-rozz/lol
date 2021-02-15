# coding: utf-8
import os, re, sys, time, argparse, subprocess
import glob
from collections import defaultdict
sys.path.append(os.getcwd())
from common import recDotDict

URL = recDotDict({
    'summoner' : {
        'summonername': '/lol/summoner/v4/summoners/by-name/%s'
    }
})


print(URL.summoner)
def main(args):
    key = open(args.apikey).readline().rstrip()
    print(key)

if __name__ == "__main__":
    parser = argparse.ArgumentParser( 
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('models_root', type=str)
    parser.add_argument('--apikey', default='configs/apikey', type=str)
    args = parser.parse_args()
    main(args)
