import json

import asyncpraw



def get_reddit():
    with open('reddit_token.json') as f:
        token = json.load(f)
    return asyncpraw.Reddit('bot1', user_agent=token['User-Agent'])
