import json

import asyncpraw
import praw


def _token():
    with open('reddit_token.json') as f:
        return json.load(f)


def asyncpraw_reddit():
    token = _token()
    return asyncpraw.Reddit('bot1', user_agent=token['User-Agent'])


def praw_reddit():
    token = _token()
    return praw.Reddit('bot1', user_agent=token['User-Agent'])