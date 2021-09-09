from typing import Generator, Set, List
from abc import ABC, abstractmethod

import asyncio

from core.connectors.reddit.reddit_config import asyncpraw_reddit


class RedditProducer(ABC):

    def __init__(self, 
                 subreddit_name: str,
                 flairs: List[str]):
        self.reddit = asyncpraw_reddit()
        self.subreddit_name = subreddit_name
        self.flairs = flairs

    async def init_subreddit(self) -> None:
        self.subreddit = await self.reddit.subreddit(self.subreddit_name)
    

    @abstractmethod
    async def publish(self, queue: asyncio.Queue) -> None:
        pass



class LiveRedditProducer(RedditProducer):

    def __init__(self, 
                 subreddit_name: str,
                 flairs: List[str]) -> None:
        super().__init__(subreddit_name, flairs)


    async def publish(self, queue: asyncio.Queue) -> None:
        await super().init_subreddit()
        async for comment in self.subreddit.stream.comments(skip_existing=True):
            await queue.put(comment)

