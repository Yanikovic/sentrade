from random import (sample, randint)

import nltk

from reddit_config import praw_reddit


sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
PREFIX = "https://reddit.com/"
subreddits = 2 * ["wallstreetbets"] + ["investing", "Stocks", "StockMarket"]
flair_blacklist = ["Meme", "YOLO", "Help Needed", "Shitpost"]


class RedditSampler:

    def __init__(self, 
                 sub_names=subreddits, 
                 num_comments=200) -> None:

        self.reddit = praw_reddit()
        self.subreddits = [self.reddit.subreddit(sub_name) for sub_name in sub_names]
        self.num_comments = num_comments


    def json(self, submission, comment):
        sents = sent_tokenizer.tokenize(comment.body.strip())
        comment_json = {"body": sents,
                        "score": comment.score,
                        "permalink": PREFIX + comment.permalink, 
                        "submissionTitle": submission.title, 
                        "flair": submission.link_flair_text,
                        "subredditName": comment.subreddit.display_name,
                        }
        return comment_json
    

    def get_comments(self, submissions):
        comments = []
        for submission in submissions:
            if submission.link_flair_text not in flair_blacklist:
                post_comments = submission.comments 
                post_comments.replace_more(limit=0)
                for top_level_comment in post_comments:
                    comments.append(self.json(submission, top_level_comment))
                    for reply in top_level_comment.replies:
                        comments.append(self.json(submission, reply))
        return comments


    def sample(self):
        subreddit = self.subreddits[randint(0, len(self.subreddits) - 1)]
        print(f"Fetching data from {subreddit.display_name}...")
        
        submissions = subreddit.top("day", limit=20)
        comments = self.get_comments(submissions)
        num_comments = min(len(comments), self.num_comments)
        if (len(comments) >= num_comments):
            comments = sample(comments, self.num_comments)
        return comments
