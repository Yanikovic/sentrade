#!/usr/bin/python3

from sample_reddit import RedditSampler
from firebase_admin import (credentials, firestore, initialize_app)
from firebase_admin.firestore import SERVER_TIMESTAMP

cred = credentials.Certificate("config/firebase_credentials.json")
initialize_app(cred)

firestore_db = firestore.client()


rs = RedditSampler()
sample = rs.sample()
num_sents = 0

for comment in sample: 
    comment["created"] = SERVER_TIMESTAMP
    comment_id = firestore_db.collection('comments').add(comment)
    for idx, sent in enumerate(comment['body']):
        num_sents += 1
        sent = {'body': sent,
                'commentBody': comment['body'],
                'sentencePosition': idx,
                'created': comment['created'],
                'commentId': comment_id,
                'sentiments': {'yanikovic': {'value': -2},
                               'starkus': {'value': -2},
                               'labeled': False,
                               'conflict': False},
                'permalink': comment['permalink'],
                'submissionTitle': comment['submissionTitle'],
                'subredditName': comment['subredditName'],
                'score': comment['score'],
                'flair': comment['flair']}

        firestore_db.collection('sentences').add(sent)

stats = firestore_db.collection('stats').document('stats')
unlabeled_count = stats.get().to_dict()['unlabeledSentences']
stats.update({'unlabeledSentences': unlabeled_count + num_sents})
