import json
from typing import List, Union

from text_preprocessing.preprocessing import Preprocessor


class Vocabulary:

    def __init__(self, 
                 preprocessor: Preprocessor,
                 source: Union[str, List[str]] = "vocabularies/vocabulary.json") -> None:
        self.preprocessor = preprocessor 
        if isinstance(source, str):
            self._load(vocab_path=source)
        else:
            self._build(corpus=source)


    def _init_vocab(self) -> None:
        self.word2idx = {}
        self.word2idx['<UNK>'] = 0 
        self.max_len = 0


    def _build(self, corpus: List[str]) -> None:
        self._init_vocab()
        idx = 2
        preprocessed_corpus: List[List[str]] = \
            [self.preprocessor.apply(text) for text in corpus]
        for sent in preprocessed_corpus:
            for token in sent:
                if token not in self.word2idx:
                    self.word2idx[token] = idx
                    idx += 1
            self.max_len = max(self.max_len, len(sent))


    def _load(self, vocab_path: str) -> None:
        with open(vocab_path) as f:
            vocab = json.load(f)
            self.word2idx = vocab["word2idx"]
            self.max_len = vocab["max_len"]


    def save(self, vocab_path: str) -> None:
        with open(vocab_path, "w") as f:
            json.dump({"max_len": self.max_len, "word2idx": self.word2idx}, f)


    def get_index(self, token: str) -> int:
        idx = self.word2idx.get(token)
        return idx if idx else 0

