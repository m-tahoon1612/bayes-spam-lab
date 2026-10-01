"""Multinomial naive Bayes for short text messages.

Nothing clever here. Counts words per class, additive (Laplace)
smoothing with alpha, everything in log space so long messages
don't underflow.
"""
import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9£$']+")


def tokenize(text):
    out = []
    for t in _TOKEN.findall(text.lower()):
        t = t.strip("'")
        if not t:
            continue
        if t[0] in "£$":
            out.append("<money>")
            t = t[1:]
            if not t:
                continue
        # long digit runs are phone numbers / short codes, the exact
        # value doesn't matter much
        if sum(c.isdigit() for c in t) >= 4:
            out.append("<num>")
        else:
            out.append(t)
    return out


class NaiveBayes:
    def __init__(self, alpha=1.0):
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        self.alpha = alpha
        self.classes = []
        self.log_prior = {}
        self.word_counts = {}
        self.total_words = {}
        self.vocab = set()

    def fit(self, docs, labels):
        if len(docs) != len(labels):
            raise ValueError("docs and labels differ in length")
        doc_counts = Counter(labels)
        self.classes = sorted(doc_counts)
        n = len(labels)
        self.log_prior = {c: math.log(doc_counts[c] / n) for c in self.classes}
        self.word_counts = {c: Counter() for c in self.classes}
        for text, c in zip(docs, labels):
            toks = tokenize(text)
            self.word_counts[c].update(toks)
            self.vocab.update(toks)
        self.total_words = {c: sum(self.word_counts[c].values()) for c in self.classes}
        return self

    def log_prob_word(self, word, c):
        v = len(self.vocab)
        return math.log(
            (self.word_counts[c][word] + self.alpha)
            / (self.total_words[c] + self.alpha * v)
        )

    def scores(self, text):
        toks = [t for t in tokenize(text) if t in self.vocab]
        return {
            c: self.log_prior[c] + sum(self.log_prob_word(t, c) for t in toks)
            for c in self.classes
        }

    def predict_one(self, text):
        s = self.scores(text)
        return max(s, key=s.get)

    def predict(self, docs):
        return [self.predict_one(d) for d in docs]

    def top_words(self, c, other, n=10):
        """words with the biggest log-ratio P(w|c)/P(w|other)"""
        ratios = [
            (self.log_prob_word(w, c) - self.log_prob_word(w, other), w)
            for w in self.vocab
        ]
        ratios.sort(reverse=True)
        return [(w, round(r, 3)) for r, w in ratios[:n]]
