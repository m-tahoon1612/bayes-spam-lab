"""Multinomial naive Bayes for short text messages.

log space everywhere, additive (Laplace) smoothing with alpha
"""
import math
from collections import Counter


def tokenize(text):
    return text.lower().split()


class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.vocab = set()

    def fit(self, docs, labels):
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
        # skip words never seen in training at all
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
