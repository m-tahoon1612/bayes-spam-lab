import math
from collections import Counter


def tokenize(text):
    return text.lower().split()


class NaiveBayes:
    def fit(self, docs, labels):
        doc_counts = Counter(labels)
        self.classes = sorted(doc_counts)
        n = len(labels)
        self.log_prior = {c: math.log(doc_counts[c] / n) for c in self.classes}
        self.word_counts = {c: Counter() for c in self.classes}
        for text, c in zip(docs, labels):
            self.word_counts[c].update(tokenize(text))
        self.total_words = {c: sum(self.word_counts[c].values()) for c in self.classes}
        return self

    def predict_one(self, text):
        best, best_score = None, None
        for c in self.classes:
            s = self.log_prior[c]
            for w in tokenize(text):
                s += math.log(self.word_counts[c][w] / self.total_words[c])
            if best_score is None or s > best_score:
                best, best_score = c, s
        return best

    def predict(self, docs):
        return [self.predict_one(d) for d in docs]
