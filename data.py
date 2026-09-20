import csv
import random


def load_tsv(path):
    docs, labels = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            labels.append(row["label"].strip())
            docs.append(row["text"].strip())
    return docs, labels


def stratified_split(docs, labels, test_frac=0.25, seed=0):
    """split per class so the tiny test set keeps roughly the same spam ratio"""
    rng = random.Random(seed)
    by_class = {}
    for d, l in zip(docs, labels):
        by_class.setdefault(l, []).append(d)
    train, test = [], []
    for l in sorted(by_class):
        items = by_class[l][:]
        rng.shuffle(items)
        k = max(1, round(len(items) * test_frac))
        test += [(d, l) for d in items[:k]]
        train += [(d, l) for d in items[k:]]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test
