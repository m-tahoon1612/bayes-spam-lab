import csv
import random

from nb import NaiveBayes

docs, labels = [], []
with open("data/messages.tsv") as f:
    for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
        docs.append(row["text"])
        labels.append(row["label"])

idx = list(range(len(docs)))
random.seed(1)
random.shuffle(idx)
cut = int(len(idx) * 0.75)
train, test = idx[:cut], idx[cut:]

m = NaiveBayes().fit([docs[i] for i in train], [labels[i] for i in train])
pred = m.predict([docs[i] for i in test])
correct = sum(p == labels[i] for p, i in zip(pred, test))
print("accuracy", correct / len(test), f"({correct}/{len(test)})")
