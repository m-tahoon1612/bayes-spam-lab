import csv
import random

from metrics import format_confusion, report
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

m = NaiveBayes(alpha=1.0).fit([docs[i] for i in train], [labels[i] for i in train])
y_true = [labels[i] for i in test]
y_pred = m.predict([docs[i] for i in test])
r = report(y_true, y_pred)
print(f"accuracy  {r['accuracy']:.3f}")
print(f"precision {r['precision']:.3f}")
print(f"recall    {r['recall']:.3f}")
print()
print(format_confusion(r))
