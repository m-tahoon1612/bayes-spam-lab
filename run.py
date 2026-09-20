"""train + evaluate the spam classifier on data/messages.tsv

    python run.py
    python run.py --alpha 0.5 --seed 3
    python run.py --predict "you have won a free prize, call now"
"""
import argparse
import os

from data import load_tsv, stratified_split
from metrics import format_confusion, report
from nb import NaiveBayes

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "data", "messages.tsv")


def evaluate(docs, labels, alpha, seed, test_frac):
    train, test = stratified_split(docs, labels, test_frac, seed)
    model = NaiveBayes(alpha).fit([d for d, _ in train], [l for _, l in train])
    y_true = [l for _, l in test]
    y_pred = model.predict([d for d, _ in test])
    return model, test, y_true, y_pred, report(y_true, y_pred)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--predict", metavar="TEXT")
    args = ap.parse_args(argv)

    docs, labels = load_tsv(args.data)

    if args.predict:
        model = NaiveBayes(args.alpha).fit(docs, labels)
        s = model.scores(args.predict)
        print(model.predict_one(args.predict), {k: round(v, 2) for k, v in s.items()})
        return

    model, test, y_true, y_pred, r = evaluate(docs, labels, args.alpha, args.seed, args.test_frac)
    print(f"{len(docs)} messages, {len(test)} in test, vocab {len(model.vocab)}, alpha={args.alpha}")
    print(f"accuracy  {r['accuracy']:.3f}")
    print(f"precision {r['precision']:.3f}  (spam)")
    print(f"recall    {r['recall']:.3f}  (spam)")
    print(f"f1        {r['f1']:.3f}")
    print()
    print(format_confusion(r))


if __name__ == "__main__":
    main()
