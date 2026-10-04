"""train + evaluate the spam classifier on data/messages.tsv

    python run.py
    python run.py --alpha 0.5 --seed 3
    python run.py --sweep
    python run.py --predict "you have won a free prize, call now"
"""
import argparse
import os
import statistics

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
    ap.add_argument("--top", type=int, default=0, help="show N most spammy words")
    ap.add_argument("--errors", action="store_true", help="print misclassified test messages")
    ap.add_argument("--sweep", action="store_true", help="alpha sweep over 10 seeds")
    ap.add_argument("--predict", metavar="TEXT")
    args = ap.parse_args(argv)

    docs, labels = load_tsv(args.data)

    if args.predict:
        model = NaiveBayes(args.alpha).fit(docs, labels)
        s = model.scores(args.predict)
        print(model.predict_one(args.predict), {k: round(v, 2) for k, v in s.items()})
        return

    if args.sweep:
        print(f"{'alpha':>6}  {'acc':>6}  {'prec':>6}  {'rec':>6}  {'f1':>6}   (mean of 10 seeds)")
        for a in (0.01, 0.1, 0.5, 1.0, 2.0, 5.0):
            rs = [evaluate(docs, labels, a, s, args.test_frac)[4] for s in range(10)]
            m = {k: statistics.mean(r[k] for r in rs) for k in ("accuracy", "precision", "recall", "f1")}
            print(f"{a:>6}  {m['accuracy']:.3f}   {m['precision']:.3f}   {m['recall']:.3f}   {m['f1']:.3f}")
        return

    model, test, y_true, y_pred, r = evaluate(docs, labels, args.alpha, args.seed, args.test_frac)
    print(f"{len(docs)} messages, {len(test)} in test, vocab {len(model.vocab)}, alpha={args.alpha}")
    print(f"accuracy  {r['accuracy']:.3f}")
    print(f"precision {r['precision']:.3f}  (spam)")
    print(f"recall    {r['recall']:.3f}  (spam)")
    print(f"f1        {r['f1']:.3f}")
    print()
    print(format_confusion(r))

    if args.errors:
        print("\nmisclassified:")
        for (d, t), p in zip(test, y_pred):
            if t != p:
                print(f"  [{t} -> {p}] {d}")

    if args.top:
        print("\nmost spammy tokens (log ratio):")
        for w, s in model.top_words("spam", "ham", args.top):
            print(f"  {w:<12} {s}")


if __name__ == "__main__":
    main()
