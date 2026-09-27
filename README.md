# bayes-spam-lab

Multinomial naive Bayes spam/ham classifier, written from scratch in plain
Python for the probabilistic / generative models part of my master's (DS & AI,
started Sep 2026). Mostly study notes so I actually get the maths.

No sklearn on purpose, the point was to write the counting and the log
probabilites myself.

## what's here

- `nb.py` - tokenizer + `NaiveBayes` (Laplace / additive smoothing, log space)
- `metrics.py` - confusion matrix, accuracy, precision, recall, f1
- `data.py` - loading the tsv + a stratified train/test split
- `run.py` - CLI
- `data/messages.tsv` - 111 short SMS-style messages (38 spam, 73 ham).
  I wrote these myself in the style of the usual SMS spam corpus, so it's
  tiny and not representative of anything. Fine for checking the maths.

## usage

```
python run.py                       # train on 75%, eval on 25% (seed 0)
python run.py --alpha 0.1 --seed 3
python run.py --predict "you have won a free prize, call now"
python -m unittest discover -s tests
```

Python 3.8+, no dependencies.

## the model (short version)

For a message with tokens w1..wn pick the class c maximising

    log P(c) + sum_i log P(wi | c)

with

    P(w | c) = (count(w, c) + alpha) / (total_words(c) + alpha * |V|)

Without smoothing, any word that never appeared in one class gives log(0) and
the whole message gets -inf for that class. That was literally the first bug I
hit (see history). Words not in the training vocabulary at all are just
skipped.

Tokenizer lowercases, strips punctuation, maps anything with 4+ digits to
`<num>` (phone numbers, short codes). Collapsing the numbers helped more than anything else I tried.

## results

`python run.py` (seed 0, alpha 1.0):

```
111 messages, 28 in test, vocab 435, alpha=1.0
accuracy  0.857
precision 0.714  (spam)
recall    1.000  (spam)
f1        0.833

            pred ham  pred spam
true ham           14         4
true spam           0        10
```

Other seeds: 0.964 (seed 1), 0.893 (seed 2), 0.893 (seed 3). With only 28
test messages it jumps around a lot, one message is ~3.5%.

Recall is fine, the mistakes are ham classified as spam. Need to look at which
ones.

Stratified split was needed, with a plain random split some seeds had only 5-6
spam messages in test.

## license

MIT
