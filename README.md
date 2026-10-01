# bayes-spam-lab

Multinomial naive Bayes spam/ham classifier, written from scratch in plain
Python for the probabilistic / generative models part of my master's (DS & AI,
started Sep 2026). Mostly study notes so I actually get the maths.

No sklearn on purpose, the point was to write the counting and the log
probabilities myself.

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
python run.py --errors --top 10     # misclassified messages + most spammy tokens
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
`<num>` (phone numbers, short codes) and splits a leading £/$ into a `<money>`
token. Collapsing the numbers helped more than anything else I tried.

## results

Default run (`python run.py --errors --top 10`, seed 0, alpha 1.0):

```
111 messages, 28 in test, vocab 436, alpha=1.0
accuracy  0.857
precision 0.714  (spam)
recall    1.000  (spam)
f1        0.833

            pred ham  pred spam
true ham           14         4
true spam           0        10

misclassified:
  [ham -> spam] happy birthday!! hope you have a great day
  [ham -> spam] meeting moved to 3pm, room changed too
  [ham -> spam] great news about your sister, congrats to her
  [ham -> spam] will you be home for dinner

most spammy tokens (log ratio):
  <num>        2.969
  now          2.171
  reply        1.92
  claim        1.92
  your         1.871
  win          1.765
  ...
```

Seed 0 happens to be one of the worse splits, seeds 1-3 give 0.96 / 0.89 / 0.89.
With only 28 test messages one message is ~3.5% accuracy.

## notes to self

- All the errors are ham -> spam. Words like "you", "your", "have", "great"
  show up in both classes, and the ham messages are short, so a couple of
  slightly-spammy common words is enough to tip them. Real spam filters would
  care much more about false positives than this, so you'd want to shift the
  decision threshold (or weight the prior) instead of plain argmax.
- TODO try different alpha values properly, over several seeds.
- Stratified split was needed, with a random split some seeds had only 5-6
  spam messages in test.
- Didn't do stop word removal or bigrams. Might try bigrams some time.

## license

MIT
