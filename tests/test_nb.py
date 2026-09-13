import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metrics import confusion, report  # noqa: E402
from nb import NaiveBayes  # noqa: E402


class NaiveBayesTest(unittest.TestCase):
    def setUp(self):
        docs = ["free prize call now", "win free cash", "see you at lunch", "lunch at noon?"]
        labels = ["spam", "spam", "ham", "ham"]
        self.m = NaiveBayes(alpha=1.0).fit(docs, labels)

    def test_predict(self):
        self.assertEqual(self.m.predict_one("free cash prize"), "spam")
        self.assertEqual(self.m.predict_one("lunch at noon"), "ham")

    def test_word_probs_sum_to_one(self):
        for c in self.m.classes:
            total = sum(math.exp(self.m.log_prob_word(w, c)) for w in self.m.vocab)
            self.assertAlmostEqual(total, 1.0, places=9)

    def test_unknown_words_fall_back_to_prior(self):
        s = self.m.scores("zzz qqq")
        self.assertAlmostEqual(s["spam"], math.log(0.5))


class MetricsTest(unittest.TestCase):
    def test_confusion(self):
        t = ["spam", "spam", "ham", "ham", "ham"]
        p = ["spam", "ham", "spam", "ham", "ham"]
        self.assertEqual(confusion(t, p), (1, 1, 1, 2))
        r = report(t, p)
        self.assertAlmostEqual(r["accuracy"], 0.6)
        self.assertAlmostEqual(r["precision"], 0.5)
        self.assertAlmostEqual(r["recall"], 0.5)

    def test_no_positive_predictions(self):
        r = report(["spam", "ham"], ["ham", "ham"])
        self.assertEqual(r["precision"], 0.0)


if __name__ == "__main__":
    unittest.main()
