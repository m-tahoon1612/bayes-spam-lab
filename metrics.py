def confusion(y_true, y_pred, positive="spam"):
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if p == positive:
            if t == positive:
                tp += 1
            else:
                fp += 1
        else:
            if t == positive:
                fn += 1
            else:
                tn += 1
    return tp, fp, fn, tn


def _div(a, b):
    return a / b if b else 0.0


def report(y_true, y_pred, positive="spam"):
    tp, fp, fn, tn = confusion(y_true, y_pred, positive)
    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    return {
        "accuracy": _div(tp + tn, tp + fp + fn + tn),
        "precision": precision,
        "recall": recall,
        "f1": _div(2 * precision * recall, precision + recall),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def format_confusion(r, positive="spam", negative="ham"):
    w = max(len(positive), len(negative), 4)
    lines = [
        " " * (w + 8) + f"pred {negative:<{w}} pred {positive:<{w}}",
        f"true {negative:<{w}}   {r['tn']:>{w + 5}} {r['fp']:>{w + 5}}",
        f"true {positive:<{w}}   {r['fn']:>{w + 5}} {r['tp']:>{w + 5}}",
    ]
    return "\n".join(lines)
