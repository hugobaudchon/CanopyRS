"""Pure-classification metrics: one predicted class per image vs one true class per image.
No COCO, no IoU matching, no filename-alignment strategies — just two {image: class} mappings."""


def classification_metrics(truth: dict, preds: dict) -> dict:
    """Accuracy, per-class precision/recall/F1 (+ macro averages) and a confusion matrix, computed
    over the images present in both mappings. Images of ``truth`` missing from ``preds`` are counted
    (``num_missing_preds``) but not scored; extra predictions are ignored."""
    keys = sorted(set(truth) & set(preds))
    if not keys:
        raise ValueError("truth and preds share no image keys; check the file names")
    classes = sorted({truth[k] for k in keys} | {preds[k] for k in keys})
    confusion = {t: {p: 0 for p in classes} for t in classes}
    for key in keys:
        confusion[truth[key]][preds[key]] += 1

    per_class = {}
    for c in classes:
        tp = confusion[c][c]
        fp = sum(confusion[t][c] for t in classes if t != c)
        fn = sum(confusion[c][p] for p in classes if p != c)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    n = len(keys)
    return {
        "accuracy": sum(confusion[c][c] for c in classes) / n,
        "macro_precision": sum(m["precision"] for m in per_class.values()) / len(classes),
        "macro_recall": sum(m["recall"] for m in per_class.values()) / len(classes),
        "macro_f1": sum(m["f1"] for m in per_class.values()) / len(classes),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "num_images": n,
        "num_missing_preds": len(set(truth) - set(preds)),
    }
