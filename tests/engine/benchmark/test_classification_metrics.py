"""Unit tests for the pure-classification evaluator (type-2 benchmark): {image: class} vs {image: class}."""

import pytest

from canopyrs.engine.benchmark.classifier.evaluator import classification_metrics


def test_perfect_predictions():
    truth = {"a.tif": "live", "b.tif": "dead"}
    metrics = classification_metrics(truth, dict(truth))
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["num_images"] == 2
    assert metrics["num_missing_preds"] == 0


def test_confusion_and_per_class():
    truth = {"a": 0, "b": 0, "c": 1, "d": 1}
    preds = {"a": 0, "b": 1, "c": 1, "d": 1}
    metrics = classification_metrics(truth, preds)
    assert metrics["accuracy"] == 0.75
    assert metrics["confusion_matrix"][0] == {0: 1, 1: 1}
    assert metrics["per_class"][0]["recall"] == 0.5           # one of two class-0 images found
    assert metrics["per_class"][1]["precision"] == pytest.approx(2 / 3)
    assert metrics["per_class"][0]["support"] == 2


def test_missing_and_extra_predictions():
    truth = {"a": 0, "b": 1}
    preds = {"a": 0, "z": 1}   # b missing, z extra (ignored)
    metrics = classification_metrics(truth, preds)
    assert metrics["num_images"] == 1
    assert metrics["num_missing_preds"] == 1


def test_disjoint_keys_raise():
    with pytest.raises(ValueError, match="share no image keys"):
        classification_metrics({"a": 0}, {"b": 0})


def test_benchmarker_loads_truth_from_csv_or_dict(tmp_path):
    from canopyrs.engine.benchmark.classifier.benchmark import ClassifierBenchmarker
    csv = tmp_path / "truth.csv"
    csv.write_text("file_name,class\na.tif,live\nb.tif,dead\n")
    assert ClassifierBenchmarker._load_truth(csv) == {"a.tif": "live", "b.tif": "dead"}
    assert ClassifierBenchmarker._load_truth({"x": 1}) == {"x": 1}


def test_benchmarker_rejects_bad_truth_csv(tmp_path):
    from canopyrs.engine.benchmark.classifier.benchmark import ClassifierBenchmarker
    csv = tmp_path / "truth.csv"
    csv.write_text("image,label\na.tif,live\n")
    with pytest.raises(ValueError, match="file_name"):
        ClassifierBenchmarker._load_truth(csv)
