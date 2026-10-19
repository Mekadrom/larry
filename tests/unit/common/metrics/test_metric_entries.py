import torch

from larry.common.metrics.metrics import AggregateMetricEntry, ImageMetric


def test_an_empty_average_is_zero_instead_of_raising() -> None:
    assert AggregateMetricEntry(tag="x").value == 0


def test_merging_aggregates_adds_their_counts() -> None:
    left = AggregateMetricEntry(tag="x")
    left.add_(1.0)
    left.add_(3.0)
    right = AggregateMetricEntry(tag="x")
    right.add_(8.0)

    right.add_(left)

    assert right.count == 3
    assert right.value == 4.0


def test_image_metric_is_constructible_with_an_image() -> None:
    image = torch.zeros(3, 2, 2)

    assert ImageMetric(tag="img", image=image).value is image
