import pytest

from partB.eta_mlflow import decide


PROD = {"mae": 100.0, "mae_rush": 100.0, "mae_rain": 100.0, "p95_ms": 8.0}
BOUNDARY = {"mae": 97.0, "mae_rush": 105.0, "mae_rain": 105.0, "p95_ms": 10.0}


def test_all_exact_thresholds_pass():
    result = decide(PROD, BOUNDARY)
    assert result["promote"] is True
    assert all(result["criteria"].values())


@pytest.mark.parametrize(
    "field,value,failed_check",
    [
        ("mae", 97.01, "overall_mae"),
        ("p95_ms", 10.01, "p95_latency"),
        ("mae_rush", 105.01, "rush_mae"),
        ("mae_rain", 105.01, "rain_mae"),
    ],
)
def test_each_criterion_can_block_promotion_on_its_own(field, value, failed_check):
    result = decide(PROD, {**BOUNDARY, field: value})
    assert result["promote"] is False
    assert {check for check, passed in result["criteria"].items() if not passed} == {
        failed_check
    }


def test_stricter_gain_override_blocks_candidate():
    candidate = {**BOUNDARY, "mae": 50.0}
    assert decide(PROD, candidate)["promote"] is True
    result = decide(PROD, candidate, min_rel_gain=0.60)
    assert result["criteria"]["overall_mae"] is False
    assert result["promote"] is False