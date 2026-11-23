import pytest

from msca_mas.calculations import (
    CalculationError,
    aggregate_named_values,
    calculate_shares,
    carbon_intensity,
    compare_named_values,
    equipment_emission,
    factor_reduction,
    material_emission,
    material_substitution,
    one_at_a_time_sensitivity,
    rank_named_values,
    transport_emission,
)


def test_transport_converts_kg_to_tonnes():
    result = transport_emission(2_000, "kg", 50, 0.1)
    assert result["result"] == pytest.approx(10.0)


def test_factor_reduction_is_deterministic():
    result = factor_reduction(1_000, 0.12)
    assert result["alternative"] == pytest.approx(880)
    assert result["delta"] == pytest.approx(-120)


def test_material_substitution_preserves_unsubstituted_share():
    result = material_substitution(1_000, 0.30, 300, 80)
    assert result["alternative"] == pytest.approx(780)
    assert result["reduction"] == pytest.approx(220)


def test_material_and_equipment_formulas_are_deterministic():
    material = material_emission(100.0, 2.0, 3.0)
    assert material["result"] == pytest.approx(600.0)
    equipment = equipment_emission(1_000.0, 50.0, 10.0, 0.5)
    assert equipment["inputs"]["machine_hours"] == pytest.approx(20.0)
    assert equipment["result"] == pytest.approx(100.0)


def test_negative_inputs_fail_closed():
    with pytest.raises(CalculationError):
        transport_emission(-1, "t", 10, 0.1)


def test_named_value_tools_aggregate_compare_share_and_rank():
    values = {"A": 10.0, "B": 30.0, "C": 60.0}
    assert aggregate_named_values(values)["total"] == pytest.approx(100.0)
    comparison = compare_named_values(values, baseline_name="A")
    assert comparison["maximum"] == {"name": "C", "value": 60.0}
    assert comparison["range"] == pytest.approx(50.0)
    assert sum(item["percentage"] for item in calculate_shares(values)["shares"]) == pytest.approx(100.0)
    assert [item["name"] for item in rank_named_values(values, limit=2)["ranked"]] == ["C", "B"]


def test_sensitivity_and_intensity_tools_are_deterministic():
    sensitivity = one_at_a_time_sensitivity(
        900.0,
        0.9,
        [1.0, 0.8, 0.9],
        parameter_name="plan_utilisation_factor",
    )
    assert [point["result"] for point in sensitivity["points"]] == pytest.approx([800.0, 900.0, 1_000.0])
    assert sensitivity["elasticity"] == 1.0
    assert carbon_intensity(1_000.0, 20.0, "m2")["result"] == pytest.approx(50.0)


def test_analytical_tools_fail_closed_on_invalid_denominators():
    with pytest.raises(CalculationError):
        calculate_shares({"A": 0.0, "B": 0.0})
    with pytest.raises(CalculationError):
        carbon_intensity(1.0, 0.0, "m2")
    with pytest.raises(CalculationError):
        one_at_a_time_sensitivity(1.0, 0.0, [1.0], parameter_name="x")
    with pytest.raises(CalculationError):
        equipment_emission(1.0, 0.0, 1.0, 1.0)
