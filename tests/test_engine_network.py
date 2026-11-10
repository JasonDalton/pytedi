"""Network routing invariants for the pytedi engine.

The existing engine tests use dams that all drain straight to the catchment outlet
(ds_damid = -99), which is the one topology that cannot exercise dam-to-dam routing.
These tests cover dams in series.
"""

import numpy as np
import pandas as pd
import pytest
from pytedi.main import tedimodel

# damlist columns, in order:
#   dam_id, ds_damid, surfacearea_m2, catchuptotal_km2, catchuplocal_km2, vol_ML,
#   year_start, group_name, demfactor, bypass_MLperT, bypass_start, bypass_end,
#   wfill_MLperT, wfill_start, wfill_end, output


def _two_dams(ds_damid_of_upstream, demfactor=0.0, downstream_local_km2=1.0,downstream_id=2,upstream_id=3):
    """A small upstream dam and a large downstream dam.

    Pass ds_damid_of_upstream=2 to put them in series (dam 3 drains into dam 2) or -99 to
    put them in parallel (both drain straight to the outlet). Steady inflow and no climate,
    so the upstream dam fills on the first timestep and spills from then on.
    """
    m = tedimodel()
    m.timestep = "day"
    m.calc_nat_or_obs = "obs"
    m.cat_area_km2 = 10.0

    # Use a string outlet marker for alphanumeric networks and an integer
    # outlet marker for legacy integer networks.
    if isinstance(downstream_id, str):
        outlet_id = "-99"
    else:
        outlet_id = -99
    
    # Large downstream dam
    m.damlist.loc[0] = [downstream_id, outlet_id,            5000.0, downstream_local_km2, downstream_local_km2, 50.0, 0, 9999, "Irrigation", demfactor, 0.0, 101, 1231, 0.0, 101, 1231, 1,]
    
    # Small upstream dam
    m.damlist.loc[1] = [upstream_id,   ds_damid_of_upstream, 1000.0, 1.0,                  1.0,                  5.0,  0, 9999, "Irrigation", demfactor, 0.0, 101, 1231, 0.0, 101, 1231, 1,]
    
    idx = pd.date_range("2000-01-01", periods=30, freq="D")
    ts = pd.DataFrame(index=idx)
    ts["rain"] = 0.0
    ts["evap"] = 0.0
    ts["flowup"] = 100.0         # 1 km2 of a 10 km2 catchment -> 10 ML/day
    ts["flowdown"] = np.nan
    ts["Irrigation"] = 0.1       # demand = capacity x demfactor x 0.1 per day
    m.ts = ts
    return m


def test_spill_from_an_upstream_dam_reaches_the_downstream_dam():
    """Dams in series must be connected: everything leaving the upstream dam has to
    arrive at the dam below it, not at the catchment outlet."""
    m = _two_dams(ds_damid_of_upstream=2)
    m.run_model()

    ind = m.indiv_results
    # Individual results must contain the original dam identifiers, not the engine's internal row indices.
    assert set(ind['dam_id'].unique()) == {2, 3}
    
    # Dam 3 drains into dam 2.
    upstream = ind[ind['dam_id'] == 3]
    downstream = ind[ind['dam_id'] == 2]

    assert upstream["outflow"].sum() > 0, "test is only meaningful when the upstream dam releases water"

    np.testing.assert_allclose(
        downstream["updam_spill"].to_numpy(dtype=float),
        upstream["outflow"].to_numpy(dtype=float),
        atol=1e-9,
        err_msg="upstream dam outflow did not arrive at the downstream dam",
    )


def test_a_dam_in_series_supplies_demand_it_could_not_meet_in_parallel():
    """A catchment-level check that needs no per-dam output. The downstream dam is given a
    small local catchment and a demand it cannot satisfy on its own, so routing the upstream
    dam into it must raise total catchment impact. Wired in parallel the two dams only
    intercept their own runoff, and total impact is bounded by capacity plus what each dam
    can supply from its own inflow."""
    kwargs = dict(demfactor=1.0, downstream_local_km2=0.1)

    series = _two_dams(ds_damid_of_upstream=2, **kwargs)
    series.run_model()

    parallel = _two_dams(ds_damid_of_upstream=-99, **kwargs)
    parallel.run_model()

    series_impact = series.catch_results["agg_impact"].sum()
    parallel_impact = parallel.catch_results["agg_impact"].sum()

    assert series_impact > parallel_impact, (
        f"series impact {series_impact:.3f} ML should exceed parallel "
        f"{parallel_impact:.3f} ML, since the downstream dam gains the upstream spill"
    )

def test_alphanumeric_dam_ids_are_preserved():
    """Alphanumeric IDs must survive preprocessing and appear in results."""

    m = _two_dams(ds_damid_of_upstream="DamPoint001",downstream_id="DamPoint001",upstream_id="DamPoly001",)

    m.run_model()

    ind = m.indiv_results

    # Both original identifiers must appear in the results.
    assert set(ind["dam_id"].unique()) == {"DamPoly001","DamPoint001",}

    # Select each dam using its original identifier.
    upstream = ind[ind["dam_id"] == "DamPoly001"]
    downstream = ind[ind["dam_id"] == "DamPoint001"]

    # Confirm that both dams produced individual results.
    assert not upstream.empty
    assert not downstream.empty

    # Make sure the test includes a period when the upstream dam releases water.
    assert upstream["outflow"].sum() > 0

    # Every upstream outflow must arrive at the downstream dam.
    np.testing.assert_allclose(
        downstream["updam_spill"].to_numpy(dtype=float),
        upstream["outflow"].to_numpy(dtype=float),
        atol=1e-9,
        err_msg=(
            "Outflow from DamPoly001 did not arrive at DamPoint001"
        ),
    )

def test_catchment_area_is_preserved_after_scaling():
    """Scaled dam areas plus the outlet area must equal the total catchment."""

    # Local dam catchments total 9 km2:
    # downstream dam = 8 km2
    # upstream dam = 1 km2
    m = _two_dams(
        ds_damid_of_upstream=2,
        downstream_local_km2=8.0,
    )

    m.cat_area_km2 = 10.0
    m.scale_limit = 0.8

    m.run_model()

    # The total local inflow represented by all dams and the outlet
    # must equal the supplied whole-catchment inflow.
    expected_flow = m.ts.loc[
        m.catch_results.index,
        "flowup",
    ]

    np.testing.assert_allclose(
        m.catch_results["flow_up"].to_numpy(dtype=float),
        expected_flow.to_numpy(dtype=float),
        atol=1e-9,
        err_msg=(
            "Catchment-area scaling did not preserve the total "
            "catchment inflow"
        ),
    )

def test_water_balance_closes():
    """Water entering storage must equal water leaving or remaining."""

    m = _two_dams(ds_damid_of_upstream=2, demfactor=1.0,)

    # Start both dams partly full.
    m.startstore_propfull = 0.2

    # Add bypass and winterfill so the test covers more engine operations.
    m.damlist.loc[:, "bypass_MLperT"] = [0.5, 0.25]
    m.damlist.loc[:, "wfill_MLperT"] = [1.0, 0.5]

    # Use varying conditions so the model encounters dry periods,
    # inflows, rainfall, evaporation, demand and high-flow periods.
    m.ts["flowup"]     = np.tile([0.0,  5.0, 20.0, 100.0,  5.0, 50.0], 5, )

    m.ts["rain"]       = np.tile([0.0,  2.0, 20.0,   0.0,  0.0, 10.0], 5, )

    m.ts["evap"]       = np.tile([5.0,  5.0,  2.0,   8.0,  5.0,  2.0], 5, )

    m.ts["Irrigation"] = np.tile([0.20, 0.10, 0.00,  0.05, 0.20, 0.00],5, )

    m.run_model()

    # ------------------------------------------------------------------
    # Whole-catchment water balance
    #
    # Start storage
    # + natural inflow
    # + actual winterfill
    # - actual net evaporation
    # - actual demand
    # - downstream flow
    # - end storage
    # = zero
    # ------------------------------------------------------------------
    catch = m.catch_results

    catch_residual = (
        catch["sstore"]
        + catch["flow_up"]
        + catch["wf_actual"]
        - catch["netevap_actual"]
        - catch["demand_actual"]
        - catch["flow_down"]
        - catch["estore"]
    )

    np.testing.assert_allclose(
        catch_residual.to_numpy(dtype=float),
        0.0,
        rtol=0.0,
        atol=1e-9,
        err_msg="The whole-catchment water balance does not close",
    )

    # ------------------------------------------------------------------
    # Individual-dam water balance
    #
    # Start storage
    # + local inflow
    # + inflow received from upstream dams
    # + actual winterfill
    # - actual net evaporation
    # - actual demand
    # - dam outflow
    # - end storage
    # = zero
    # ------------------------------------------------------------------
    individual = m.indiv_results

    individual_residual = (
        individual["sstore"]
        + individual["local_inflow"]
        + individual["updam_spill"]
        + individual["wf_actual"]
        - individual["netevap_actual"]
        - individual["demand_actual"]
        - individual["outflow"]
        - individual["es"]
    )

    np.testing.assert_allclose(
        individual_residual.to_numpy(dtype=float),
        0.0,
        rtol=0.0,
        atol=1e-9,
        err_msg="One or more individual-dam water balances do not close",
    )

def test_empty_dam_storage_cannot_become_negative():
    """Evaporation and demand must be curtailed when a dam is empty."""

    m = _two_dams(ds_damid_of_upstream=2,demfactor=1.0,)

    m.startstore_propfull = 0.0

    # No water enters the dams.
    m.ts["flowup"] = 0.0
    m.ts["rain"] = 0.0

    # Deliberately impose evaporation and demand on empty dams.
    m.ts["evap"] = 100.0
    m.ts["Irrigation"] = 1.0

    # Prevent winterfill from supplying water.
    m.damlist.loc[:, "wfill_MLperT"] = 0.0

    m.run_model()

    individual = m.indiv_results

    # Ensure the test actually requested evaporation and demand.
    assert (individual["netevap"] > 0.0).all()
    assert (individual["demand"] > 0.0).all()

    # Empty storage cannot supply either flux.
    np.testing.assert_allclose(
        individual["netevap_actual"].to_numpy(dtype=float),
        0.0,
        rtol=0.0,
        atol=1e-9,
    )

    np.testing.assert_allclose(
        individual["demand_actual"].to_numpy(dtype=float),
        0.0,
        rtol=0.0,
        atol=1e-9,
    )

    # Storage must remain exactly zero.
    np.testing.assert_allclose(
        individual["es"].to_numpy(dtype=float),
        0.0,
        rtol=0.0,
        atol=1e-9,
    )

    assert (individual["es"] >= -1e-9).all()

def test_dam_storage_cannot_exceed_capacity():
    """Excess inflow must spill rather than increasing storage above capacity."""

    m = _two_dams(ds_damid_of_upstream=2,demfactor=0.0,)

    m.startstore_propfull = 0.0

    # Supply enough flow to fill both dams immediately.
    m.ts["flowup"] = 1000.0
    m.ts["rain"] = 0.0
    m.ts["evap"] = 0.0
    m.ts["Irrigation"] = 0.0

    m.damlist.loc[:, "wfill_MLperT"] = 0.0
    m.damlist.loc[:, "bypass_MLperT"] = 0.0

    m.run_model()

    individual = m.indiv_results

    # Capacities defined by _two_dams():
    # dam 2 = 50 ML
    # dam 3 = 5 ML
    capacity = individual["dam_id"].map({
        2: 50.0,
        3: 5.0,
    })

    assert capacity.notna().all()

    # Storage must never be negative or greater than capacity.
    assert (individual["es"] >= -1e-9).all()

    assert (individual["es"] <= capacity + 1e-9).all(), ("One or more dams exceeded their storage capacity")

    # The test must actually produce spills.
    assert (individual["spills"] > 0.0).any(), ("The test did not generate any spills")

    # With this very large inflow, both dams should be full.
    np.testing.assert_allclose(
        individual["es"].to_numpy(dtype=float),
        capacity.to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-9,
    )
def test_winterfill_is_limited_by_available_storage():
    """Winterfill must stop at capacity and switch off when a dam is full."""

    m = _two_dams(ds_damid_of_upstream=2, demfactor=0.0, )

    m.startstore_propfull = 0.0

    # Remove all other water-balance fluxes.
    m.ts["flowup"] = 0.0
    m.ts["rain"] = 0.0
    m.ts["evap"] = 0.0
    m.ts["Irrigation"] = 0.0

    # Request much more winterfill than either dam can hold.
    m.damlist.loc[:, "wfill_MLperT"] = 100.0
    m.damlist.loc[:, "bypass_MLperT"] = 0.0

    m.run_model()

    individual = m.indiv_results

    capacity = individual["dam_id"].map({
        2: 50.0,
        3: 5.0,
    })

    assert capacity.notna().all()

    first_step = individual["istep"] == 0
    later_steps = individual["istep"] > 0

    # On the first timestep, actual winterfill must equal the available
    # storage capacity, rather than the requested 100 ML.
    np.testing.assert_allclose(
        individual.loc[first_step, "wf_actual"].to_numpy(dtype=float),
        capacity.loc[first_step].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-9,
    )

    # Both dams should finish the first timestep exactly full.
    np.testing.assert_allclose(
        individual.loc[first_step, "es"].to_numpy(dtype=float),
        capacity.loc[first_step].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-9,
    )

    # Once full, the dams must not take any additional winterfill.
    np.testing.assert_allclose(
        individual.loc[later_steps, "wf_actual"].to_numpy(dtype=float),
        0.0,
        rtol=0.0,
        atol=1e-9,
    )

    # Storage must remain at capacity without exceeding it.
    np.testing.assert_allclose(
        individual.loc[later_steps, "es"].to_numpy(dtype=float),
        capacity.loc[later_steps].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-9,
    )

def test_network_cycle_is_rejected():
    """A closed loop in the dam network must raise a clear error."""

    m = tedimodel()

    m.timestep = "day"
    m.calc_nat_or_obs = "obs"
    m.cat_area_km2 = 10.0

    # Deliberately create a closed network loop:
    #
    # Dam01 -> Dam02 -> Dam03 -> Dam01
    #
    # There is no path from these dams to the catchment outlet.
    m.damlist.loc[0] = [
        "Dam01",        # dam_id
        "Dam02",        # ds_damid
        1000.0,         # surfacearea_m2
        1.0,            # catchuptotal_km2
        1.0,            # catchuplocal_km2
        5.0,            # vol_ML
        0,              # year_start
        9999,           # year_end
        "Irrigation",   # group_name
        0.0,            # demfactor
        0.0,            # bypass_MLperT
        101,             # bypass_start_mmdd
        1231,            # bypass_end_mmdd
        0.0,            # wfill_MLperT
        101,             # wfill_start_mmdd
        1231,            # wfill_end_mmdd
        1,               # output
    ]

    m.damlist.loc[1] = [
        "Dam02",
        "Dam03",
        1000.0,
        1.0,
        1.0,
        5.0,
        0,
        9999,
        "Irrigation",
        0.0,
        0.0,
        101,
        1231,
        0.0,
        101,
        1231,
        1,
    ]

    m.damlist.loc[2] = [
        "Dam03",
        "Dam01",
        1000.0,
        1.0,
        1.0,
        5.0,
        0,
        9999,
        "Irrigation",
        0.0,
        0.0,
        101,
        1231,
        0.0,
        101,
        1231,
        1,
    ]

    dates = pd.date_range("2000-01-01",periods=3,freq="D",)

    m.ts = pd.DataFrame({"rain": 0.0,
                         "evap": 0.0,
                         "flowup": 10.0,
                         "flowdown": np.nan,
                         "Irrigation": 0.0,},
                        index=dates,)

    # The model must reject the network before starting the calculations.
    with pytest.raises(ValueError, match="(?i)cycle") as error:
        m.run_model()

    # The error should identify the dams involved, making the input
    # problem easier for a user to locate.
    error_message = str(error.value)

    assert "Dam01" in error_message
    assert "Dam02" in error_message
    assert "Dam03" in error_message
    
def test_dam_cannot_drain_to_itself():
    """A dam whose downstream dam is itself must be rejected."""

    m = _two_dams(
        downstream_id="DamPoint001",
        upstream_id="DamPoly001",

        # Deliberate error:
        # DamPoly001 drains to DamPoly001.
        ds_damid_of_upstream="DamPoly001",
    )

    with pytest.raises(ValueError, match="(?i)cycle") as error:
        m.run_model()

    error_message = str(error.value)

    # The error should identify the self-referencing dam.
    assert "DamPoly001" in error_message
    
def test_duplicate_dam_ids_are_rejected():
    """Every dam must have a unique user-facing identifier."""

    m = _two_dams(
        downstream_id="Dam01",
        upstream_id="Dam01",

        # Both dams drain directly to the outlet, so duplication is
        # the only deliberate error in this model.
        ds_damid_of_upstream="-99",
    )

    with pytest.raises(ValueError, match="(?i)duplicate") as error:
        m.run_model()

    error_message = str(error.value)

    # The error should identify the duplicated value.
    assert "Dam01" in error_message
    
def test_unknown_downstream_dam_id_is_rejected():
    """Every non-outlet ds_damid must identify an existing dam."""

    m = _two_dams(
        downstream_id="DamPoint001",
        upstream_id="DamPoly001",

        # Deliberate error: this dam does not exist.
        ds_damid_of_upstream="DamMissing999",
    )

    with pytest.raises(ValueError) as error:
        m.run_model()

    error_message = str(error.value)

    # The message should explain which field and value caused the problem.
    assert "downstream dam id" in error_message
    assert "DamMissing999" in error_message

def test_missing_dam_id_is_rejected():
    """Every dam record must have a non-missing identifier."""

    m = _two_dams(ds_damid_of_upstream=2)

    # Deliberately remove the first dam's identifier.
    m.damlist.loc[0, "dam_id"] = pd.NA

    with pytest.raises(ValueError) as error:
        m.run_model()

    error_message = str(error.value).lower()

    # Check the meaning without requiring one exact sentence.
    assert "dam" in error_message
    assert "missing" in error_message

def test_unknown_demand_group_is_rejected():
    """Every dam demand group must have a matching timeseries column."""

    m = _two_dams(ds_damid_of_upstream=2)

    # Deliberate error: m.ts has an Irrigation column, but no
    # Industrial demand column.
    m.damlist.loc[0, "group_name"] = "Industrial"

    with pytest.raises(ValueError) as error:
        m.run_model()

    error_message = str(error.value)

    assert "demand" in error_message.lower()
    assert "Industrial" in error_message

def test_negative_dam_capacity_is_rejected():
    """A dam cannot have a negative storage capacity."""

    m = _two_dams(ds_damid_of_upstream=2)

    # Deliberately assign an impossible capacity.
    m.damlist.loc[0, "vol_ML"] = -5.0

    with pytest.raises(ValueError) as error:
        m.run_model()

    error_message = str(error.value).lower()

    assert "vol_ml" in error_message or "capacity" in error_message
    assert "negative" in error_message
    
def test_negative_dam_surface_area_is_rejected():
    """A dam cannot have a negative surface area."""

    m = _two_dams(ds_damid_of_upstream=2)

    # Deliberately assign an impossible surface area.
    m.damlist.loc[0, "surfacearea_m2"] = -100.0

    with pytest.raises(ValueError) as error:
        m.run_model()

    error_message = str(error.value).lower()

    assert "surface" in error_message
    assert "negative" in error_message
    
def test_negative_local_catchment_area_is_rejected():
    """A dam cannot have a negative local catchment area."""

    m = _two_dams(ds_damid_of_upstream=2)

    # Deliberately assign an impossible local catchment area.
    m.damlist.loc[0, "catchuplocal_km2"] = -1.0

    with pytest.raises(ValueError) as error:
        m.run_model()

    error_message = str(error.value).lower()

    assert "catchment" in error_message
    assert "negative" in error_message

def test_negative_total_catchment_area_is_rejected():
    """A dam cannot have a negative total upstream catchment area."""

    m = _two_dams(ds_damid_of_upstream=2)

    # Deliberately assign an impossible total catchment area.
    m.damlist.loc[0, "catchuptotal_km2"] = -1.0

    with pytest.raises(ValueError) as error:
        m.run_model()

    error_message = str(error.value).lower()

    assert "catchment" in error_message
    assert "negative" in error_message

@pytest.mark.parametrize(
    "invalid_catchment_area",
    [0.0, -10.0],
)
def test_nonpositive_total_catchment_area_is_rejected(invalid_catchment_area):
    """The model catchment area must be greater than zero."""

    m = _two_dams(ds_damid_of_upstream=2)
    m.cat_area_km2 = invalid_catchment_area

    with pytest.raises(ValueError) as error:
        m.run_model()

    error_message = str(error.value).lower()

    assert "catchment" in error_message

@pytest.mark.parametrize(
    "invalid_scale_limit",
    [-0.1, 1.1, np.nan],
)
def test_invalid_scale_limit_is_rejected(invalid_scale_limit):
    """scale_limit must be a finite proportion between zero and one."""

    m = _two_dams(ds_damid_of_upstream=2)
    m.scale_limit = invalid_scale_limit

    with pytest.raises(ValueError) as error:
        m.run_model()

    error_message = str(error.value).lower()

    assert "scale_limit" in error_message
    
@pytest.mark.parametrize(
    "column",
    [
        "vol_ML",
        "surfacearea_m2",
        "catchuplocal_km2",
        "catchuptotal_km2",
    ],
)
def test_missing_physical_dam_value_is_rejected(column):
    """Required physical dam properties cannot be missing."""

    m = _two_dams(ds_damid_of_upstream=2)

    # Deliberately remove one required value.
    m.damlist.loc[0, column] = np.nan

    with pytest.raises(ValueError) as error:
        m.run_model()

    error_message = str(error.value)

    # The message should identify the problematic column and dam.
    assert column.lower() in error_message.lower()
    assert "2" in error_message

def test_reverse_model_reconstructs_known_upstream_flow():
    """Reverse modelling should recover a known upstream flow series."""

    upstream_flow = np.tile([80.0, 100.0, 120.0, 150.0, 90.0, 110.0], 5, )

    # ------------------------------------------------------------------
    # First run: calculate downstream flow from known upstream flow.
    # ------------------------------------------------------------------
    forward = _two_dams(ds_damid_of_upstream=2, demfactor=0.1, )

    forward.calc_nat_or_obs = "obs"
    forward.startstore_propfull = 1.0

    forward.ts["flowup"] = upstream_flow
    forward.ts["rain"] = 0.0
    forward.ts["evap"] = 0.0
    forward.ts["Irrigation"] = 0.1

    forward.run_model()

    calculated_downstream = (forward.catch_results["flow_down"].copy())

    # Confirm that the dams had a measurable effect. Otherwise, the
    # round-trip test would only be checking an unchanged flow series.
    assert ( np.abs( upstream_flow - calculated_downstream.to_numpy(dtype=float) ).max() > 0.1 )

    # ------------------------------------------------------------------
    # Second run: reconstruct upstream flow from downstream flow.
    # ------------------------------------------------------------------
    reverse = _two_dams( ds_damid_of_upstream=2, demfactor=0.1, )

    reverse.calc_nat_or_obs = "nat"
    reverse.startstore_propfull = 1.0

    reverse.ts["flowup"] = np.nan
    reverse.ts["flowdown"] = calculated_downstream.to_numpy(dtype=float)
    reverse.ts["rain"] = 0.0
    reverse.ts["evap"] = 0.0
    reverse.ts["Irrigation"] = 0.1

    reverse.run_model()

    reconstructed_upstream = reverse.catch_results["flow_up"]

    # The reverse solver uses a 0.02 ML convergence tolerance, so allow
    # slightly more than that for numerical rounding.
    np.testing.assert_allclose(
        reconstructed_upstream.to_numpy(dtype=float),
        upstream_flow,
        rtol=0.0,
        atol=0.03,
        err_msg=(
            "Reverse modelling did not reconstruct the known upstream flow"
        ),
    )

    # The final downstream flows calculated during the reverse run
    # should also reproduce the flows supplied to that run.
    np.testing.assert_allclose(
        reverse.catch_results["flow_down"].to_numpy(dtype=float),
        calculated_downstream.to_numpy(dtype=float),
        rtol=0.0,
        atol=0.03,
        err_msg=(
            "Reverse-model downstream flows do not match the supplied downstream flows"
        ),
    )

def test_reverse_model_with_changing_storage():
    """Reverse modelling should work while dam storage changes."""

    upstream_flow = np.tile([5.0, 15.0, 40.0, 100.0, 10.0, 60.0],  5, )
    rain =          np.tile([0.0,  2.0, 20.0,   0.0,  0.0, 10.0],  5, )
    evap =          np.tile([5.0,  5.0,  2.0,   8.0,  5.0,  2.0],  5,  )
    demand =        np.tile([0.20, 0.10, 0.00,  0.05, 0.20, 0.00], 5, )

    # ------------------------------------------------------------------
    # Forward run
    # ------------------------------------------------------------------
    forward = _two_dams(ds_damid_of_upstream=2, demfactor=0.5, )

    forward.calc_nat_or_obs = "obs"
    forward.startstore_propfull = 0.0

    forward.ts["flowup"] = upstream_flow
    forward.ts["rain"] = rain
    forward.ts["evap"] = evap
    forward.ts["Irrigation"] = demand

    forward.run_model()

    calculated_downstream = (forward.catch_results["flow_down"].copy())

    # Confirm this is genuinely a changing-storage test.
    assert forward.catch_results["estore"].max() > (forward.catch_results["estore"].min())

    # Confirm that at least one dam eventually becomes full and spills.
    assert (forward.catch_results["spills"] > 0.0).any()

    # ------------------------------------------------------------------
    # Reverse run
    # ------------------------------------------------------------------
    reverse = _two_dams(ds_damid_of_upstream=2, demfactor=0.5, )

    reverse.calc_nat_or_obs = "nat"
    reverse.startstore_propfull = 0.0

    reverse.ts["flowup"] = np.nan
    reverse.ts["flowdown"] = calculated_downstream.to_numpy(dtype=float)
    reverse.ts["rain"] = rain
    reverse.ts["evap"] = evap
    reverse.ts["Irrigation"] = demand

    reverse.run_model()

    reconstructed_upstream = (reverse.catch_results["flow_up"].to_numpy(dtype=float))

    np.testing.assert_allclose(
        reconstructed_upstream,
        upstream_flow,
        rtol=0.0,
        atol=0.05,
        err_msg=(
            "Reverse modelling did not reconstruct upstream flow "
            "while dam storage was changing"
        ),
    )

    np.testing.assert_allclose(
        reverse.catch_results["flow_down"].to_numpy(dtype=float),
        calculated_downstream.to_numpy(dtype=float),
        rtol=0.0,
        atol=0.05,
        err_msg=(
            "Reverse-model downstream flow does not match the "
            "supplied downstream flow"
        ),
    )

def test_monthly_model_preserves_dates_and_water_balance():
    """Monthly models must return one balanced result per input month."""

    m = _two_dams(ds_damid_of_upstream=2,demfactor=0.2,)

    m.timestep = "month"
    m.startstore_propfull = 0.25

    dates = pd.date_range("2000-01-01",periods=24, freq="MS",)

    m.ts = pd.DataFrame(
        {
            "rain":       np.tile([ 20.0, 10.0,  5.0,  5.0, 10.0, 30.0, 50.0, 60.0, 40.0, 30.0, 20.0, 15.0], 2, ),
            "evap":       np.tile([100.0, 80.0, 60.0, 40.0, 30.0, 20.0, 20.0, 25.0, 35.0, 50.0, 70.0, 90.0], 2, ),
            "flowup":     np.tile([ 20.0, 15.0, 10.0,  8.0, 12.0, 30.0, 60.0, 80.0, 50.0, 35.0, 25.0, 20.0], 2, ),
            "flowdown":   np.nan,
            "Irrigation": np.tile([  0.10, 0.10, 0.10, 0.08, 0.05, 0.02, 0.00, 0.00, 0.02, 0.05, 0.08, 0.10], 2,),
        },
        index=dates,
    )

    m.run_model()

    results = m.catch_results

    # There must be exactly one result for each input month.
    assert len(results) == 24

    # The monthly dates must be retained exactly.
    pd.testing.assert_index_equal(results.index, dates, )

    # Check the monthly whole-catchment water balance.
    residual = (
        results["sstore"]
        + results["flow_up"]
        + results["wf_actual"]
        - results["netevap_actual"]
        - results["demand_actual"]
        - results["flow_down"]
        - results["estore"]
    )

    np.testing.assert_allclose(
        residual.to_numpy(dtype=float),
        0.0,
        rtol=0.0,
        atol=1e-9,
        err_msg="The monthly water balance does not close",
    )

    # Each output dam should have one individual result per month.
    counts = m.indiv_results.groupby("dam_id").size()

    assert (counts == 24).all()

def test_daily_model_handles_leap_day():
    """Daily processing must retain 29 February without skipping a timestep."""

    m = _two_dams(ds_damid_of_upstream=2, demfactor=0.2,)

    m.timestep = "day"
    m.startstore_propfull = 0.25

    dates = pd.date_range("2000-02-27", "2000-03-02", freq="D",)

    m.ts = pd.DataFrame(
        {
            "rain": [0.0, 5.0, 20.0, 0.0, 0.0],
            "evap": [5.0, 4.0, 2.0, 5.0, 5.0],
            "flowup": [10.0, 20.0, 50.0, 30.0, 15.0],
            "flowdown": np.nan,
            "Irrigation": [0.10, 0.10, 0.05, 0.05, 0.10],
        },
        index=dates,
    )

    m.run_model()

    results = m.catch_results

    assert len(results) == 5

    pd.testing.assert_index_equal(results.index, dates, )

    assert pd.Timestamp("2000-02-29") in results.index

    # Confirm that leap-day processing still closes the water balance.
    residual = (
        results["sstore"]
        + results["flow_up"]
        + results["wf_actual"]
        - results["netevap_actual"]
        - results["demand_actual"]
        - results["flow_down"]
        - results["estore"]
    )

    np.testing.assert_allclose(
        residual.to_numpy(dtype=float),
        0.0,
        rtol=0.0,
        atol=1e-9,
        err_msg="The daily water balance does not close across leap day",
    )

    # Each output dam should have one result for every day.
    counts = m.indiv_results.groupby("dam_id").size()

    assert (counts == 5).all()
    
@pytest.mark.parametrize(
    "column, expected_description",
    [
        ("flowup", "flow"),
        ("rain", "rainfall"),
        ("Irrigation", "demand"),
    ],
)
def test_negative_required_timeseries_value_is_rejected(column,expected_description):
    """Flow, rainfall and demand cannot contain negative values."""

    m = _two_dams(ds_damid_of_upstream=2)

    bad_date = m.ts.index[5]

    # Deliberately insert one negative value.
    m.ts.loc[bad_date, column] = -1.0

    with pytest.raises(ValueError) as error:
        m.run_model()

    error_message = str(error.value).lower()

    assert "negative" in error_message
    assert expected_description in error_message
    assert str(bad_date.date()) in error_message

def test_negative_evaporation_is_allowed():
    """Negative evaporation values are valid and must not be rejected.
    
    There may be a situation where net evap is placed in the evap column, this is valid."""

    m = _two_dams(ds_damid_of_upstream=2)

    test_date = m.ts.index[5]
    m.ts.loc[test_date, "evap"] = -5.0

    m.run_model()

    # Confirm that the negative evaporation reached the engine and
    # produced negative net evaporation, equivalent to a water gain.
    assert m.catch_results.loc[test_date, "netevap"] < 0.0

    assert np.isfinite(m.catch_results.to_numpy(dtype=float)).all()
    
def test_historical_scaling_above_one_is_allowed():
    """Historical scaling may legitimately exceed one.
    This allows for projections of future increased levels of development."""

    m = _two_dams(ds_damid_of_upstream=2)

    m.histoscale.loc[:, "histoscale"] = 1.25

    m.run_model()

    assert len(m.catch_results) == len(m.ts)

    assert np.isfinite(m.catch_results.to_numpy(dtype=float)).all()

@pytest.mark.parametrize(
    "column, expected_description",
    [
        ("flowup", "flow"),
        ("rain", "rainfall"),
        ("evap", "evaporation"),
        ("Irrigation", "demand"),
    ],
)
def test_internal_missing_timeseries_value_is_rejected(column,expected_description):
    """Required timeseries cannot contain missing values internally."""

    m = _two_dams(ds_damid_of_upstream=2)

    # Choose an internal date rather than the start or end.
    bad_date = m.ts.index[5]
    m.ts.loc[bad_date, column] = np.nan

    with pytest.raises(ValueError) as error:
        m.run_model()

    error_message = str(error.value).lower()

    assert "missing" in error_message
    assert expected_description in error_message
    assert str(bad_date.date()) in error_message
    
@pytest.mark.parametrize(
    "column, invalid_value, expected_description",
    [
        ("flowup", np.inf, "flow"),
        ("evap", -np.inf, "evaporation"),
        ("Irrigation", np.inf, "demand"),
    ],
)
def test_infinite_timeseries_value_is_rejected(column,invalid_value,expected_description):
    """All timeseries values must be finite."""

    m = _two_dams(ds_damid_of_upstream=2)

    bad_date = m.ts.index[5]
    m.ts.loc[bad_date, column] = invalid_value

    with pytest.raises(ValueError) as error:
        m.run_model()

    error_message = str(error.value).lower()

    assert "infinite" in error_message
    assert expected_description in error_message
    assert str(bad_date.date()) in error_message
    
def test_duplicate_timeseries_dates_are_rejected():
    """Timeseries indexes cannot contain duplicate dates."""

    m = _two_dams(ds_damid_of_upstream=2)

    duplicate_date = m.ts.index[5]

    # Insert a second copy of one complete timeseries row.
    m.ts = pd.concat([m.ts.iloc[:6],
                      m.ts.iloc[[5]],
                      m.ts.iloc[6:], ])

    assert m.ts.index.has_duplicates

    with pytest.raises(ValueError) as error:
        m.run_model()

    error_message = str(error.value).lower()

    assert "duplicate" in error_message
    assert str(duplicate_date.date()) in error_message
    
def test_unsorted_timeseries_dates_are_rejected():
    """Timeseries dates must be arranged in ascending order."""

    m = _two_dams(ds_damid_of_upstream=2)

    # Swap two internal rows while retaining the original first and
    # last dates. This creates an unsorted index without changing the
    # overall modelling period.
    row_order = list(range(len(m.ts)))
    row_order[5], row_order[6] = row_order[6], row_order[5]

    m.ts = m.ts.iloc[row_order].copy()

    assert not m.ts.index.is_monotonic_increasing

    with pytest.raises(ValueError) as error:
        m.run_model()

    error_message = str(error.value).lower()

    assert "ascending" in error_message
    
def test_mismatched_timeseries_indexes_are_rejected():
    """All active model inputs must contain the same dates."""

    m = _two_dams(ds_damid_of_upstream=2)

    missing_date = m.ts.index[5]

    # Remove one complete day from the model timeseries. The resampled
    # historical-scaling series still contains this date, producing an
    # index mismatch.
    m.ts = m.ts.drop(index=missing_date)

    assert missing_date not in m.ts.index

    with pytest.raises(ValueError) as error:
        m.run_model()

    error_message = str(error.value).lower()

    assert "do not match" in error_message
    assert "historical scaling" in error_message

def test_negative_downstream_flow_is_rejected_in_reverse_mode():
    """Reverse modelling must reject negative downstream flow."""

    m = _two_dams(ds_damid_of_upstream=2)

    m.calc_nat_or_obs = "nat"

    # In reverse mode, flowdown is the active input.
    m.ts["flowup"] = np.nan
    m.ts["flowdown"] = 10.0

    bad_date = m.ts.index[5]
    m.ts.loc[bad_date, "flowdown"] = -1.0

    with pytest.raises(ValueError) as error:
        m.run_model()

    error_message = str(error.value).lower()

    assert "negative" in error_message
    assert "flow" in error_message
    assert str(bad_date.date()) in error_message
    
@pytest.mark.parametrize(
    "column, invalid_mmdd",
    [
        ("bypass_start_mmdd", 0),
        ("bypass_end_mmdd", 1332),
        ("wfill_start_mmdd", 230),
        ("wfill_end_mmdd", 431),
    ],
)
def test_invalid_seasonal_mmdd_is_rejected(column,invalid_mmdd):
    """Bypass and winterfill dates must be valid calendar dates."""

    m = _two_dams(ds_damid_of_upstream=2)

    m.damlist.loc[0, column] = invalid_mmdd

    with pytest.raises(ValueError) as error:
        m.run_model()

    error_message = str(error.value)

    assert column.lower() in error_message.lower()
    assert str(invalid_mmdd) in error_message
    
def test_reverse_solver_meets_mixed_residual_tolerance_across_flow_magnitudes():
    """The reverse solver must meet its absolute-plus-relative downstream residual tolerance from low to very high flows."""

    upstream_flow = np.tile([50.0, 100.0, 10000.0, 1000000.0, 500.0, 50000.0], 5)

    forward = _two_dams(ds_damid_of_upstream=2, demfactor=0.1)
    forward.calc_nat_or_obs = "obs"
    forward.startstore_propfull = 1.0
    forward.ts["flowup"] = upstream_flow
    forward.ts["rain"] = 0.0
    forward.ts["evap"] = 0.0
    forward.ts["Irrigation"] = 0.1
    forward.run_model()

    observed_downstream = forward.catch_results["flow_down"].copy()

    reverse = _two_dams(ds_damid_of_upstream=2, demfactor=0.1)
    reverse.calc_nat_or_obs = "nat"
    reverse.startstore_propfull = 1.0
    reverse.ts["flowup"] = np.nan
    reverse.ts["flowdown"] = observed_downstream.to_numpy(dtype=float)
    reverse.ts["rain"] = 0.0
    reverse.ts["evap"] = 0.0
    reverse.ts["Irrigation"] = 0.1
    reverse.run_model()

    downstream_residual = np.abs(reverse.catch_results["flow_down"].to_numpy(dtype=float) - observed_downstream.to_numpy(dtype=float))
    residual_tolerance = 0.02 + 1e-6 * np.abs(observed_downstream.to_numpy(dtype=float))

    assert np.all(downstream_residual <= residual_tolerance), f"Maximum downstream residual was {downstream_residual.max():.12g} ML"
    
def test_reverse_solver_extrapolates_when_no_nonnegative_exact_solution_exists():
    """An infeasible reverse timestep uses the recent reconstructed-flow trend."""

    m = tedimodel()
    m.timestep = "day"
    m.calc_nat_or_obs = "nat"
    m.cat_area_km2 = 1.0
    m.scale_limit = 1.0
    m.startstore_propfull = 1.0

    m.damlist.loc[0] = ["Dam001", -99, 1_000_000.0, 1.0, 1.0, 10.0, 0, 9999, "Irrigation", 0.0, 0.0, 101, 1231, 0.0, 101, 1231, 1]

    index = pd.date_range("2000-01-01", periods=3, freq="D")

    m.ts = pd.DataFrame(index=index)
    m.ts["rain"] = [0.0, 0.0, 10.0]
    m.ts["evap"] = 0.0
    m.ts["flowup"] = np.nan
    m.ts["flowdown"] = [5.0, 4.0, 0.0]
    m.ts["Irrigation"] = 0.0

    with pytest.warns(RuntimeWarning, match="two-timestep linear extrapolation") as warning_record:
        m.run_model()

    results = m.catch_results

    expected_fallback_flow = max(0.0, 2.0 * results.iloc[1]["flow_up"] - results.iloc[0]["flow_up"])

    assert results.iloc[2]["flow_up"] == pytest.approx(expected_fallback_flow)
    assert results.iloc[2]["flow_up"] > 0.0

    # Direct rainfall on the full dam means that the calculated downstream flow cannot match the zero-flow observation.
    assert results.iloc[2]["flow_down"] > m.ts.iloc[2]["flowdown"]

    warning_message = str(warning_record[0].message)

    assert "1 timestep" in warning_message
    assert "2000-01-03" in warning_message
    assert "largest downstream residual" in warning_message.lower()

    # The physical water balance must still close, despite the mismatch with the observation.
    np.testing.assert_allclose(results["sstore"] + results["flow_up"] - results["netevap"] - results["demand"] + results["wf_actual"], results["estore"] + results["flow_down"], atol=1e-8)

def test_decommissioned_dam_releases_storage_on_first_inactive_timestep():
    """A dam releases its stored water as spill on the first timestep of year_end."""

    m = tedimodel()
    m.timestep = "day"
    m.calc_nat_or_obs = "obs"
    m.cat_area_km2 = 1.0
    m.scale_limit = 1.0
    m.startstore_propfull = 1.0

    m.damlist.loc[0] = ["Dam001", -99, 1000.0, 1.0, 1.0, 10.0, 0, 2000, "Irrigation", 0.0, 0.0, 101, 1231, 0.0, 101, 1231, 1]

    index = pd.date_range("1999-12-31", periods=3, freq="D")

    m.ts = pd.DataFrame(index=index)
    m.ts["rain"] = 0.0
    m.ts["evap"] = 0.0
    m.ts["flowup"] = 2.0
    m.ts["flowdown"] = np.nan
    m.ts["Irrigation"] = 0.0

    m.run_model()

    results = m.catch_results

    # On 31 December the dam is active, full, and spills only its 2 ML inflow.
    assert results.loc["1999-12-31", "sstore"] == pytest.approx(10.0)
    assert results.loc["1999-12-31", "estore"] == pytest.approx(10.0)
    assert results.loc["1999-12-31", "spills"] == pytest.approx(2.0)
    assert results.loc["1999-12-31", "flow_down"] == pytest.approx(2.0)

    # On 1 January the dam is decommissioned. Its 10 ML storage and 2 ML inflow are released.
    assert results.loc["2000-01-01", "sstore"] == pytest.approx(10.0)
    assert results.loc["2000-01-01", "estore"] == pytest.approx(0.0)
    assert results.loc["2000-01-01", "spills"] == pytest.approx(12.0)
    assert results.loc["2000-01-01", "flow_down"] == pytest.approx(12.0)

    # On later timesteps, storage remains zero and inflow passes straight through.
    assert results.loc["2000-01-02", "sstore"] == pytest.approx(0.0)
    assert results.loc["2000-01-02", "estore"] == pytest.approx(0.0)
    assert results.loc["2000-01-02", "spills"] == pytest.approx(2.0)
    assert results.loc["2000-01-02", "flow_down"] == pytest.approx(2.0)

    # With no climate or demand, the water balance must close on every timestep.
    np.testing.assert_allclose(results["sstore"] + results["flow_up"], results["estore"] + results["flow_down"], atol=1e-10)

def test_dam_decommissioned_before_model_start_has_no_initial_storage():
    """startstore_propfull must not initialize a dam that was already decommissioned."""

    m = tedimodel()
    m.timestep = "day"
    m.calc_nat_or_obs = "obs"
    m.cat_area_km2 = 1.0
    m.scale_limit = 1.0
    m.startstore_propfull = 1.0

    m.damlist.loc[0] = ["Dam001", -99, 1000.0, 1.0, 1.0, 10.0, 0, 2000, "Irrigation", 0.0, 0.0, 101, 1231, 0.0, 101, 1231, 1]

    index = pd.date_range("2001-01-01", periods=3, freq="D")

    m.ts = pd.DataFrame(index=index)
    m.ts["rain"] = 0.0
    m.ts["evap"] = 0.0
    m.ts["flowup"] = 2.0
    m.ts["flowdown"] = np.nan
    m.ts["Irrigation"] = 0.0

    m.run_model()

    results = m.catch_results

    # Although startstore_propfull is 1.0, the already-decommissioned dam must start empty.
    np.testing.assert_allclose(results["sstore"], [0.0, 0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(results["estore"], [0.0, 0.0, 0.0], atol=1e-10)

    # The 2 ML inflow passes straight through on every timestep, with no artificial 10 ML release.
    np.testing.assert_allclose(results["spills"], [2.0, 2.0, 2.0], atol=1e-10)
    np.testing.assert_allclose(results["flow_down"], [2.0, 2.0, 2.0], atol=1e-10)

    # The water balance remains closed.
    np.testing.assert_allclose(results["sstore"] + results["flow_up"], results["estore"] + results["flow_down"], atol=1e-10)

def test_decommissioned_dam_has_no_operational_fluxes():
    """Demand, evaporation, bypass and winterfill must stop after decommissioning."""

    m = tedimodel()
    m.timestep = "day"
    m.calc_nat_or_obs = "obs"
    m.cat_area_km2 = 1.0
    m.scale_limit = 1.0
    m.startstore_propfull = 1.0

    m.damlist.loc[0] = ["Dam001", -99, 1000.0, 1.0, 1.0, 10.0, 0, 2000, "Irrigation", 1.0, 1.0, 101, 1231, 1.0, 101, 1231, 1]

    index = pd.date_range("2001-01-01", periods=3, freq="D")

    m.ts = pd.DataFrame(index=index)
    m.ts["rain"] = 20.0
    m.ts["evap"] = 5.0
    m.ts["flowup"] = 2.0
    m.ts["flowdown"] = np.nan
    m.ts["Irrigation"] = 1.0

    m.run_model()

    dam_results = m.indiv_results[m.indiv_results["dam_id"] == "Dam001"]

    # Every process associated with an operating dam must be disabled.
    np.testing.assert_allclose(dam_results["netevap"], 0.0, atol=1e-10)
    np.testing.assert_allclose(dam_results["netevap_actual"], 0.0, atol=1e-10)
    np.testing.assert_allclose(dam_results["demand"], 0.0, atol=1e-10)
    np.testing.assert_allclose(dam_results["demand_actual"], 0.0, atol=1e-10)
    np.testing.assert_allclose(dam_results["wf"], 0.0, atol=1e-10)
    np.testing.assert_allclose(dam_results["wf_actual"], 0.0, atol=1e-10)
    np.testing.assert_allclose(dam_results["bp"], 0.0, atol=1e-10)
    np.testing.assert_allclose(dam_results["bp_actual"], 0.0, atol=1e-10)

    # The dam remains empty and passes all incoming flow downstream.
    np.testing.assert_allclose(dam_results["sstore"], 0.0, atol=1e-10)
    np.testing.assert_allclose(dam_results["es"], 0.0, atol=1e-10)
    np.testing.assert_allclose(dam_results["spills"], 2.0, atol=1e-10)
    np.testing.assert_allclose(dam_results["outflow"], 2.0, atol=1e-10)
    
def test_decommissioning_release_reaches_downstream_dam():
    """Stored water released at decommissioning must follow the dam network."""

    m = tedimodel()
    m.timestep = "day"
    m.calc_nat_or_obs = "obs"
    m.cat_area_km2 = 10.0
    m.scale_limit = 1.0
    m.startstore_propfull = 1.0

    # Downstream dam remains operational.
    m.damlist.loc[0] = ["DamDown", -99, 5000.0, 1.0, 1.0, 50.0, 0, 9999, "Irrigation", 0.0, 0.0, 101, 1231, 0.0, 101, 1231, 1]

    # Upstream dam is decommissioned at the beginning of 2000.
    m.damlist.loc[1] = ["DamUp", "DamDown", 1000.0, 1.0, 1.0, 5.0, 0, 2000, "Irrigation", 0.0, 0.0, 101, 1231, 0.0, 101, 1231, 1]

    index = pd.date_range("1999-12-31", periods=3, freq="D")

    m.ts = pd.DataFrame(index=index)
    m.ts["rain"] = 0.0
    m.ts["evap"] = 0.0
    m.ts["flowup"] = 0.0
    m.ts["flowdown"] = np.nan
    m.ts["Irrigation"] = 0.0

    m.run_model()

    individual = m.indiv_results

    upstream = individual[individual["dam_id"] == "DamUp"].set_index("date")
    downstream = individual[individual["dam_id"] == "DamDown"].set_index("date")

    # The upstream dam releases its complete 5 ML storage on the first timestep of 2000.
    assert upstream.loc["2000-01-01", "sstore"] == pytest.approx(5.0)
    assert upstream.loc["2000-01-01", "es"] == pytest.approx(0.0)
    assert upstream.loc["2000-01-01", "spills"] == pytest.approx(5.0)
    assert upstream.loc["2000-01-01", "outflow"] == pytest.approx(5.0)

    # The complete release reaches the immediately downstream dam.
    assert downstream.loc["2000-01-01", "updam_spill"] == pytest.approx(5.0)

    # The downstream dam is already full, so the release continues to the outlet.
    assert downstream.loc["2000-01-01", "sstore"] == pytest.approx(50.0)
    assert downstream.loc["2000-01-01", "es"] == pytest.approx(50.0)
    assert downstream.loc["2000-01-01", "spills"] == pytest.approx(5.0)
    assert downstream.loc["2000-01-01", "outflow"] == pytest.approx(5.0)
    assert m.catch_results.loc["2000-01-01", "flow_down"] == pytest.approx(5.0)

    # The catchment water balance includes the loss of 5 ML from total storage.
    assert m.catch_results.loc["2000-01-01", "sstore"] == pytest.approx(55.0)
    assert m.catch_results.loc["2000-01-01", "estore"] == pytest.approx(50.0)
    assert m.catch_results.loc["2000-01-01", "sstore"] + m.catch_results.loc["2000-01-01", "flow_up"] == pytest.approx(m.catch_results.loc["2000-01-01", "estore"] + m.catch_results.loc["2000-01-01", "flow_down"])

def test_monthly_decommissioning_releases_storage_in_january_timestep():
    """A monthly model releases all stored water in the January timestep of year_end."""

    m = tedimodel()
    m.timestep = "month"
    m.calc_nat_or_obs = "obs"
    m.cat_area_km2 = 1.0
    m.scale_limit = 1.0
    m.startstore_propfull = 1.0

    m.damlist.loc[0] = ["Dam001", -99, 1000.0, 1.0, 1.0, 10.0, 0, 2000, "Irrigation", 0.0, 0.0, 101, 1231, 0.0, 101, 1231, 1]

    index = pd.date_range("1999-12-01", periods=3, freq="MS")

    m.ts = pd.DataFrame(index=index)
    m.ts["rain"] = 0.0
    m.ts["evap"] = 0.0
    m.ts["flowup"] = 2.0
    m.ts["flowdown"] = np.nan
    m.ts["Irrigation"] = 0.0

    m.run_model()

    results = m.catch_results

    # December: the active full dam spills only the monthly inflow.
    assert results.loc["1999-12-01", "sstore"] == pytest.approx(10.0)
    assert results.loc["1999-12-01", "estore"] == pytest.approx(10.0)
    assert results.loc["1999-12-01", "flow_down"] == pytest.approx(2.0)

    # January: the dam releases its 10 ML storage plus the 2 ML monthly inflow.
    assert results.loc["2000-01-01", "sstore"] == pytest.approx(10.0)
    assert results.loc["2000-01-01", "estore"] == pytest.approx(0.0)
    assert results.loc["2000-01-01", "spills"] == pytest.approx(12.0)
    assert results.loc["2000-01-01", "flow_down"] == pytest.approx(12.0)

    # February: the empty decommissioned dam passes the monthly inflow through.
    assert results.loc["2000-02-01", "sstore"] == pytest.approx(0.0)
    assert results.loc["2000-02-01", "estore"] == pytest.approx(0.0)
    assert results.loc["2000-02-01", "flow_down"] == pytest.approx(2.0)

    np.testing.assert_allclose(results["sstore"] + results["flow_up"], results["estore"] + results["flow_down"], atol=1e-10)
    
# ------------------------------------------------------------------------------------------------------------------------------------------------------

def _historical_scaling_model(vary_surfarea, demfactor):
    """Create one initially full dam with historical development fixed at 40%."""

    m = tedimodel()
    m.timestep = "day"
    m.calc_nat_or_obs = "obs"
    m.cat_area_km2 = 10.0
    m.scale_limit = 1.0
    m.startstore_propfull = 1.0
    m.vary_surfarea = vary_surfarea

    # V = a * surface_area ** b.
    # With a=0.01 and b=1, a 10 ML dam has a maximum area of 1000 m².
    m.vol_surfarea_a = 0.01
    m.vol_surfarea_b = 1.0

    m.damlist.loc[0] = [
        "Dam001",       # dam_id
        "-99",          # ds_damid: catchment outlet
        1000.0,         # surfacearea_m2
        1.0,            # catchuptotal_km2
        1.0,            # catchuplocal_km2
        10.0,           # vol_ML
        0,              # year_start
        9999,           # year_end
        "Irrigation",   # group_name
        demfactor,      # demfactor
        0.0,            # bypass_MLperT
        101,            # bypass_start_mmdd
        1231,           # bypass_end_mmdd
        0.0,            # wfill_MLperT
        101,            # wfill_start_mmdd
        1231,           # wfill_end_mmdd
        1,              # output
    ]

    dates = pd.date_range("2000-01-01", periods=2, freq="D")

    m.ts = pd.DataFrame(index=dates)
    m.ts["rain"] = 0.0
    m.ts["evap"] = 1000.0
    m.ts["flowup"] = 0.0
    m.ts["flowdown"] = np.nan
    m.ts["Irrigation"] = 1.0

    histdates = pd.DatetimeIndex(["1999-01-01", "2001-01-01"], name="Date")
    m.histoscale = pd.DataFrame({"histoscale": [0.4, 0.4]}, index=histdates)

    return m

def test_historical_scaling_is_applied_once_to_capacity_surface_area_and_demand():
    """At 40% development, capacity, fixed surface area and total demand should each be 40% of their full-development values."""

    m = _historical_scaling_model(vary_surfarea=False, demfactor=0.5)
    m.run_model()

    result = m.catch_results.iloc[0]

    # Scaled capacity and therefore initial storage:
    # 10 ML × 0.4 = 4 ML.
    assert result["sstore"] == pytest.approx(4.0)

    # Surface area:
    # 1000 m² × 0.4 = 400 m².
    #
    # Net evaporation:
    # 400 m² × 1000 mm / 1,000,000 = 0.4 ML.
    assert result["netevap"] == pytest.approx(0.4)
    assert result["netevap_actual"] == pytest.approx(0.4)

    # Demand:
    # scaled capacity × unscaled demand factor × demand pattern
    # 4 ML × 0.5 × 1.0 = 2 ML.
    #
    # The previous double-scaling calculation would have produced 0.8 ML.
    assert result["demand"] == pytest.approx(2.0)
    assert result["demand_actual"] == pytest.approx(2.0)

    # End storage:
    # 4.0 - 0.4 - 2.0 = 1.6 ML.
    assert result["estore"] == pytest.approx(1.6)

def test_historical_scaling_is_applied_once_to_variable_surface_area():
    """Variable surface area should represent the developed fraction of dams at their current proportional fullness."""

    m = _historical_scaling_model(vary_surfarea=True, demfactor=0.0)
    m.run_model()

    result = m.catch_results.iloc[0]

    # The effective dam capacity is 4 ML and the dam begins full.
    assert result["sstore"] == pytest.approx(4.0)

    # At full effective storage, the historically scaled surface area should be:
    # 1000 m² × 0.4 = 400 m².
    #
    # Net evaporation:
    # 400 m² × 1000 mm / 1,000,000 = 0.4 ML.
    #
    # The previous double-scaled formula would have produced only 0.16 ML.
    assert result["netevap"] == pytest.approx(0.4)
    assert result["netevap_actual"] == pytest.approx(0.4)

    # No demand or inflow occurs, so:
    # 4.0 - 0.4 = 3.6 ML.
    assert result["estore"] == pytest.approx(3.6)

def test_text_dam_ids_with_integer_outlet_marker_under_pyarrow_dtypes():
    """Text dam ids with the integer -99 outlet must validate when pandas stores the
    columns as pyarrow arrays (the default for some dtypes once pyarrow is installed).
    Comparing an int64[pyarrow] column with string values raised ArrowTypeError."""
    pytest.importorskip("pyarrow")
    from pytedi.engine import damlist_preprocessing

    m = tedimodel()
    m.cat_area_km2 = 10.0
    m.damlist.loc[0] = ["Dam001", -99, 1000.0, 1.0, 1.0, 10.0, 0, 9999, "Irrigation", 0.0, 0.0, 101, 1231, 0.0, 101, 1231, 1]
    m.damlist.loc[1] = ["Dam002", -99, 500.0, 0.5, 0.5, 5.0, 0, 9999, "Irrigation", 0.0, 0.0, 101, 1231, 0.0, 101, 1231, 1]
    damlist = m.damlist.copy()
    damlist["dam_id"] = damlist["dam_id"].astype("string[pyarrow]")
    damlist["ds_damid"] = pd.Series([-99, -99], dtype="int64[pyarrow]")  # integer outlet markers only
    demands = pd.DataFrame({"Irrigation": [0.0]}, index=pd.date_range("2000-01-01", periods=1))
    damlist_preprocessing(damlist, demands, m.cat_area_km2, m.scale_limit)  # must not raise
