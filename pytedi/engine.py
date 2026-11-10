# -*- coding: utf-8 -*-
"""

pytedi - Python Tool for Estimating Dam Impacts
-----------------------------------------------

Dr Robert Morden
November 2026

This is the TEDI engine where the timestep calculations are done.

The functions below are only invoked if the user calls the run_model() method in the scenario class.

Calculations moving downstream are easy:
    - Loop through each day
      - Loop through each dam in sequence from upstream to downstream
        - Calculate a water balance for each dam
      - Aggregate for each day
    - Calculate the total catchment outflow at the end of the catchment
    
Calculations moving upstream are harder:
    - Loop through each day
      - Guess the total upstream flow for the day
        - Do the downstream calcs for that single day
        - Check if the calc outflow matches the obs outflow
      - Use a bisection algorithm to choose a revised upstream flow, and repeat
    - Calculate the total catchment outflow at the end of the catchment

@author: robert.morden
"""

import warnings
import pandas as pd
import numpy as np
from numba import jit


# %%=======================================================================================================================
'''
This function runs the model.
It starts with pre-processing the data and checking data integrity, then runs either calc_from_up_to_down or calc_from_down_to_up.
'''
def calc_dams(model):
    
    # get model parameters and inputs -----------------------------------------
    damlist_df = model.damlist                                                 
    cat_area = model.cat_area_km2
    scale_limit = model.scale_limit
    
    histoscale_df = model.histoscale.resample('D').interpolate('time')
    if model.timestep == 'month':
        histoscale_df = histoscale_df.resample('MS').first()
    
    
    demand_fields = ~model.ts.columns.isin(['rain', 'evap', 'flowup', 'flowdown'])
    demands_df = model.ts.loc[:,demand_fields]
    
    rain_df = model.ts.loc[:,['rain']]
    evap_df = model.ts.loc[:,['evap']]
    flowin_df = model.ts.loc[:,['flowup']]
    flowout_df = model.ts.loc[:,['flowdown']]
    
    if model.calc_nat_or_obs == 'obs':
        q_df = flowin_df
    elif model.calc_nat_or_obs == 'nat':
        q_df = flowout_df
    
    store_params = {}
    store_params['vary_surfarea'] = model.vary_surfarea
    store_params['vol_surfarea_a'] = model.vol_surfarea_a
    store_params['vol_surfarea_b'] = model.vol_surfarea_b
    store_params['startstore_propfull'] = model.startstore_propfull
    
    # check data integrity ----------------------------------------------------
    # connectivity, catch areas all add up, demand groups all exist in timeseries
    # haven't added this yet, but it could go here
    
    # Clip timeseries to minimum overlapping period
    sdate = max(rain_df.first_valid_index(),evap_df.first_valid_index(),q_df.first_valid_index(),demands_df.first_valid_index(),histoscale_df.first_valid_index())
    edate = min(rain_df.last_valid_index(),evap_df.last_valid_index(),q_df.last_valid_index(),demands_df.last_valid_index(),histoscale_df.last_valid_index())
    
    if edate > sdate:
        rain_df = rain_df.loc[sdate:edate]
        evap_df = evap_df.loc[sdate:edate]
        flowin_df = flowin_df.loc[sdate:edate]
        flowout_df = flowout_df.loc[sdate:edate]
        demands_df = demands_df.loc[sdate:edate,:]
        histoscale_df = histoscale_df.loc[sdate:edate]
    else:
        raise ValueError("Timeseries do not overlap. Check dates in rain, evap, flow, and demand timeseries.")
    
    if model.calc_nat_or_obs == "obs":
        active_flow_df = flowin_df
    else:
        active_flow_df = flowout_df
    
    check_timeseries_inputs(active_flow_df, rain_df, evap_df, demands_df, histoscale_df,)

    # run the model -----------------------------------------------------------
    if model.calc_nat_or_obs == 'obs':                                          
        global_results, alldam_results = calc_from_up_to_down(damlist_df, demands_df, rain_df, evap_df, flowin_df, cat_area, scale_limit, histoscale_df, store_params)
    
    elif model.calc_nat_or_obs == 'nat':
        global_results, alldam_results = calc_from_down_to_up(damlist_df, demands_df, rain_df, evap_df, flowout_df, cat_area, scale_limit, histoscale_df, store_params)

    return global_results, alldam_results

# %%=======================================================================================================================
'''
This function will calculate the flow DOWNstream of the dams given the total catchment flow UPstream.
'''
def calc_from_up_to_down(damlist_df, demands_df, rain_df, evap_df, flowin_df, cat_area, scale_limit, histoscale_df, store_params):

    damlist_df, int_to_ext = damlist_preprocessing(damlist_df, demands_df, cat_area, scale_limit)
    
    damlist_array = damlist_df.loc[:,['vol_ML','surfacearea_m2','catchuplocal_km2',
                                      'demfactor','year_start','year_end','demand_id',
                                      'bypass_MLperT','bypass_start_mmdd','bypass_end_mmdd',
                                      'wfill_MLperT','wfill_start_mmdd','wfill_end_mmdd',
                                      'dsdam','output']].to_numpy(dtype='float64')
    
    # convert timeseries to numpy arrays
    flow_array = flowin_df.to_numpy(dtype='float64').flatten()
    rain_array = rain_df.to_numpy(dtype='float64').flatten()
    evap_array = evap_df.to_numpy(dtype='float64').flatten()
    demd_array = demands_df.to_numpy(dtype='float64')
    
    # create array of year number for each timestep to identify when a dam should activate
    year_array = flowin_df.index.year.to_numpy(dtype='int64')
    mmdd_array = flowin_df.index.strftime('%m%d').astype('int32').to_numpy()
    history_array = histoscale_df.to_numpy(dtype='float64').flatten()
    
    # get start and end of flow series
    
    dtstart = flowin_df.first_valid_index()
    dtend = flowin_df.last_valid_index()
    istart = flowin_df.index.get_loc(dtstart)
    iend = flowin_df.index.get_loc(dtend)
    
    # run calculations
    global_array, alldam_array = tedi_engine_down(flow_array, rain_array, evap_array, demd_array, year_array, mmdd_array, history_array, damlist_array, cat_area, istart, iend,
                                                  store_params['vary_surfarea'], store_params['vol_surfarea_a'], store_params['vol_surfarea_b'], store_params['startstore_propfull'])

    global_results, alldam_results = _format_engine_results(global_array, alldam_array, flowin_df.index, int_to_ext)

    return global_results, alldam_results

# %%=======================================================================================================================
@jit (nopython=True, cache=True)
def tedi_engine_down(flow,rain,evap,demd,years,monthday,history,dams,catcharea,istart,iend,varyarea,vsa_a,vsa_b,startprop):

    """
    Calculate farm dam impacts on flow (numba @jit compiled).

    Returns:
        global_array  (nrows, 9)  - catchment fluxes per timestep
        alldam_array  (nrows, n_output, 11) - per-output-dam results per timestep
    where nrows = iend - istart + 1.
    """
    numdams = len(dams)

    n_output = 0                                                               # number of dams flagged for individual output
    for i in range(numdams):
        if dams[i,14] > 0:
            n_output += 1

    nrows = iend - istart + 1
    global_array = np.empty((nrows, 14))
    alldam_array = np.empty((nrows, n_output, 16))

    # set up ENDstore array, set ENDstore = capacity x start proportion    ( x historical scaling too!!)
    estore = dams[:,0] * startprop * history[istart]

    for dam_id in range(numdams):
        buildyr = int(dams[dam_id,4])
        endyr = int(dams[dam_id,5])
    
        if years[istart] < buildyr or years[istart] >= endyr:
            estore[dam_id] = 0.0
        
    for k in range(nrows):                                                     # for each timestep in the useful range
        istep = istart + k
        sstore = estore
        res = stedi_engine_down_singlestep(flow[istep], rain[istep], evap[istep], demd[istep,:],
                                           years[istep], monthday[istep], history[istep],
                                           dams, catcharea, istep, sstore, True, varyarea, vsa_a, vsa_b)

        estore = res[0]                     # estore
        
        global_array[k,0] = res[1]          # sstore_sum
        global_array[k,1] = res[2]          # flow_up_sum
        global_array[k,2] = res[3]          # netevap_sum
        global_array[k,3] = res[4]          # netevapact_sum
        global_array[k,4] = res[5]          # demd_sum
        global_array[k,5] = res[6]          # demdact_sum
        global_array[k,6] = res[7]          # wf_sum
        global_array[k,7] = res[8]          # wfact_sum
        global_array[k,8] = res[9]          # bp_sum
        global_array[k,9] = res[10]         # bpact_sum
        global_array[k,10] = res[11]        # estore_sum
        global_array[k,11] = res[12]        # spills_sum
        global_array[k,12] = res[13]        # outflow_sum
        global_array[k,13] = res[14]        # aggimpact_sum
        
        if n_output > 0:
            alldam_array[k,:,:] = res[15]   # dam_results

    return global_array, alldam_array

# %%=======================================================================================================================
'''
This function will calculate the flow DOWNstream of the dams given the total catchment flow UPstream.
'''
def calc_from_down_to_up(damlist_df, demands_df, rain_df, evap_df, flowout_df, cat_area, scale_limit, histoscale_df, store_params):

    damlist_df, int_to_ext = damlist_preprocessing(damlist_df, demands_df, cat_area, scale_limit)
    
    damlist_array = damlist_df.loc[:,['vol_ML','surfacearea_m2','catchuplocal_km2',
                                      'demfactor','year_start','year_end','demand_id',
                                      'bypass_MLperT','bypass_start_mmdd','bypass_end_mmdd',
                                      'wfill_MLperT','wfill_start_mmdd','wfill_end_mmdd',
                                      'dsdam','output']].to_numpy(dtype='float64')

    # convert timeseries to numpy arrays
    flow_array = flowout_df.to_numpy().astype('float64').flatten()
    rain_array = rain_df.to_numpy().astype('float64').flatten()
    evap_array = evap_df.to_numpy().astype('float64').flatten()
    demd_array = demands_df.to_numpy(dtype='float64')

    # create array of year number for each timestep to identify when a dam should activate
    year_array = flowout_df.index.year.to_numpy(dtype='int64')
    mmdd_array = flowout_df.index.strftime('%m%d').astype('int32').to_numpy()
    history_array = histoscale_df.to_numpy().flatten()
    
    # get start and end of flow series
    dtstart = flowout_df.first_valid_index()
    dtend = flowout_df.last_valid_index()
    istart = flowout_df.index.get_loc(dtstart)
    iend = flowout_df.index.get_loc(dtend)
    
    # run calculations
    global_array, alldam_array, solver_status = tedi_engine_up(flow_array, rain_array, evap_array, demd_array, year_array, mmdd_array, history_array, damlist_array, cat_area, istart, iend,
                                                store_params['vary_surfarea'], store_params['vol_surfarea_a'], store_params['vol_surfarea_b'], store_params['startstore_propfull'])
    
    failure_code = int(solver_status[0])
    
    if failure_code != 0:
        failure_step = int(solver_status[1])
        failure_date = flowout_df.index[failure_step]
        observed_flow = solver_status[2]
        downstream_residual = solver_status[3]
        lower_bound = solver_status[4]
        upper_bound = solver_status[5]
        iterations = int(solver_status[6])
    
        if failure_code == 2:
            reason = "an upper flow bound could not be found"
        elif failure_code == 3:
            reason = "the downstream residual did not converge within the iteration limit"
        else:
            reason = f"an unexpected solver failure code was returned ({failure_code})"
    
        raise RuntimeError(f"Reverse solver failed on {failure_date.date()}: {reason}. Observed downstream flow={observed_flow:.12g} ML, final downstream residual={downstream_residual:.12g} ML, upstream bracket=[{lower_bound:.12g}, {upper_bound:.12g}] ML, iterations={iterations}.")
    
    global_results, alldam_results = _format_engine_results(global_array, alldam_array, flowout_df.index, int_to_ext)
    
    observed_downstream = flowout_df.iloc[:, 0].astype('float64').reindex(global_results.index)
    solver_residual = global_results['flow_down'] - observed_downstream
    residual_tolerance = 0.02 + 1e-6 * observed_downstream.abs()
    extrapolated_mask = solver_residual > residual_tolerance
    
    if extrapolated_mask.any():
        extrapolated_count = int(extrapolated_mask.sum())
        largest_residual_date = solver_residual.loc[extrapolated_mask].idxmax()
        largest_residual = solver_residual.loc[largest_residual_date]
    
        warnings.warn(f"Reverse solver could not reproduce the observed downstream flow with a non-negative upstream flow on {extrapolated_count} timestep(s). Upstream flow was estimated using recent reconstructed flow, with two-timestep linear extrapolation where available. The largest downstream residual was {largest_residual:.6g} ML on {largest_residual_date.date()}.", RuntimeWarning, stacklevel=2)

    return global_results, alldam_results

# %%=======================================================================================================================
def _format_engine_results(global_array, alldam_array, index, internal_to_external):
    '''Turn the engine's numpy result arrays into the catch_results and indiv_results
    DataFrames. global_array is (nsteps, 9); alldam_array is (nsteps, n_output, 11).'''
    global_results_columns = ['sstore', 'flow_up', 'netevap', 'netevap_actual', 'demand', 'demand_actual', 'wf', 'wf_actual', 'bp', 'bp_actual', 'estore', 'spills', 'flow_down', 'agg_impact']
    global_results = pd.DataFrame(global_array, columns=global_results_columns, index=index)

    alldam_results_columns = ['istep','_dam_index','sstore','local_inflow','updam_spill','netevap','netevap_actual','demand','demand_actual','wf','wf_actual','bp','bp_actual','es','spills','outflow']
    alldam_flat = alldam_array.reshape(-1, len(alldam_results_columns))         # (nsteps * n_output, 11)
    alldam_results = pd.DataFrame(alldam_flat, columns=alldam_results_columns)
    alldam_results['istep'] = alldam_results['istep'].astype(int)
    
    internal_ids = alldam_results['_dam_index'].astype(int)
    dam_ids = internal_ids.map(internal_to_external)

    if dam_ids.isna().any():
        missing_internal_ids = internal_ids.loc[dam_ids.isna()].unique()

        raise RuntimeError(f"The engine returned dam indices that could not be mapped to dam identifiers: {missing_internal_ids.tolist()}")

    # Public identifier: contains the original integer or alphanumeric dam ID.
    alldam_results['dam_id'] = dam_ids

    alldam_results['date'] = (alldam_results['istep'].apply(lambda x: index[x]))

    alldam_results = alldam_results[[
                                    'istep',
                                    'dam_id',
                                    'date',
                                    'sstore',
                                    'local_inflow',
                                    'updam_spill',
                                    'netevap',
                                    'netevap_actual',
                                    'demand',
                                    'demand_actual',
                                    'wf',
                                    'wf_actual',
                                    'bp',
                                    'bp_actual',
                                    'es',
                                    'spills',
                                    'outflow',
                                    ]]


    return global_results, alldam_results
    # (ss, upflows, netevap, netevap_act, demand, demand_act, wf_rate, wfrate_act, bprate, bprate_act, es, spills, outflow, agg_impact),
    # (istep, idam, ss, localflow, updamspills, netevap, netevap_act, demand, demand_act, wf_rate, wfrate_act, bprate, bprate_act, es, spills, outflow)
# %%=======================================================================================================================
'''
This function will calculate the flow UPstream of the dams given the total catchment flow DOWNstream.
This is much harder than calculating the other way, because the flow has to be estimated iteratively.
It includes a bisection solver to estimate the upstream flow.
The original Windows STEDI used a bisection solver, but I'm not sure how it was implemented. This is a new implementation.
'''
@jit (nopython=True, cache=True)
def tedi_engine_up(flow,rain,evap,demd,years,monthday,history,dams,catcharea,istart,iend,varyarea,vsa_a,vsa_b,startprop):
    """
    Calculate flow UPstream of the dams given the catchment flow DOWNstream, using a
    per-timestep bisection solver (numba @jit compiled).

    Returns global_array (nrows, 9) and alldam_array (nrows, n_output, 11), as for
    tedi_engine_down.
    
    Also returns a solver_status array. solver_status[0] is zero on success and otherwise
    contains a failure code for the first timestep that could not be solved.
    """
    numdams = len(dams)

    n_output = 0

    for i in range(numdams):
        if dams[i, 14] > 0:
            n_output += 1

    nrows = iend - istart + 1

    global_array = np.empty((nrows, 14))
    alldam_array = np.empty((nrows, n_output, 16))

    # Elements are:
    #   failure code, timestep, observed flow, residual,
    #   lower bound, upper bound, iterations
    #
    # Failure code 2 = no upper bracket was found.
    # Failure code 3 = bisection did not converge.
    # A physically infeasible lower boundary is handled by temporal extrapolation and is reported later as a warning.
    solver_status = np.zeros(7)

    estore = dams[:, 0] * startprop * history[istart]

    for dam_id in range(numdams):
        buildyr = int(dams[dam_id,4])
        endyr = int(dams[dam_id,5])
    
        if years[istart] < buildyr or years[istart] >= endyr:
            estore[dam_id] = 0.0

    # Mixed absolute and relative convergence tolerances.
    residual_atol = 0.02
    residual_rtol = 1e-6
    upstream_atol = 0.02
    upstream_rtol = 1e-6

    bracket_growth = 2.0
    max_bracket_iterations = 50
    max_bisection_iterations = 100

    # Location of downstream flow in the singlestep return tuple.
    outflowcol = 13

    for k in range(nrows):

        istep = istart + k

        q_obs = flow[istep]
        r = rain[istep]
        e = evap[istep]
        d = demd[istep, :]
        mmdd = monthday[istep]
        yy = years[istep]
        h = history[istep]

        sstore = estore

        residual_tolerance = residual_atol + residual_rtol * abs(q_obs)

        # Zero is the physical lower bound for upstream flow.
        lower_bound = 0.0

        res = stedi_engine_down_singlestep(lower_bound, r, e, d, yy, mmdd, h, dams, catcharea, istep, sstore, False, varyarea, vsa_a, vsa_b)

        lower_residual = res[outflowcol] - q_obs

        flow_us = 0.0
        converged = False
        bracket_iterations = 0
        bisection_iterations = 0
        final_residual = lower_residual
        upper_bound = 0.0

        # If zero upstream flow reproduces the observation within tolerance,use zero. This is also the minimum physically possible solution.
        if abs(lower_residual) <= residual_tolerance:
            converged = True

        # If zero upstream flow already produces too much downstream flow, no exact physically valid solution exists. Estimate upstream flow from the recent reconstructed-flow trend and continue.
        elif lower_residual > 0.0:
            if k >= 2:
                flow_us = max(0.0, 2.0 * global_array[k - 1, 1] - global_array[k - 2, 1])
            elif k == 1:
                flow_us = max(0.0, global_array[k - 1, 1])
            else:
                flow_us = max(0.0, q_obs)
        
            converged = True

        else:
            # Begin the upper-bound search at the observed downstream flow.
            # residual_atol provides a non-zero starting value for very small flows.
            upper_bound = max(q_obs, residual_atol)

            res = stedi_engine_down_singlestep(upper_bound, r, e, d, yy, mmdd, h, dams, catcharea, istep, sstore, False, varyarea, vsa_a, vsa_b)

            upper_residual = res[outflowcol] - q_obs

            # Increase the upper bound until the residual changes sign.
            while upper_residual < 0.0 and bracket_iterations < max_bracket_iterations:
                bracket_iterations += 1
                upper_bound *= bracket_growth

                res = stedi_engine_down_singlestep(upper_bound, r, e, d, yy, mmdd, h, dams, catcharea, istep, sstore, False, varyarea, vsa_a, vsa_b)

                upper_residual = res[outflowcol] - q_obs

            if upper_residual < 0.0 or not np.isfinite(upper_bound):
                solver_status[0] = 2.0
                solver_status[1] = istep
                solver_status[2] = q_obs
                solver_status[3] = upper_residual
                solver_status[4] = lower_bound
                solver_status[5] = upper_bound
                solver_status[6] = bracket_iterations

                return global_array, alldam_array, solver_status

            # Apply bisection to the valid bracket.
            while bisection_iterations < max_bisection_iterations:
                bisection_iterations += 1

                midpoint = (lower_bound + upper_bound) / 2.0

                res = stedi_engine_down_singlestep(midpoint, r, e, d, yy, mmdd, h, dams, catcharea, istep, sstore, False, varyarea, vsa_a, vsa_b)

                midpoint_residual = res[outflowcol] - q_obs
                half_bracket = (upper_bound - lower_bound) / 2.0
                upstream_tolerance = upstream_atol + upstream_rtol * abs(midpoint)

                # Require both a sufficiently small downstream residual and
                # a sufficiently narrow upstream-flow bracket.
                if abs(midpoint_residual) <= residual_tolerance and half_bracket <= upstream_tolerance:
                    flow_us = midpoint
                    final_residual = midpoint_residual
                    converged = True
                    break

                if midpoint_residual < 0.0:
                    lower_bound = midpoint
                    lower_residual = midpoint_residual
                else:
                    upper_bound = midpoint
                    upper_residual = midpoint_residual

                final_residual = midpoint_residual

            if not converged:
                solver_status[0] = 3.0
                solver_status[1] = istep
                solver_status[2] = q_obs
                solver_status[3] = final_residual
                solver_status[4] = lower_bound
                solver_status[5] = upper_bound
                solver_status[6] = bracket_iterations + bisection_iterations

                return global_array, alldam_array, solver_status

        # Run the accepted solution again with individual-dam output enabled.
        res = stedi_engine_down_singlestep(flow_us, r, e, d, yy, mmdd, h, dams, catcharea, istep, sstore, True, varyarea, vsa_a, vsa_b)

        estore = res[0]

        global_array[k, 0] = res[1]
        global_array[k, 1] = res[2]
        global_array[k, 2] = res[3]
        global_array[k, 3] = res[4]
        global_array[k, 4] = res[5]
        global_array[k, 5] = res[6]
        global_array[k, 6] = res[7]
        global_array[k, 7] = res[8]
        global_array[k, 8] = res[9]
        global_array[k, 9] = res[10]
        global_array[k, 10] = res[11]
        global_array[k, 11] = res[12]
        global_array[k, 12] = res[13]
        global_array[k, 13] = res[14]

        if n_output > 0:
            alldam_array[k, :, :] = res[15]

    return global_array, alldam_array, solver_status
        
# %%=======================================================================================================================
'''
This function will calculate the flow from up to down for a single timestep.
It needs to be run recursively to find a suitable upstream flow, or can just be run once to find the downstream flow.
'''
@jit (nopython=True, cache=True)
def stedi_engine_down_singlestep(flow,rain,evap,demd,yy,mmdd,history,dams,catcharea,istep,sstore,output_tf,varyarea,a,b):
    
    """
    Calculate farm dam impacts on flow for a SINGLE TIMESTEP
    
    Takes the following inputs:
        flow / rain / evap         (scalar, one row per timestep, all units ML or mm)
        demd                       (1d numpy array, cols = demand for each group)
        yy,mmdd                    (integers, year and month/day)
        history                    (scalar, impact scaling factor from 0 to 1 for all dams)
        dams                       (2d numpy array, 1 row per dam, cols = capacity(ML)/surfarea(m2)/catcharea(km2)/demfactor/buildyear/demandid/bypassrate/bypassstart/bypassend/dsdam/output)
              NOTE that dams includes a dummy dam for the catchment outlet (no capacity, demands, or bypasses)
              The total catchment outflow is the outflow of the dummy dam
        catcharea                  (scalar, size of catchment in km2)
        istep                      (integer, index of timestep in flow/rain/evap/demd arrays)
        sstore                     (1d numpy array, 1 col per dam, cols = start storage for each dam)
    
    Note that all inputs are numpy-based so that NUMBA can make it work super fast
    """
    # set up variables needed for THIS timestep
    numdams = len(dams)
    dsflows = np.zeros(numdams)
    if output_tf:                                                              # only allocate the (per output dam) results when reporting
        dam_results = np.empty((numdams, 16))
    else:
        dam_results = np.empty((0, 16))
    nout = 0

    # set up variables needed for the NEXT timestep
    estore = sstore * 0.0

    # set up AGGREGATE variables for the timestep
    sstore_sum = 0.0
    flow_up_sum = 0.0
    netevap_sum = 0.0
    demd_sum = 0.0
    wf_sum = 0.0
    bp_sum = 0.0
    estore_sum = 0.0
    outflow_sum = 0.0
    aggimpact_sum = 0.0
    netevapact_sum = 0.0
    demdact_sum = 0.0
    bpact_sum = 0.0
    wfact_sum = 0.0
    spills_sum = 0.0

    for idam in range(0,numdams):                                              # for each dam in the catchment

        ss = sstore[idam]

        capc = dams[idam,0] * history
        
        if varyarea:
            if history > 0.0:
                surf = ((ss/(history*a))**(1/b)) * history                     # if histoscale > 0 then surfarea varies, but non-linearly with volume.
            else:
                surf = 0.0                                                     # if histoscale = 0 then surfarea = 0
        else:
            surf = dams[idam,1] * history                                      # if constant surface area, vary with histoscale
        
        ctch =        dams[idam,2]
        dfac =        dams[idam,3]                                             # DO NOT vary demand factor with history, this already happens via the capacity. Doun't double dip.
        buildyr = int(dams[idam,4])
        endyr =   int(dams[idam,5])
        demdid =  int(dams[idam,6])
        bprate =      dams[idam,7] * history
        bpstart = int(dams[idam,8])
        bpend =   int(dams[idam,9])
        wfrate =      dams[idam,10] * history
        wfstart = int(dams[idam,11])
        wfend =   int(dams[idam,12])
        dsdam =   int(dams[idam,13])

        damoutput = False
        if output_tf:
            if dams[idam,14] > 0:
                damoutput = True

        dem = demd[demdid]

        if bpstart<=bpend:                                                     # eg. bypass starts in March and ends in August
            if (mmdd<bpstart) or (mmdd>bpend):                                    # if today is outside the season
                bprate = 0.0                                                      # bypass rate is zero
        else:                                                                  # eg. bypass starts in Nov and ends in May
            if (mmdd<bpstart) and (mmdd>bpend):                                   # if today is outside the season
                bprate = 0.0                                                      # bypass rate is zero

        if wfstart<=wfend:                                                     # eg. wfill starts in March and ends in August
            if (mmdd<wfstart) or (mmdd>wfend):                                    # if today is outside the season
                wfrate = 0.0                                                      # wfill rate is zero
        else:                                                                  # eg. wfill starts in Nov and ends in May
            if (mmdd<wfstart) and (mmdd>wfend):                                   # if today is outside the season
                wfrate = 0.0                                                      # wfill rate is zero

        if yy < buildyr:                                                       # if the build year has not happened yet
            capc = 0.0                                                            # set to zero -> capacity, surface area, demand factor, bypass, winterfill, start storage
            surf = 0.0
            ctch = dams[idam,2]
            dfac = 0.0
            bprate = 0.0
            wfrate = 0.0
            ss = 0.0
        
        elif yy >= endyr:                                                      # if the decommissioning year has passed
            capc = 0.0                                                            # set to zero -> capacity, surface area, demand factor, bypass, winterfill
            surf = 0.0                                                            # DO NOT SET THE START STORAGE TO ZERO - this will be spilled in the current timestep to preserve volume
            ctch = dams[idam,2]
            dfac = 0.0
            bprate = 0.0
            wfrate = 0.0

        # calc inflow, rain, evap, demand, outflows, impact
        localinflow = ctch / catcharea * flow                                  # inflow = damarea / catcharea x flow
        updamspill = dsflows[idam]                                             # outflows form upstream dams
        bprate_act = max(0.0, min(bprate, localinflow + updamspill))           # ACTUAL bypass rate
        netinflow = localinflow + updamspill - bprate_act                      # netinflow (make sure bypass flow is taken before it reaches the dam)
        netevap = surf * (evap-rain) / 1e6                                     # netevap = surface area x netevap / 1e6
        demand = capc * dfac * dem                                             # demand = capacity x demfactor x pattern
        
        spills = 0.0                                           # initialise spills
        storstep = ss                                          # step storage = starting storage for timestep
        wfrate_act = 0.0                                       # winterfill is initially SWITCHED OFF
        netevap_act = netevap
        demand_act = demand
        
        projected_storage = ss + netinflow - netevap - demand  # projected storage without winterfill
        
        if projected_storage < capc:                           # if no spills
            available_airspace = capc - max(projected_storage,0.0)
            wfrate_act = min(wfrate, available_airspace)           # wfill SWITCHED ON
        
        storstep += netinflow                                  # priority 1 - inflow (always add this, work out spills later)
        storstep += wfrate_act                                 # priority 2 - winterfill
        
        storstep -= netevap                                    # priority 3 - net evaporation
        if storstep < 0.0:                                        # check if net evaporation exceeds available water
            netevap_act = netevap - (0-storstep)
            storstep = 0.0

        storstep -= demand                                     # priority 4 - on-farm demands
        if storstep < 0.0:                                        # check if demand exceeds available water
            demand_act = demand - (0-storstep)
            storstep = 0.0

        if storstep > capc:                                    # check for spills
            spills = storstep - capc
            storstep = capc

        es = storstep                                          # end storage = step storage
        
        outflow = spills + bprate_act
        if dsdam>=0:                                                           # if the next downstream id is -99, then no need to aggregate
            dsflows[dsdam] = dsflows[dsdam] + outflow                          # spills directed into another dam
            
        # aggregate variables across dams as required
        sstore_sum += ss
        flow_up_sum += localinflow
        netevap_sum += netevap
        demd_sum += demand
        bp_sum += bprate
        wf_sum += wfrate
        spills_sum += spills
        estore_sum += es
        
        # aggregate ACTUAL values where the storage status can affect them
        netevapact_sum += netevap_act
        demdact_sum += demand_act
        bpact_sum += bprate_act
        wfact_sum += wfrate_act
        
        if idam == numdams-1:
            spills_sum -= spills                                               # need to remove the spills from the outlet (they are not really spills, just a calculation convenience)
            outflow_sum = outflow
            aggimpact_sum = flow_up_sum - outflow_sum                          # impact = Q-NoDams - Q-WithDams; bypass is already part of outflow_sum, do not subtract again
    
        # prepare results for a single dam only if the dam is marked for output
        if damoutput:
            dam_results[nout,0] = istep
            dam_results[nout,1] = idam
            dam_results[nout,2] = ss
            dam_results[nout,3] = localinflow
            dam_results[nout,4] = updamspill
            dam_results[nout,5] = netevap
            dam_results[nout,6] = netevap_act
            dam_results[nout,7] = demand
            dam_results[nout,8] = demand_act
            dam_results[nout,9] = wfrate
            dam_results[nout,10] = wfrate_act
            dam_results[nout,11] = bprate
            dam_results[nout,12] = bprate_act
            dam_results[nout,13] = es
            dam_results[nout,14] = spills
            dam_results[nout,15] = outflow
            nout += 1

        # prepare variables needed for next timestep
        estore[idam] = es                                                      # end storage

    # returns: estore array, then the 14 global fluxes in catch_results column order
    # (ss, flow_up, netevap, netevap_act, demand, demand_act, wf_rate, wfrate_act, bprate, bprate_act, es, spills, outflow, agg_impact),
    # then the (n_output, 11) individual-dam results for this timestep
    return (estore, sstore_sum, flow_up_sum, netevap_sum, netevapact_sum, demd_sum, demdact_sum, wf_sum, wfact_sum, bp_sum, bpact_sum, estore_sum, spills_sum, outflow_sum, aggimpact_sum, dam_results[:nout,:])

# %%=======================================================================================================================

def get_us_to_ds_order(df):
    
    reordered_id_list = []
    
    branch_ends = df[~df['dam_id'].isin(df['ds_damid'])]
    branch_junctions = df[df['dam_id'].isin(df['ds_damid'])]
    
    while len(branch_ends)>0:
        reordered_id_list.extend(branch_ends['dam_id'].to_list())
        df = branch_junctions
        branch_ends = df[~df['dam_id'].isin(df['ds_damid'])]
        branch_junctions = df[df['dam_id'].isin(df['ds_damid'])]
        
    return reordered_id_list

# %%=======================================================================================================================

def damlist_preprocessing(damlist_df, demands_df, cat_area, scale_limit):

    # Work on a copy. The user's damlist should retain its original identifiers.
    damlist_df = damlist_df.copy(deep=True)

    # -------------------------------------------------------------------------
    # Validate year start and end columns
    # -------------------------------------------------------------------------
    if 'year_end' not in damlist_df.columns:
        damlist_df['year_end'] = 9999
    else:
        damlist_df['year_end'] = damlist_df['year_end'].fillna(9999)
    
    damlist_df['year_start'] = pd.to_numeric(damlist_df['year_start'], errors='coerce')
    damlist_df['year_end'] = pd.to_numeric(damlist_df['year_end'], errors='coerce')
    
    invalid_year_values = damlist_df['year_start'].isna() | damlist_df['year_end'].isna()
    
    if invalid_year_values.any():
        invalid_dams = damlist_df.loc[invalid_year_values, 'dam_id'].tolist()
        raise ValueError(f"The following dams have missing or non-numeric year_start or year_end values: {invalid_dams}")
    
    non_integer_years = (damlist_df['year_start'] % 1 != 0) | (damlist_df['year_end'] % 1 != 0)
    
    if non_integer_years.any():
        invalid_dams = damlist_df.loc[non_integer_years, 'dam_id'].tolist()
        raise ValueError(f"The following dams have non-integer year_start or year_end values: {invalid_dams}")
    
    invalid_year_order = damlist_df['year_end'] <= damlist_df['year_start']
    
    if invalid_year_order.any():
        invalid_dams = damlist_df.loc[invalid_year_order, ['dam_id', 'year_start', 'year_end']].to_dict('records')
        raise ValueError(f"Each year_end must be later than year_start. Invalid dams: {invalid_dams}")
    
    damlist_df['year_start'] = damlist_df['year_start'].astype('int32')
    damlist_df['year_end'] = damlist_df['year_end'].astype('int32')
    
    # -------------------------------------------------------------------------
    # Validate external dam identifiers
    # -------------------------------------------------------------------------
    if damlist_df['dam_id'].isna().any():
        raise ValueError("dam_id cannot contain missing values.")

    duplicated = damlist_df.loc[damlist_df['dam_id'].duplicated(keep=False), 'dam_id' ].unique()

    if len(duplicated) > 0:
        raise ValueError(f"dam_id values must be unique. Duplicate IDs: {duplicated.tolist()}")

    # For the DS dam, accept -99, "-99", or a missing value as the catchment outlet.
    ds_text = damlist_df['ds_damid'].astype('string').str.strip()

    outlet_link = (damlist_df['ds_damid'].isna() | damlist_df['ds_damid'].eq(-99).fillna(False) | ds_text.eq('-99').fillna(False) )

    # -99 is reserved for the outlet and cannot itself be a dam identifier.
    dam_id_text = damlist_df['dam_id'].astype('string').str.strip()

    reserved_id = (damlist_df['dam_id'].eq(-99).fillna(False)  |  dam_id_text.eq('-99').fillna(False))

    if reserved_id.any():
        raise ValueError("-99 is reserved for the catchment outlet and cannot be used as a dam_id.")

    # Every dam must connect to a valid dam or the outlet.
    known_dam_ids = damlist_df['dam_id']

    # Compare as Python objects: with pyarrow-backed columns, isin between an integer column
    # and string ids raises instead of returning False. Object comparison keeps the same
    # meaning as before (the integer 2 still does not match the text "2").
    unknown_downstream = (~outlet_link & ~damlist_df['ds_damid'].astype(object).isin(known_dam_ids.astype(object)))

    if unknown_downstream.any():
        unknown_ids = damlist_df.loc[unknown_downstream, 'ds_damid'].unique()
        raise ValueError(f"The following downstream dam id values do not match an existing dam id: {unknown_ids.tolist()}")

    # -------------------------------------------------------------------------
    # Validate dam capacities and areas
    # -------------------------------------------------------------------------
    # Check required physical dam properties.
    required_physical_fields = ["vol_ML","surfacearea_m2","catchuplocal_km2","catchuptotal_km2",]
    
    for column in required_physical_fields:
        numeric_values = pd.to_numeric(damlist_df[column],errors="coerce",)
    
        invalid_value = (numeric_values.isna() | ~np.isfinite(numeric_values))
    
        if invalid_value.any():
            affected_dams = damlist_df.loc[invalid_value,"dam_id",].tolist()
    
            raise ValueError(f"{column} must contain finite numeric values. Check the following dams: {affected_dams}")
    
        # Retain the validated numeric representation.
        damlist_df[column] = numeric_values
    
    # Check for negative dam capacities
    negative_capacity = damlist_df["vol_ML"] < 0.0
    
    if negative_capacity.any():
        affected_dams = damlist_df.loc[negative_capacity, "dam_id", ].tolist()
    
        raise ValueError(f"Dam capacity (vol_ML) cannot be negative. Check the following dams: {affected_dams}")
    
    # Check for negative dam surface areas.
    negative_surface_area = damlist_df["surfacearea_m2"] < 0.0
    
    if negative_surface_area.any():
        affected_dams = damlist_df.loc[negative_surface_area, "dam_id", ].tolist()
        
        raise ValueError(f"Dam surface area (surfacearea_m2) cannot be negative. Check the following dams: {affected_dams}")
    
    # Check for negative dam catchment areas.
    catchment_area_fields = {"catchuplocal_km2": "Local dam catchment area",
                             "catchuptotal_km2": "Total upstream dam catchment area",}
    
    for column, description in catchment_area_fields.items():
        negative_area = damlist_df[column] < 0.0
    
        if negative_area.any():
            affected_dams = damlist_df.loc[negative_area,"dam_id",].tolist()
    
            raise ValueError(f"{description} ({column}) cannot be negative. Check the following dams: {affected_dams}")
            
    # Validate bypass and winterfill seasonal dates.
    seasonal_date_fields = [
        "bypass_start_mmdd",
        "bypass_end_mmdd",
        "wfill_start_mmdd",
        "wfill_end_mmdd",
    ]
    
    for column in seasonal_date_fields:
        valid_date = damlist_df[column].apply(_is_valid_mmdd)
        invalid_date = ~valid_date
    
        if invalid_date.any():
            affected = damlist_df.loc[invalid_date, ["dam_id", column], ]
    
            details = [f"{row['dam_id']}: {row[column]}" for _, row in affected.iterrows()]
    
            raise ValueError(f"{column} must contain valid MMDD calendar dates. Invalid dam/value pairs: {details}")
            
    # -------------------------------------------------------------------------
    # Scale local catchment areas
    # -------------------------------------------------------------------------
    # Check the total model catchment area.
    if not np.isfinite(cat_area) or cat_area <= 0.0:
        raise ValueError("The total catchment area (cat_area_km2) must be a finite value greater than zero.")
    
    # Check the catchment-area scaling limit.
    if (not np.isfinite(scale_limit) or scale_limit < 0.0 or scale_limit > 1.0):
        raise ValueError("scale_limit must be a finite value between 0 and 1 inclusive.")
    
    impounded_km2 = damlist_df['catchuplocal_km2'].sum()
    pc_impound = impounded_km2 / cat_area

    if pc_impound > scale_limit:
        damlist_df['catchuplocal_km2'] *= scale_limit / pc_impound

    # Calculate this after scaling. This also fixes the previously identified stale outlet-area calculation.
    unimp_km2 = cat_area - damlist_df['catchuplocal_km2'].sum()

    # -------------------------------------------------------------------------
    # Convert demand groups to numeric engine indices
    # -------------------------------------------------------------------------
    demgroup_mapping = {column_name: index for index, column_name in enumerate(demands_df.columns)}

    damlist_df['demand_id'] = (damlist_df['group_name'].map(demgroup_mapping))

    unknown_groups = (damlist_df['demand_id'].isna() & damlist_df['group_name'].ne('none'))

    if unknown_groups.any():
        groups = damlist_df.loc[unknown_groups, 'group_name'].unique()

        raise ValueError(f"The following dam demand groups do not have a corresponding timeseries column: {groups.tolist()}")

    # -------------------------------------------------------------------------
    # Sort dams from upstream to downstream using the external identifiers
    # -------------------------------------------------------------------------
    new_dam_order = get_us_to_ds_order(damlist_df[['dam_id', 'ds_damid']])

    # If not every dam was returned, the network probably contains a cycle.
    if len(new_dam_order) != len(damlist_df):
        unordered_ids = damlist_df.loc[~damlist_df['dam_id'].isin(new_dam_order),'dam_id'].tolist()

        raise ValueError(f"The dam network could not be ordered from upstream to downstream. Check for a network cycle involving these dam ids: {unordered_ids}" )

    order_lookup = {dam_id: position for position, dam_id in enumerate(new_dam_order)}

    damlist_df['_engine_order'] = (damlist_df['dam_id'].map(order_lookup))

    damlist_df = (
        damlist_df
        .sort_values('_engine_order')
        .drop(columns='_engine_order')
        .reset_index(drop=True)
    )

    # Recalculate the outlet mask because the dataframe has now been reordered.
    ds_text = damlist_df['ds_damid'].astype('string').str.strip()

    outlet_link = (
        damlist_df['ds_damid'].isna()
        | damlist_df['ds_damid'].eq(-99).fillna(False)
        | ds_text.eq('-99').fillna(False)
    )

    # -------------------------------------------------------------------------
    # Convert external IDs to integer engine indices
    # -------------------------------------------------------------------------
    # Examples:
    #   DamPoly001  -> 0
    #   DamPoint004 -> 1
    #
    # These indices correspond exactly to row positions after network sorting.
    external_to_internal = {
        dam_id: internal_id
        for internal_id, dam_id
        in enumerate(damlist_df['dam_id'].tolist())
    }

    # This mapping is used later to restore the original identifiers in output.
    internal_to_external = {
        internal_id: dam_id
        for dam_id, internal_id
        in external_to_internal.items()
    }

    # Convert downstream references before overwriting dam_id.
    damlist_df['dsdam'] = (damlist_df['ds_damid'].map(external_to_internal))

    # The synthetic outlet will be the row immediately after the real dams.
    outlet_index = len(damlist_df)

    damlist_df.loc[outlet_link, 'dsdam'] = outlet_index

    # Both of these columns are now private integer engine identifiers.
    damlist_df['dam_id'] = (damlist_df['dam_id'].map(external_to_internal).astype('int32'))

    damlist_df['ds_damid'] = damlist_df['dsdam'].astype('int32')
    damlist_df.loc[outlet_link, 'ds_damid'] = -99
    damlist_df['dsdam'] = damlist_df['dsdam'].astype('int32')

    # -------------------------------------------------------------------------
    # Add the synthetic catchment outlet
    # -------------------------------------------------------------------------
    outlet_row = {
        'dam_id': outlet_index,
        'ds_damid': -99,
        'surfacearea_m2': 0.0,
        'catchuptotal_km2': cat_area,
        'catchuplocal_km2': unimp_km2,
        'vol_ML': 0.0,
        'year_start': 0,
        'year_end' : 9999,
        'group_name': 'none',
        'demfactor': 0.0,
        'bypass_MLperT': 0.0,
        'bypass_start_mmdd': 101,
        'bypass_end_mmdd': 1231,
        'wfill_MLperT': 0.0,
        'wfill_start_mmdd': 101,
        'wfill_end_mmdd': 1231,
        'output': 0,
        'demand_id': 0,
        'dsdam': -99,
    }

    for column, value in outlet_row.items():
        damlist_df.loc[outlet_index, column] = value

    return damlist_df, internal_to_external

#%%
def check_timeseries_inputs(flow_df, rain_df, evap_df, demands_df, histoscale_df):
    """
    Validate aligned model timeseries.

    Negative evaporation is allowed. All other inputs must be
    finite and non-negative.
    """

    inputs = {"flow": flow_df,
              "rainfall": rain_df,
              "evaporation": evap_df,
              "demand": demands_df,
              "historical scaling": histoscale_df,}

    # ------------------------------------------------------------------
    # Check index structure
    # ------------------------------------------------------------------
    for name, dataframe in inputs.items():

        if not isinstance(dataframe.index, pd.DatetimeIndex):
            raise ValueError(f"The {name} timeseries must use a pandas DatetimeIndex.")

        if dataframe.index.has_duplicates:
            duplicated_dates = (dataframe.index[dataframe.index.duplicated(keep=False)].unique().tolist())

            raise ValueError(f"The {name} timeseries contains duplicate dates: {duplicated_dates}")

        if not dataframe.index.is_monotonic_increasing:
            raise ValueError(f"The {name} timeseries dates must be in ascending order.")

    # All inputs must have exactly the same dates after overlap clipping.
    reference_index = flow_df.index

    for name, dataframe in inputs.items():
        if not dataframe.index.equals(reference_index):
            raise ValueError(f"The {name} timeseries dates do not match the active flow timeseries dates.")

    # ------------------------------------------------------------------
    # Check that values are numeric, present and finite
    # ------------------------------------------------------------------
    numeric_inputs = {}

    for name, dataframe in inputs.items():

        numeric = dataframe.apply(pd.to_numeric, errors="coerce", )

        invalid = (numeric.isna() | ~np.isfinite(numeric))

        if invalid.any().any():
            rows, columns = np.where(invalid.to_numpy())

            first_row = rows[0]
            first_column = columns[0]

            bad_date = dataframe.index[first_row]
            bad_column = dataframe.columns[first_column]

            raise ValueError(f"The {name} timeseries contains a missing, non-numeric or infinite value in column '{bad_column}' on {bad_date.date()}.")

        numeric_inputs[name] = numeric

    # ------------------------------------------------------------------
    # Check values that cannot be negative
    # ------------------------------------------------------------------
    # Deliberately do not perform a negative-value check on evaporation.
    nonnegative_inputs = {
        "flow":               numeric_inputs["flow"],
        "rainfall":           numeric_inputs["rainfall"],
        "demand":             numeric_inputs["demand"],
        "historical scaling": numeric_inputs["historical scaling"],
    }

    for name, dataframe in nonnegative_inputs.items():

        negative = dataframe < 0.0

        if negative.any().any():
            rows, columns = np.where(negative.to_numpy())

            first_row = rows[0]
            first_column = columns[0]

            bad_date = dataframe.index[first_row]
            bad_column = dataframe.columns[first_column]
            bad_value = dataframe.iloc[first_row, first_column]

            raise ValueError(f"The {name} timeseries contains a negative value ({bad_value}) in column '{bad_column}' on {bad_date.date()}.")

#%%
def _is_valid_mmdd(value):
    """Return True when value represents a valid recurring MMDD date."""

    if pd.isna(value):
        return False

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False

    if not np.isfinite(numeric_value):
        return False

    # Values such as 101.5 are not valid MMDD integers.
    if not numeric_value.is_integer():
        return False

    mmdd = int(numeric_value)
    month = mmdd // 100
    day = mmdd % 100

    try:
        # Use a leap year so that 0229 is accepted.
        pd.Timestamp(year=2000, month=month, day=day, )
    
    except (TypeError, ValueError):
        return False

    return True
# %%=======================================================================================================================
