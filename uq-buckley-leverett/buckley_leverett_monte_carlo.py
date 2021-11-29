import numpy as np
from buckley_leverett import buckley_solution

def eval_bl_mc(
    number_of_samples,
    porosity_distribution,
    wat_vicosity_distribution,
    oil_viscosity_distribution,
    total_simulation_time, 
    diameter, 
    length, 
    injection_rate, 
    aw, 
    bw, 
    ao, 
    bo, 
    sw_min, 
    sw_max, 
    output_times,
):
    np.random.seed(127)

    samples = np.array([
      porosity_distribution.rvs(number_of_samples), 
      wat_vicosity_distribution.rvs(number_of_samples), 
      oil_viscosity_distribution.rvs(number_of_samples),
    ])
    
    model = lambda poro_sample, wat_visc_sample, oil_visc_sample: buckley_solution(
      total_simulation_time, 
      poro_sample, 
      diameter, 
      length, 
      injection_rate, 
      wat_visc_sample, 
      oil_visc_sample, 
      aw, 
      bw, 
      ao, 
      bo, 
      sw_min, 
      sw_max, 
      output_times,
    )
    evals = np.array([ model(sample[0], sample[1] , sample[2]) for sample in samples.T ])
    
    xd = np.array(evals[0][0])
    mean_sw = []
    sw_upp = []
    sw_low = []
    
    pc = 95
    r_plus  = 0.5*(100 + pc)
    r_minus = 0.5*(100 - pc)

    
    for i in range(len(output_times)):
      sw_arrays = [ev[1][i] for ev in evals]
      mean_sw.append(np.mean(sw_arrays, 0))
      sw_upp.append(np.percentile(sw_arrays , r_plus , axis=0))
      sw_low.append(np.percentile(sw_arrays , r_minus , axis=0))
    
    time = np.array(evals[0][2])
    npd_arrays = np.array([ev[3] for ev in evals])
    mean_npd = np.mean(npd_arrays, 0)
    npd_upp = np.percentile(npd_arrays , r_plus , axis=0)
    npd_low = np.percentile(npd_arrays , r_minus , axis=0)

    # calculate the integral of the npd curve for all npd curves
    npd_squared_integrals = np.array([np.trapz(np.square(npd), time) for npd in npd_arrays])
    
    return xd, mean_sw, sw_upp, sw_low, time, mean_npd, npd_upp, npd_low, npd_squared_integrals