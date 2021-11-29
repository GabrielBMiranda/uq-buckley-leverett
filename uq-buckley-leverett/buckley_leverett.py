import math
import numpy as np
from scipy.optimize import fsolve
from scipy.misc import derivative

def kro(kro0, no, sw_min, sw_max, sw):
  s_hat = (sw - sw_min) / (sw_max - sw_min)
  
  # in some cases 1.0 - s_hat ends up being a very small negative number
  delta = np.maximum(0, 1.0 - s_hat)
  ko = kro0 * np.power(delta, no)

  return ko

def krw(krw0, nw, sw_min, sw_max, sw):
  s_hat  = (sw - sw_min)/ (sw_max - sw_min) 
  kw = krw0 * np.power(s_hat, nw)

  return kw


def fractional_flow(mu_w, mu_o, aw, bw, ao, bo, sw_min, sw_max, sw):
  kr_w = krw(aw, bw, sw_min, sw_max, sw)
  kr_o = kro(ao, bo, sw_min, sw_max, sw)
  
  fw = 1.0 / (1.0 + (kr_o / mu_o) * (mu_w / kr_w))
  
  return fw


def fractional_flow_derivative(mu_w, mu_o, aw, bw, ao, bo, sw_min, sw_max, sw):
  """
  Calculate the derivative of the fractional flow function using Scipy.
  Wraps the fractional flow function such that the derivative is taken 
  with respect to the eigthth parameter, i.e, the saturation.
  """
  var = 8
  args = [mu_w, mu_o, aw, bw, ao, bo, sw_min, sw_max, sw]
  def wraps(x):
    args[var] = x
    return fractional_flow(*args)
  return derivative(wraps, args[var], dx = 1e-6)


def flow_profile(sw, *params):
  mu_w, mu_o, aw, bw, ao, bo, sw_min, sw_max, vd = params

  return fractional_flow_derivative(mu_w, mu_o, aw, bw, ao, bo, sw_min, sw_max, sw) - vd


def tangent(sw, *params):
  mu_w, mu_o, aw, bw, ao, bo, sw_min, sw_max = params
  dfw_dsw = fractional_flow_derivative(mu_w, mu_o, aw, bw, ao, bo, sw_min, sw_max, sw)
  fw = fractional_flow(mu_w, mu_o, aw, bw, ao, bo, sw_min, sw_max, sw)
  
  return dfw_dsw - fw / (sw - sw_min)


def buckley_solution(total_time, porosity, diameter, length, injection_rate, mu_w, mu_o, aw, bw, ao, bo, sw_min, sw_max, output_times):
  def non_dimensional_time(t, q, l, a, phi):
    return (t * q) / (l * a * phi)

  area = math.pi * diameter * diameter / 4.0
  dimensionless_time = non_dimensional_time(total_time, injection_rate, length, area, porosity)
  vd = np.array(np.linspace(0.0, 2 * dimensionless_time, 100))
  sw = np.zeros_like(vd)
  npd = np.zeros_like(vd)
  td = np.linspace(0.0, dimensionless_time, len(npd))
  
  output_tds = []
  for t in output_times:
    if t >= total_time:
      raise ValueError('Output time should not be bigger than the total simualtion time!')
    output_tds.append(non_dimensional_time(t, injection_rate, length, area, porosity))

  # obtain the minimum velocity as a derivative of the fractional flow evaluated at sw_max
  vd_min = fractional_flow_derivative(aw, bw, ao, bo, mu_w, mu_o, sw_min, sw_max, sw_max)

  # now find what is the saturation value at the left of the shock front
  ini_guess = 0.5 * (sw_min + sw_max)
  params = (mu_w, mu_o, aw, bw, ao, bo, sw_min, sw_max)
  sw_at_shock_front = fsolve(tangent, ini_guess, args=params)
  
  vd_shock = fractional_flow(mu_w, mu_o, aw, bw, ao, bo, sw_min, sw_max, sw_at_shock_front) / (sw_at_shock_front - sw_min)

  sw_end_o = sw_at_shock_front
  for i in range(len(vd)):
    # obtain the saturation profile
    if vd[i] <= vd_min:
      sw[i] = sw_max
    elif vd[i] > vd_min and vd[i] <= vd_shock:
      params = (mu_w, mu_o, aw, bw, ao, bo, sw_min, sw_max, vd[i])
      sw[i] = fsolve(flow_profile, sw[i-1], params)
    else:
      sw[i] = sw_min

    # and also the number of pore volumes produced
    if td[i] < 1.0/vd_shock:
      npd[i] = td[i]
    else:
      params = (mu_w, mu_o, aw, bw, ao, bo, sw_min, sw_max, 1.0/td[i])
      sw_end = fsolve(flow_profile, sw_end_o, params)
      sw_end_o = sw_end
      fw_end = fractional_flow(mu_w, mu_o, aw, bw, ao, bo, sw_min, sw_max, sw_end)
      sw_bar = sw_end + td[i] * (1 - fw_end) 
      npd[i] = sw_bar - sw_min

    # rescale the Sw x vd output into Sw x xd for each output_time
    xd = np.linspace(0.0, 1.0, 100)
    out_sw = []
    for t in output_tds:
      x_scale = t * vd
      out_sw.append(np.interp(xd, x_scale, sw))

    # convert td to time in seconds
    out_time = np.linspace(0.0, total_time, 100)
    time = td * length * area * porosity / injection_rate
    out_npd = np.interp(out_time, time, npd)
  
  return xd, out_sw, out_time, out_npd