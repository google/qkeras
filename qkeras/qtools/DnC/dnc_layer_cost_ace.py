# Copyright 2019 Google LLC
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""divide_and_conquer per layer cost modeling using ACE and data fitting.

For a given layer with its hardware design params, predict its cost
in actual ASIC implementation using ACE metric and actual MAC gates data points.
"""

import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Rule-of-thumb mapping between bits and gates in memory area estimate.
MemoryGatesPerBit = {
    'Register': 10.0,
    'SRAM': 1.0,
    'ROM': 0.1,
}


# Previously calculated 3D polynomial coefficients with relative MAE<5%.
MAC_POLY3D_PARAMS = np.array([7.70469119, 13.76199652, -92.15756665])


# MAC area data points generated from go/mac_vs_area.
MAC24 = pd.read_csv(io.StringIO('''
283,280,286,313,325,336,356,,
274,290,325,372,401,428,485,,
285,325,388,510,568,614,713,,
308,372,509,750,865,1002,1167,,
336,427,617,1003,1151,1309,,,
356,480,722,1165,,,,,
'''), header=None)

MAC32 = pd.read_csv(io.StringIO('''
391,365,377,410,453,433,458,507,
364,382,418,466,497,521,578,685,
378,418,485,594,659,721,832,1035,
408,466,596,843,1029,1151,1321,1642,
432,521,724,1153,1363,1512,1797,,
457,578,830,1330,1551,1782,2273,,
'''), header=None)

MAC40 = pd.read_csv(io.StringIO('''
458,457,470,500,522,527,551,605,664
457,475,513,561,597,616,670,782,888
470,513,579,699,766,816,928,1150,1358
499,561,699,996,1161,1273,1499,1850,2189
527,612,818,1275,1545,1691,2054,2516,
549,670,927,1496,1798,2035,2490,3294,
'''), header=None)

MAC48 = pd.read_csv(io.StringIO('''
595,550,566,594,659,624,642,694,745
551,566,607,654,727,707,763,881,984
566,607,679,794,871,921,1017,1270,1489
594,655,793,1097,1285,1401,1668,2101,2378
624,711,921,1397,1816,1950,2277,2763,3301
642,762,1015,1669,1974,2264,2718,3631,4415
'''), header=None)


def mac_gates_polynomial_3d(xyz, a, b, c):
  """Using a 3d polynomial function to model MAC area.

  This function models the MAC area to be the sum of multipler, accumulator
  and a constant shift. Particularly, multiplier area is modeled to be linear
  # to input_bits * weight_bits, per ACE rule.

  Args:
    xyz: tuple includes input, weight and accumulator bits.
    a: polynomial coefficient 0.
    b: polynomial coefficient 1.
    c: polynomial coefficient 2.

  Returns:
    MAC area predicted by the function.
  """
  x, y, z = xyz
  return a * x * y + b * z + c


def gen_mac_gate_model(do_plot=False):
  """Generate the polynomial cost model coefficients using given data.

  Args:
    do_plot: Bool indicates whether plot the raw data and the fitted curve.

  Returns:
    params: The esitimated params of the polynomical function.
    mae_predict: Calculate the mean absolute error of the predictions.
    parameter_std_deviation: one standard deviation errors on the parameters,
      indicating the uncertainties of the params.
  """
  # acc bits, 1st index
  abit = np.array([24, 32, 40, 48])
  abit = np.repeat(abit, 54)

  # weight bits, 2nd index
  wbit = np.array([1, 2, 4, 8, 12, 16])
  wbit = np.tile(np.repeat(wbit, 9), 4)

  # input bits, 3rd index
  xbit = np.array([1, 2, 4, 8, 10, 12, 16, 24, 32])
  xbit = np.tile(xbit, 24)

  # Record all mac area data points associated with each accumulator bitwidth
  mac_arrs = []
  # Record the start and end index of the mac area data points
  # associated with each accumulator bitwidth
  mac_arrs_index = {}
  # Record index of all valid data points
  valid_index = []
  start_pos = 0

  for (mac_acc, acc_bits) in zip(
      [MAC24, MAC32, MAC40, MAC48], [24, 32, 40, 48]):
    cur_mac = mac_acc.to_numpy().reshape(-1)
    # Filter out nan data points
    cur_valid_index = ~np.isnan(cur_mac)
    cur_valid_mac = cur_mac[cur_valid_index]
    # Record the data length for each accumulator bits
    end_pos = start_pos + len(cur_valid_mac)
    mac_arrs_index[acc_bits] = (start_pos, end_pos)
    # Append mac areas of each accumulator bits to a list
    mac_arrs += list(cur_valid_mac)
    start_pos = end_pos
    valid_index += list(cur_valid_index)

  # Filter out invalid data
  xbit = xbit[valid_index]
  wbit = wbit[valid_index]
  abit = abit[valid_index]

  # curve fitting for all data points
  params, covariance = curve_fit(
      mac_gates_polynomial_3d, (xbit, wbit, abit), mac_arrs)

  # Compute one standard deviation errors on the parameters.
  parameter_std_deviation = np.sqrt(np.diag(covariance))

  # Calculate the mean absolute error between prediction and given data.
  mac_predict = mac_gates_polynomial_3d((xbit, wbit, abit), *params)
  mae = np.mean(np.abs(mac_predict - mac_arrs))
  mae_predict = mae / np.mean(mac_arrs)

  if do_plot:
    # Plot all raw data points
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xbit, wbit, mac_arrs, label='Data')

    ax.set_xlabel('X_bits')
    ax.set_ylabel('W_bits')
    ax.set_zlabel('MAC')

    plt.title('MAC area data points')
    plt.show()

    # Generate a mesh grid for plotting.
    x_fit = np.linspace(min(xbit), max(xbit), 50)
    w_fit = np.linspace(min(wbit), max(wbit), 50)
    xmesh, wmesh = np.meshgrid(x_fit, w_fit)

    fig = plt.figure(figsize=(16, 16))
    index = 1

    # Plotting 3D fitting curve for each accumulator bitwidth
    for acc_bits in [24, 32, 40, 48]:
      ax = fig.add_subplot(2, 2, index, projection='3d')

      start_pos = mac_arrs_index[acc_bits][0]
      end_pos = mac_arrs_index[acc_bits][1]
      ax.scatter(xbit[start_pos:end_pos], wbit[start_pos:end_pos],
                 mac_arrs[start_pos:end_pos], label='Data')

      amesh = np.full(shape=(50, 50), fill_value=acc_bits)
      poly_fit = mac_gates_polynomial_3d((xmesh, wmesh, amesh), *params)

      ax.plot_surface(
          xmesh, wmesh, poly_fit, cmap='viridis', alpha=0.8,
          label=f'Fitted Surface | acc_bits={acc_bits}')

      ax.set_xlabel('X')
      ax.set_ylabel('W')
      ax.set_zlabel('MAC')
      ax.set_title(f'accumulator bitwidth: {acc_bits}')
      index += 1

    plt.show()

  return params, mae_predict, parameter_std_deviation


def get_ace_mac_gates(xbit, wbit, abit, regen_params=False):
  """Function to estimate MAC area, including 1 multipler and 1 accumulator.

  Args:
    xbit: int. input bits.
    wbit: int. weight bits.
    abit: int. accumulator bits.
    regen_params: Bool. If True, regenerate the MAC cost model coefficients.
      If False, reuse the previously generated model coefficients.

  Returns:
    Estimated MAC gates.
  """
  if regen_params:
    mac_params, _, _ = gen_mac_gate_model(do_plot=True)
  else:
    mac_params = MAC_POLY3D_PARAMS

  return mac_gates_polynomial_3d((xbit, wbit, abit), *mac_params)
