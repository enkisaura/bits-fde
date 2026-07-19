# bits-fde
Baguette in the sky module for Fault Detection and Exclusion

The module depends on [Baguette in the sky](https://github.com/enkisaura/Baguette-In-The-Sky).

---
## Usage
### SPP
```python
import os
import bits

from bits_fde.src import spp

data_filepath = os.path.join(os.getcwd(), "bits_fde", "test", "data")
raw_filepath = os.path.join(data_filepath, "RX0100FRA_R_20261241729_00U_01S_MO.rnx")
ephemeris_filepath = os.path.join(data_filepath, "TLSG00FRA_R_20261240000_01D_MN.rnx")

alpha=0.05
sigma=3.5

# Parse data
raw_pd = bits.parsers.gnss_raw.rinex_obs(raw_filepath)

# Apply FDE
# Global test
pd_gnss_pvt_gt, pd_gnss_raw_gt = spp.global_test(raw_pd, alpha=alpha, sigma=sigma, ephem_filepath=ephemeris_filepath)

# Classic Method
pd_gnss_pvt_cm, pd_gnss_raw_cm = spp.classic_fde(raw_pd, alpha=alpha, sigma=sigma, ephem_filepath=ephemeris_filepath)

# Subset Test
pd_gnss_pvt_st, pd_gnss_raw_st = spp.subset_test_fde(raw_pd, alpha=alpha, sigma=sigma, ephem_filepath=ephemeris_filepath)

# Iterative Local Test
pd_gnss_pvt_lt, pd_gnss_raw_lt = spp.iterative_local_test_fde(raw_pd, alpha=alpha, sigma=sigma, 
                                                              ephem_filepath=ephemeris_filepath)

# Forward-Backward
pd_gnss_pvt, pd_gnss_raw = spp.forward_backward_fde(raw_pd, alpha=alpha, sigma=sigma, ephem_filepath=ephemeris_filepath)

# Danish Method
pd_gnss_pvt_dm, pd_gnss_raw_dm = spp.danish_fde(raw_pd, alpha=alpha, sigma=sigma, ephem_filepath=ephemeris_filepath)

# Iterative Reweighted Least Square
pd_gnss_pvt_irls, pd_gnss_raw_irls = spp.irls_fde(raw_pd, alpha=alpha, ephem_filepath=ephemeris_filepath)
```

## Other positioning algorithm
```python
import os
import pandas as pd
import numpy as np
import bits
import bits_prd # Available at https://github.com/enkisaura/bits-prd.git

from bits_fde.src import test, fde, hpl

data_filepath = os.path.join(os.getcwd(), "bits_fde", "test", "data")
rx1_raw_filepath = os.path.join(data_filepath, "RX0100FRA_R_20261241729_00U_01S_MO.rnx")
rx2_raw_filepath = os.path.join(data_filepath, "RX0200FRA_R_20261241729_00U_01S_MO.rnx")
ephemeris_filepath = os.path.join(data_filepath, "TLSG00FRA_R_20261240000_01D_MN.rnx")

alpha=0.05
sigma=1.5

# Parse data
rx1_raw_pd = bits.parsers.gnss_raw.rinex_obs(rx1_raw_filepath)
rx2_raw_pd = bits.parsers.gnss_raw.rinex_obs(rx2_raw_filepath)

rx1_raw_pd["weight"] = 1/(sigma**2)

# Compute baseline
_, raw_pd = bits_prd.code_prd.compute_baseline(rx_obs_pd=rx1_raw_pd, rx2_obs_pd=rx2_raw_pd, compute_dd=False, 
                                               ephemeris_filepath=ephemeris_filepath)

# IRLS
baseline_pd, raw_pd = fde.irls(raw_pd, positioning_func=bits_prd.code_prd.window_compute_baseline, alpha=alpha,
                               steering_vector_column=("e_x_rx1", "e_y_rx1", "e_z_rx1"), 
                               estimate_column=("bx_rx_m", "by_rx_m", "bz_rx_m"),
                               clock_bias_vector_column=None, max_iter=20, verbose=True, mode="sd")

# HPL
raw_pd["unix_time"] = raw_pd["unix_time"].astype(float)
baseline_pd["unix_time"] = baseline_pd["unix_time"].astype(float)
raw_pd = pd.merge_asof(raw_pd, baseline_pd[["unix_time", "cov_bx_rx_m", "cov_by_rx_m", "cov_bz_rx_m"]],
                       on="unix_time", direction="nearest", tolerance=0.1)
cov = raw_pd[raw_pd][["cov_bx_rx_m", "cov_by_rx_m", "cov_bz_rx_m"]].astype(float)
d_major = np.sqrt(cov.sum(axis=1))
raw_pd["uncertainty_m"] = d_major
valid_raw_pd = hpl.hpl(raw_pd, alpha=0.05, dof=4, uncertainty_column= "uncertainty_m",
                 steering_vector_column=("e_x_rx1", "e_y_rx1", "e_z_rx1"), weight_column="weight",
                 time_column="unix_time", residuals_column="residuals_m")
```