import os
import pandas as pd
import numpy as np
import bits

from bits_fde.src import spp

"""
Tests for SPP with FDE

Usage: Used from pytest
======
    python -m pytest -v
"""

data_filepath = os.path.join(os.getcwd(), "bits_fde", "test", "data")
raw_filepath = os.path.join(data_filepath, "RX0100FRA_R_20261241729_00U_01S_MO.rnx")
nmea_filepath = os.path.join(data_filepath, "RX01_nmea.txt")
ephemeris_filepath = os.path.join(data_filepath, "TLSG00FRA_R_20261240000_01D_MN.rnx")

alpha=0.05
sigma=3.5
gt_uncertainty = 40

# Parse data
raw_pd = bits.parsers.gnss_raw.rinex_obs(raw_filepath)
nmea_pd = bits.parsers.nmea.gga(nmea_filepath)


def test_global_test():
    pd_gnss_pvt, pd_gnss_raw = spp.global_test(raw_pd, alpha=alpha, sigma=sigma, ephem_filepath=ephemeris_filepath,
                                              verbose=True)

    pd_gnss_pvt = (
        pd.merge_asof(pd_gnss_pvt, nmea_pd[["unix_time", "x_rx_m", "y_rx_m", "z_rx_m"]],
                      on="unix_time", suffixes=("", "_gt"), direction="nearest", tolerance=0.1))

    pd_gnss_pvt["x_error_m"] = pd_gnss_pvt["x_rx_m"] - pd_gnss_pvt["x_rx_m_gt"]
    pd_gnss_pvt["y_error_m"] = pd_gnss_pvt["y_rx_m"] - pd_gnss_pvt["y_rx_m_gt"]
    pd_gnss_pvt["z_error_m"] = pd_gnss_pvt["z_rx_m"] - pd_gnss_pvt["z_rx_m_gt"]
    pd_gnss_pvt["error_m"] = np.sqrt(pd_gnss_pvt["x_error_m"]**2 + pd_gnss_pvt["y_error_m"]**2 + pd_gnss_pvt["z_error_m"]**2)

    pd_gnss_pvt = pd_gnss_pvt[pd_gnss_pvt["valid_estimate"] == True]

    txt = f"FDE yield insufficient performances with {len(pd_gnss_pvt)} valid estimate and {pd_gnss_pvt["error_m"].mean()}m mean error for {pd_gnss_pvt["hpl_m"].mean()}m mean protection level."
    assert (len(pd_gnss_pvt) > 0) and (abs(pd_gnss_pvt["error_m"]) - gt_uncertainty < pd_gnss_pvt["hpl_m"]).all(), txt

def test_classic_fde():
    pd_gnss_pvt, pd_gnss_raw = spp.classic_fde(raw_pd, alpha=alpha, sigma=sigma, ephem_filepath=ephemeris_filepath,
                                               max_iter=20, verbose=True)

    pd_gnss_pvt = (
        pd.merge_asof(pd_gnss_pvt, nmea_pd[["unix_time", "x_rx_m", "y_rx_m", "z_rx_m"]],
                      on="unix_time", suffixes=("", "_gt"), direction="nearest", tolerance=0.1))

    pd_gnss_pvt["x_error_m"] = pd_gnss_pvt["x_rx_m"] - pd_gnss_pvt["x_rx_m_gt"]
    pd_gnss_pvt["y_error_m"] = pd_gnss_pvt["y_rx_m"] - pd_gnss_pvt["y_rx_m_gt"]
    pd_gnss_pvt["z_error_m"] = pd_gnss_pvt["z_rx_m"] - pd_gnss_pvt["z_rx_m_gt"]
    pd_gnss_pvt["error_m"] = np.sqrt(
        pd_gnss_pvt["x_error_m"] ** 2 + pd_gnss_pvt["y_error_m"] ** 2 + pd_gnss_pvt["z_error_m"] ** 2)

    pd_gnss_pvt = pd_gnss_pvt[pd_gnss_pvt["valid_estimate"] == True]

    txt = f"FDE yield insufficient performances with {len(pd_gnss_pvt)} valid estimate and {pd_gnss_pvt["error_m"].mean()}m mean error for {pd_gnss_pvt["hpl_m"].mean()}m mean protection level."
    assert (len(pd_gnss_pvt) > 0) and (abs(pd_gnss_pvt["error_m"]) - gt_uncertainty < pd_gnss_pvt["hpl_m"]).all(), txt

def test_subset_test_fde():
    pd_gnss_pvt, pd_gnss_raw = spp.subset_test_fde(raw_pd, alpha=alpha, sigma=sigma, ephem_filepath=ephemeris_filepath,
                                                   verbose=True, max_depth=2)

    pd_gnss_pvt = (
        pd.merge_asof(pd_gnss_pvt, nmea_pd[["unix_time", "x_rx_m", "y_rx_m", "z_rx_m"]],
                      on="unix_time", suffixes=("", "_gt"), direction="nearest", tolerance=0.1))

    pd_gnss_pvt["x_error_m"] = pd_gnss_pvt["x_rx_m"] - pd_gnss_pvt["x_rx_m_gt"]
    pd_gnss_pvt["y_error_m"] = pd_gnss_pvt["y_rx_m"] - pd_gnss_pvt["y_rx_m_gt"]
    pd_gnss_pvt["z_error_m"] = pd_gnss_pvt["z_rx_m"] - pd_gnss_pvt["z_rx_m_gt"]
    pd_gnss_pvt["error_m"] = np.sqrt(
        pd_gnss_pvt["x_error_m"] ** 2 + pd_gnss_pvt["y_error_m"] ** 2 + pd_gnss_pvt["z_error_m"] ** 2)

    pd_gnss_pvt = pd_gnss_pvt[pd_gnss_pvt["valid_estimate"] == True]

    txt = f"FDE yield insufficient performances with {len(pd_gnss_pvt)} valid estimate and {pd_gnss_pvt["error_m"].mean()}m mean error for {pd_gnss_pvt["hpl_m"].mean()}m mean protection level."
    assert (len(pd_gnss_pvt) > 0) and (abs(pd_gnss_pvt["error_m"]) - gt_uncertainty < pd_gnss_pvt["hpl_m"]).all(), txt

if __name__ == '__main__':
    #test_global_test()
    #test_classic_fde()
    test_subset_test_fde()