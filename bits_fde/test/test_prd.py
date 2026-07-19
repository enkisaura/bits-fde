import os
import pandas as pd
import numpy as np
import bits
import bits_prd # Available at https://github.com/enkisaura/bits-prd.git

from bits_fde.src import test, fde, hpl

"""
Tests for PRD with FDE

Usage: Used from pytest
======
    python -m pytest -v
"""

data_filepath = os.path.join(os.getcwd(), "bits_fde", "test", "data")
rx1_raw_filepath = os.path.join(data_filepath, "RX0100FRA_R_20261241729_00U_01S_MO.rnx")
rx2_raw_filepath = os.path.join(data_filepath, "RX0200FRA_R_20261241729_00U_01S_MO.rnx")
rx1_nmea_filepath = os.path.join(data_filepath, "RX01_nmea.txt")
rx2_nmea_filepath = os.path.join(data_filepath, "RX02_nmea.txt")
ephemeris_filepath = os.path.join(data_filepath, "TLSG00FRA_R_20261240000_01D_MN.rnx")

alpha=0.05
sigma=1.5
gt_uncertainty = 2

# Parse data
rx1_raw_pd = bits.parsers.gnss_raw.rinex_obs(rx1_raw_filepath)
rx2_raw_pd = bits.parsers.gnss_raw.rinex_obs(rx2_raw_filepath)
rx1_nmea_pd = bits.parsers.nmea.gga(rx1_nmea_filepath)
rx2_nmea_pd = bits.parsers.nmea.gga(rx2_nmea_filepath)

rx1_raw_pd["weight"] = 1/(sigma**2)

def test_global_test():
    # Compute baseline
    baseline_pd, raw_pd = bits_prd.code_prd.compute_baseline(rx_obs_pd=rx1_raw_pd, rx2_obs_pd=rx2_raw_pd,
                                                             compute_dd=False, ephemeris_filepath=ephemeris_filepath,
                                                             pos_pd_rx1=rx1_nmea_pd, pos_pd_rx2=rx2_nmea_pd)

    # Global test
    raw_pd = test.global_test(raw_pd, sigma=sigma, alpha=alpha, number_of_unknown=4, weight_column="weight",
                             time_column="unix_time", residuals_column="residuals_m")

    # HPL
    raw_pd["unix_time"] = raw_pd["unix_time"].astype(float)
    baseline_pd["unix_time"] = baseline_pd["unix_time"].astype(float)
    raw_pd = pd.merge_asof(raw_pd, baseline_pd[["unix_time", "cov_bx_rx_m", "cov_by_rx_m", "cov_bz_rx_m"]],
                           on="unix_time", direction="nearest", tolerance=0.1)
    cov = raw_pd[["cov_bx_rx_m", "cov_by_rx_m", "cov_bz_rx_m"]].astype(float)
    d_major = np.sqrt(cov.sum(axis=1))
    raw_pd["uncertainty_m"] = d_major
    raw_pd = hpl.hpl(raw_pd, sigma=sigma, alpha=alpha, dof=4, uncertainty_column= "uncertainty_m",
                     steering_vector_column=("e_x_rx1", "e_y_rx1", "e_z_rx1"), weight_column="weight",
                     time_column="unix_time", residuals_column="residuals_m")

    # Add results to baseline_pd
    baseline_pd = pd.merge_asof(baseline_pd, raw_pd[["unix_time", "valid_estimate", "hpl_m"]], on="unix_time",
                                direction="nearest", tolerance=0.1)

    # Compute ground_truth
    baseline_gt_pd = pd.merge_asof(rx1_nmea_pd, rx2_nmea_pd, on="unix_time", suffixes=("_rx1", "_rx2"),
                                   direction="nearest", tolerance=0.1)
    baseline_gt_pd["bx_rx_m"] = baseline_gt_pd["x_rx_m_rx1"] - baseline_gt_pd["x_rx_m_rx2"]
    baseline_gt_pd["by_rx_m"] = baseline_gt_pd["y_rx_m_rx1"] - baseline_gt_pd["y_rx_m_rx2"]
    baseline_gt_pd["bz_rx_m"] = baseline_gt_pd["z_rx_m_rx1"] - baseline_gt_pd["z_rx_m_rx2"]
    baseline_gt_pd["baseline_m"] = np.sqrt(baseline_gt_pd["bx_rx_m"]**2 + baseline_gt_pd["by_rx_m"]**2 + baseline_gt_pd["bz_rx_m"]**2)

    baseline_pd = pd.merge_asof(baseline_pd, baseline_gt_pd[["unix_time", "baseline_m"]], on="unix_time", suffixes=("", "_gt"),
                                direction="nearest", tolerance=0.1)

    baseline_pd["error_m"] = baseline_pd["baseline_m"] - baseline_pd["baseline_m_gt"]

    baseline_pd = baseline_pd[baseline_pd["valid_estimate"] == True]

    txt = f"FDE yield insufficient performances with {len(baseline_pd)} valid estimate and {baseline_pd["error_m"].mean()}m mean error for {baseline_pd["hpl_m"].mean()}m mean protection level."
    assert (len(baseline_pd) > 0) and (abs(baseline_pd["error_m"]) - gt_uncertainty < baseline_pd["hpl_m"]).all(), txt

def test_classic_fde():
    # Compute baseline
    baseline_pd, raw_pd = bits_prd.code_prd.compute_baseline(rx_obs_pd=rx1_raw_pd, rx2_obs_pd=rx2_raw_pd,
                                                             compute_dd=False, ephemeris_filepath=ephemeris_filepath,
                                                             pos_pd_rx1=rx1_nmea_pd, pos_pd_rx2=rx2_nmea_pd)

    # Classic FDE
    baseline_pd, raw_pd = (
        fde.classic(raw_pd, positioning_func=bits_prd.code_prd.window_compute_baseline, alpha=alpha, sigma=sigma,
                    steering_vector_column=("e_x_rx1", "e_y_rx1", "e_z_rx1"), max_iter=20 , number_of_unknown=4,
                    verbose=True, mode="sd"))

    # HPL
    raw_pd["unix_time"] = raw_pd["unix_time"].astype(float)
    baseline_pd["unix_time"] = baseline_pd["unix_time"].astype(float)
    raw_pd = pd.merge_asof(raw_pd, baseline_pd[["unix_time", "cov_bx_rx_m", "cov_by_rx_m", "cov_bz_rx_m"]],
                           on="unix_time", direction="nearest", tolerance=0.1)
    valid_mask = raw_pd["valid_estimate"] == True
    valid_raw_pd = raw_pd[valid_mask]
    cov = raw_pd[valid_mask][["cov_bx_rx_m", "cov_by_rx_m", "cov_bz_rx_m"]].astype(float)
    d_major = np.sqrt(cov.sum(axis=1))
    valid_raw_pd["uncertainty_m"] = d_major
    valid_raw_pd = hpl.hpl(valid_raw_pd, sigma=sigma, alpha=0.05, dof=4, uncertainty_column= "uncertainty_m",
                     steering_vector_column=("e_x_rx1", "e_y_rx1", "e_z_rx1"), weight_column="weight",
                     time_column="unix_time", residuals_column="residuals_m")

    # Add results to baseline_pd
    baseline_pd = pd.merge_asof(baseline_pd, valid_raw_pd[["unix_time", "hpl_m"]], on="unix_time",
                                direction="nearest", tolerance=0.1)

    # Compute ground_truth
    baseline_gt_pd = pd.merge_asof(rx1_nmea_pd, rx2_nmea_pd, on="unix_time", suffixes=("_rx1", "_rx2"),
                                   direction="nearest", tolerance=0.1)
    baseline_gt_pd["bx_rx_m"] = baseline_gt_pd["x_rx_m_rx1"] - baseline_gt_pd["x_rx_m_rx2"]
    baseline_gt_pd["by_rx_m"] = baseline_gt_pd["y_rx_m_rx1"] - baseline_gt_pd["y_rx_m_rx2"]
    baseline_gt_pd["bz_rx_m"] = baseline_gt_pd["z_rx_m_rx1"] - baseline_gt_pd["z_rx_m_rx2"]
    baseline_gt_pd["baseline_m"] = np.sqrt(baseline_gt_pd["bx_rx_m"]**2 + baseline_gt_pd["by_rx_m"]**2 + baseline_gt_pd["bz_rx_m"]**2)

    baseline_pd = pd.merge_asof(baseline_pd, baseline_gt_pd[["unix_time", "baseline_m"]], on="unix_time", suffixes=("", "_gt"),
                                direction="nearest", tolerance=0.1)

    baseline_pd["error_m"] = baseline_pd["baseline_m"] - baseline_pd["baseline_m_gt"]

    baseline_pd = baseline_pd[baseline_pd["valid_estimate"] == True]

    txt = f"FDE yield insufficient performances with {len(baseline_pd)} valid estimate and {baseline_pd["error_m"].mean()}m mean error for {baseline_pd["hpl_m"].mean()}m mean protection level."
    assert (len(baseline_pd) > 0) and (abs(baseline_pd["error_m"]) - gt_uncertainty < baseline_pd["hpl_m"]).all(), txt

def test_subset_testing():
    # Compute baseline
    baseline_pd, raw_pd = bits_prd.code_prd.compute_baseline(rx_obs_pd=rx1_raw_pd, rx2_obs_pd=rx2_raw_pd,
                                                             compute_dd=False, ephemeris_filepath=ephemeris_filepath,
                                                             pos_pd_rx1=rx1_nmea_pd, pos_pd_rx2=rx2_nmea_pd)

    # FDE
    baseline_pd, raw_pd = (
        fde.subset_test(raw_pd, positioning_func=bits_prd.code_prd.window_compute_baseline, alpha=alpha, sigma=sigma,
                        number_of_unknown=4, verbose=True, max_depth=2, mode="sd"))

    # HPL
    raw_pd["unix_time"] = raw_pd["unix_time"].astype(float)
    baseline_pd["unix_time"] = baseline_pd["unix_time"].astype(float)
    raw_pd = pd.merge_asof(raw_pd, baseline_pd[["unix_time", "cov_bx_rx_m", "cov_by_rx_m", "cov_bz_rx_m"]],
                           on="unix_time", direction="nearest", tolerance=0.1)
    valid_mask = raw_pd["valid_estimate"] == True
    valid_raw_pd = raw_pd[valid_mask]
    cov = raw_pd[valid_mask][["cov_bx_rx_m", "cov_by_rx_m", "cov_bz_rx_m"]].astype(float)
    d_major = np.sqrt(cov.sum(axis=1))
    valid_raw_pd["uncertainty_m"] = d_major
    valid_raw_pd = hpl.hpl(valid_raw_pd, sigma=sigma, alpha=0.05, dof=4, uncertainty_column= "uncertainty_m",
                     steering_vector_column=("e_x_rx1", "e_y_rx1", "e_z_rx1"), weight_column="weight",
                     time_column="unix_time", residuals_column="residuals_m")

    # Add results to baseline_pd
    baseline_pd = pd.merge_asof(baseline_pd, valid_raw_pd[["unix_time", "hpl_m"]], on="unix_time",
                                direction="nearest", tolerance=0.1)

    # Compute ground_truth
    baseline_gt_pd = pd.merge_asof(rx1_nmea_pd, rx2_nmea_pd, on="unix_time", suffixes=("_rx1", "_rx2"),
                                   direction="nearest", tolerance=0.1)
    baseline_gt_pd["bx_rx_m"] = baseline_gt_pd["x_rx_m_rx1"] - baseline_gt_pd["x_rx_m_rx2"]
    baseline_gt_pd["by_rx_m"] = baseline_gt_pd["y_rx_m_rx1"] - baseline_gt_pd["y_rx_m_rx2"]
    baseline_gt_pd["bz_rx_m"] = baseline_gt_pd["z_rx_m_rx1"] - baseline_gt_pd["z_rx_m_rx2"]
    baseline_gt_pd["baseline_m"] = np.sqrt(baseline_gt_pd["bx_rx_m"]**2 + baseline_gt_pd["by_rx_m"]**2 + baseline_gt_pd["bz_rx_m"]**2)

    baseline_pd = pd.merge_asof(baseline_pd, baseline_gt_pd[["unix_time", "baseline_m"]], on="unix_time", suffixes=("", "_gt"),
                                direction="nearest", tolerance=0.1)

    baseline_pd["error_m"] = baseline_pd["baseline_m"] - baseline_pd["baseline_m_gt"]

    baseline_pd = baseline_pd[baseline_pd["valid_estimate"] == True]

    txt = f"FDE yield insufficient performances with {len(baseline_pd)} valid estimate and {baseline_pd["error_m"].mean()}m mean error for {baseline_pd["hpl_m"].mean()}m mean protection level."
    assert (len(baseline_pd) > 0) and (abs(baseline_pd["error_m"]) - gt_uncertainty < baseline_pd["hpl_m"]).all(), txt

def test_iterative_local_test_fde():
    # Compute baseline
    baseline_pd, raw_pd = bits_prd.code_prd.compute_baseline(rx_obs_pd=rx1_raw_pd, rx2_obs_pd=rx2_raw_pd,
                                                             compute_dd=False, ephemeris_filepath=ephemeris_filepath,
                                                             pos_pd_rx1=rx1_nmea_pd, pos_pd_rx2=rx2_nmea_pd)

    # FDE
    baseline_pd, raw_pd = (
        fde.iterative_local_test(raw_pd, positioning_func=bits_prd.code_prd.window_compute_baseline, alpha=alpha,
                                 sigma=sigma, steering_vector_column=("e_x_rx1", "e_y_rx1", "e_z_rx1"), max_iter=20 ,
                                 number_of_unknown=4, verbose=True, mode="sd"))

    # HPL
    raw_pd["unix_time"] = raw_pd["unix_time"].astype(float)
    baseline_pd["unix_time"] = baseline_pd["unix_time"].astype(float)
    raw_pd = pd.merge_asof(raw_pd, baseline_pd[["unix_time", "cov_bx_rx_m", "cov_by_rx_m", "cov_bz_rx_m"]],
                           on="unix_time", direction="nearest", tolerance=0.1)
    valid_mask = raw_pd["valid_estimate"] == True
    valid_raw_pd = raw_pd[valid_mask]
    cov = raw_pd[valid_mask][["cov_bx_rx_m", "cov_by_rx_m", "cov_bz_rx_m"]].astype(float)
    d_major = np.sqrt(cov.sum(axis=1))
    valid_raw_pd["uncertainty_m"] = d_major
    valid_raw_pd = hpl.hpl(valid_raw_pd, sigma=sigma, alpha=0.05, dof=4, uncertainty_column= "uncertainty_m",
                     steering_vector_column=("e_x_rx1", "e_y_rx1", "e_z_rx1"), weight_column="weight",
                     time_column="unix_time", residuals_column="residuals_m")

    # Add results to baseline_pd
    baseline_pd = pd.merge_asof(baseline_pd, valid_raw_pd[["unix_time", "hpl_m"]], on="unix_time",
                                direction="nearest", tolerance=0.1)

    # Compute ground_truth
    baseline_gt_pd = pd.merge_asof(rx1_nmea_pd, rx2_nmea_pd, on="unix_time", suffixes=("_rx1", "_rx2"),
                                   direction="nearest", tolerance=0.1)
    baseline_gt_pd["bx_rx_m"] = baseline_gt_pd["x_rx_m_rx1"] - baseline_gt_pd["x_rx_m_rx2"]
    baseline_gt_pd["by_rx_m"] = baseline_gt_pd["y_rx_m_rx1"] - baseline_gt_pd["y_rx_m_rx2"]
    baseline_gt_pd["bz_rx_m"] = baseline_gt_pd["z_rx_m_rx1"] - baseline_gt_pd["z_rx_m_rx2"]
    baseline_gt_pd["baseline_m"] = np.sqrt(baseline_gt_pd["bx_rx_m"]**2 + baseline_gt_pd["by_rx_m"]**2 + baseline_gt_pd["bz_rx_m"]**2)

    baseline_pd = pd.merge_asof(baseline_pd, baseline_gt_pd[["unix_time", "baseline_m"]], on="unix_time", suffixes=("", "_gt"),
                                direction="nearest", tolerance=0.1)

    baseline_pd["error_m"] = baseline_pd["baseline_m"] - baseline_pd["baseline_m_gt"]

    baseline_pd = baseline_pd[baseline_pd["valid_estimate"] == True]

    txt = f"FDE yield insufficient performances with {len(baseline_pd)} valid estimate and {baseline_pd["error_m"].mean()}m mean error for {baseline_pd["hpl_m"].mean()}m mean protection level."
    assert (len(baseline_pd) > 0) and (abs(baseline_pd["error_m"]) - gt_uncertainty < baseline_pd["hpl_m"]).all(), txt

def test_forward_backward_fde():
    # Compute baseline
    baseline_pd, raw_pd = bits_prd.code_prd.compute_baseline(rx_obs_pd=rx1_raw_pd, rx2_obs_pd=rx2_raw_pd,
                                                             compute_dd=False, ephemeris_filepath=ephemeris_filepath,
                                                             pos_pd_rx1=rx1_nmea_pd, pos_pd_rx2=rx2_nmea_pd)

    # FDE
    baseline_pd, raw_pd = fde.forward_backward(raw_pd, positioning_func=bits_prd.code_prd.window_compute_baseline,
                                               alpha=alpha, sigma=sigma,
                                               steering_vector_column=("e_x_rx1", "e_y_rx1", "e_z_rx1"), max_iter=20,
                                               number_of_unknown=4, verbose=True, mode="sd")

    # HPL
    raw_pd["unix_time"] = raw_pd["unix_time"].astype(float)
    baseline_pd["unix_time"] = baseline_pd["unix_time"].astype(float)
    raw_pd = pd.merge_asof(raw_pd, baseline_pd[["unix_time", "cov_bx_rx_m", "cov_by_rx_m", "cov_bz_rx_m"]],
                           on="unix_time", direction="nearest", tolerance=0.1)
    valid_mask = raw_pd["valid_estimate"] == True
    valid_raw_pd = raw_pd[valid_mask]
    cov = raw_pd[valid_mask][["cov_bx_rx_m", "cov_by_rx_m", "cov_bz_rx_m"]].astype(float)
    d_major = np.sqrt(cov.sum(axis=1))
    valid_raw_pd["uncertainty_m"] = d_major
    valid_raw_pd = hpl.hpl(valid_raw_pd, sigma=sigma, alpha=0.05, dof=4, uncertainty_column= "uncertainty_m",
                     steering_vector_column=("e_x_rx1", "e_y_rx1", "e_z_rx1"), weight_column="weight",
                     time_column="unix_time", residuals_column="residuals_m")

    # Add results to baseline_pd
    baseline_pd = pd.merge_asof(baseline_pd, valid_raw_pd[["unix_time", "hpl_m"]], on="unix_time",
                                direction="nearest", tolerance=0.1)

    # Compute ground_truth
    baseline_gt_pd = pd.merge_asof(rx1_nmea_pd, rx2_nmea_pd, on="unix_time", suffixes=("_rx1", "_rx2"),
                                   direction="nearest", tolerance=0.1)
    baseline_gt_pd["bx_rx_m"] = baseline_gt_pd["x_rx_m_rx1"] - baseline_gt_pd["x_rx_m_rx2"]
    baseline_gt_pd["by_rx_m"] = baseline_gt_pd["y_rx_m_rx1"] - baseline_gt_pd["y_rx_m_rx2"]
    baseline_gt_pd["bz_rx_m"] = baseline_gt_pd["z_rx_m_rx1"] - baseline_gt_pd["z_rx_m_rx2"]
    baseline_gt_pd["baseline_m"] = np.sqrt(baseline_gt_pd["bx_rx_m"]**2 + baseline_gt_pd["by_rx_m"]**2 + baseline_gt_pd["bz_rx_m"]**2)

    baseline_pd = pd.merge_asof(baseline_pd, baseline_gt_pd[["unix_time", "baseline_m"]], on="unix_time", suffixes=("", "_gt"),
                                direction="nearest", tolerance=0.1)

    baseline_pd["error_m"] = baseline_pd["baseline_m"] - baseline_pd["baseline_m_gt"]

    baseline_pd = baseline_pd[baseline_pd["valid_estimate"] == True]

    txt = f"FDE yield insufficient performances with {len(baseline_pd)} valid estimate and {baseline_pd["error_m"].mean()}m mean error for {baseline_pd["hpl_m"].mean()}m mean protection level."
    assert (len(baseline_pd) > 0) and (abs(baseline_pd["error_m"]) - gt_uncertainty < baseline_pd["hpl_m"]).all(), txt

def test_irls():
    # Compute baseline
    baseline_pd, raw_pd = bits_prd.code_prd.compute_baseline(rx_obs_pd=rx1_raw_pd, rx2_obs_pd=rx2_raw_pd,
                                                             compute_dd=False, ephemeris_filepath=ephemeris_filepath,
                                                             pos_pd_rx1=rx1_nmea_pd, pos_pd_rx2=rx2_nmea_pd)

    # FDE
    baseline_pd, raw_pd = fde.irls(raw_pd, positioning_func=bits_prd.code_prd.window_compute_baseline, alpha=alpha,
                                   steering_vector_column=("e_x_rx1", "e_y_rx1", "e_z_rx1"), estimate_column=("bx_rx_m", "by_rx_m", "bz_rx_m"),
                                   clock_bias_vector_column=None, max_iter=20, verbose=True,
                                   mode="sd")

    # HPL
    raw_pd["unix_time"] = raw_pd["unix_time"].astype(float)
    baseline_pd["unix_time"] = baseline_pd["unix_time"].astype(float)
    raw_pd = pd.merge_asof(raw_pd, baseline_pd[["unix_time", "cov_bx_rx_m", "cov_by_rx_m", "cov_bz_rx_m"]],
                           on="unix_time", direction="nearest", tolerance=0.1)
    valid_mask = raw_pd["valid_estimate"] == True
    valid_raw_pd = raw_pd[valid_mask]
    cov = raw_pd[valid_mask][["cov_bx_rx_m", "cov_by_rx_m", "cov_bz_rx_m"]].astype(float)
    d_major = np.sqrt(cov.sum(axis=1))
    valid_raw_pd["uncertainty_m"] = d_major
    valid_raw_pd = hpl.hpl(valid_raw_pd, alpha=0.05, dof=4, uncertainty_column= "uncertainty_m",
                     steering_vector_column=("e_x_rx1", "e_y_rx1", "e_z_rx1"), weight_column="weight",
                     time_column="unix_time", residuals_column="residuals_m")

    # Add results to baseline_pd
    baseline_pd = pd.merge_asof(baseline_pd, valid_raw_pd[["unix_time", "hpl_m"]], on="unix_time",
                                direction="nearest", tolerance=0.1)

    # Compute ground_truth
    baseline_gt_pd = pd.merge_asof(rx1_nmea_pd, rx2_nmea_pd, on="unix_time", suffixes=("_rx1", "_rx2"),
                                   direction="nearest", tolerance=0.1)
    baseline_gt_pd["bx_rx_m"] = baseline_gt_pd["x_rx_m_rx1"] - baseline_gt_pd["x_rx_m_rx2"]
    baseline_gt_pd["by_rx_m"] = baseline_gt_pd["y_rx_m_rx1"] - baseline_gt_pd["y_rx_m_rx2"]
    baseline_gt_pd["bz_rx_m"] = baseline_gt_pd["z_rx_m_rx1"] - baseline_gt_pd["z_rx_m_rx2"]
    baseline_gt_pd["baseline_m"] = np.sqrt(baseline_gt_pd["bx_rx_m"]**2 + baseline_gt_pd["by_rx_m"]**2 + baseline_gt_pd["bz_rx_m"]**2)

    baseline_pd = pd.merge_asof(baseline_pd, baseline_gt_pd[["unix_time", "baseline_m"]], on="unix_time", suffixes=("", "_gt"),
                                direction="nearest", tolerance=0.1)

    baseline_pd["error_m"] = baseline_pd["baseline_m"] - baseline_pd["baseline_m_gt"]

    baseline_pd = baseline_pd[baseline_pd["valid_estimate"] == True]

    txt = f"FDE yield insufficient performances with {len(baseline_pd)} valid estimate and {baseline_pd["error_m"].mean()}m mean error for {baseline_pd["hpl_m"].mean()}m mean protection level."
    assert (len(baseline_pd) > 0) and (abs(baseline_pd["error_m"]) - gt_uncertainty < baseline_pd["hpl_m"]).all(), txt

if __name__ == '__main__':
    #test_global_test()
    #test_classic_fde()
    #test_subset_testing()
    #test_iterative_local_test_fde()
    #test_forward_backward_fde()
    test_irls()