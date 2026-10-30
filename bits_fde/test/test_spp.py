import os
import pandas as pd
import numpy as np
import bits
from bits import parse
from bits.src.utils import get_example_data_filepath, fast_parse

from bits_fde.src import spp

"""
Tests for SPP with FDE

Usage: Used from pytest
======
    python -m pytest -v
"""
# Parse data
raw_pd = fast_parse(get_example_data_filepath("raw")[0], parse.raw.rinex)
nmea_pd = fast_parse(get_example_data_filepath("pvt")[0], parse.pvt.rmc)
ephemeris_df = fast_parse(get_example_data_filepath("ephemeris")[0], parse.ephemeris.rinex)

alpha=0.05
sigma=3.5


def test_global_test(verbose=False):
    pd_gnss_pvt, pd_gnss_raw = spp.global_test(raw_pd, alpha=alpha, sigma=sigma, pd_ephemeris=ephemeris_df,
                                              verbose=verbose)

    pd_gnss_pvt = (
        pd.merge_asof(pd_gnss_pvt, nmea_pd[["time", "x_rx_m", "y_rx_m", "z_rx_m"]],
                      on="time", suffixes=("", "_gt"), direction="nearest", tolerance=pd.Timedelta(seconds=0.1)))

    pd_gnss_pvt["x_error_m"] = pd_gnss_pvt["x_rx_m"] - pd_gnss_pvt["x_rx_m_gt"]
    pd_gnss_pvt["y_error_m"] = pd_gnss_pvt["y_rx_m"] - pd_gnss_pvt["y_rx_m_gt"]
    pd_gnss_pvt["z_error_m"] = pd_gnss_pvt["z_rx_m"] - pd_gnss_pvt["z_rx_m_gt"]
    pd_gnss_pvt["error_m"] = np.sqrt(pd_gnss_pvt["x_error_m"]**2 + pd_gnss_pvt["y_error_m"]**2 + pd_gnss_pvt["z_error_m"]**2)

    pd_gnss_pvt = pd_gnss_pvt[pd_gnss_pvt["valid_estimate"] == True]

    report = (f"### Report for Global Test:\n"
              f"# {len(pd_gnss_pvt)} valid estimates\n"
              f"# {pd_gnss_pvt["error_m"].mean()}m mean error\n"
              f"# {pd_gnss_pvt["hpl_m"].mean()}m mean protection level.\n")

    if verbose:
        print(report)

    txt = f"FDE yield insufficient performances: \n{report}"
    assert (len(pd_gnss_pvt) > 0) and (abs(pd_gnss_pvt["error_m"]) < pd_gnss_pvt["hpl_m"]).all(), txt

    return report

def test_classic_fde(verbose=False):
    pd_gnss_pvt, pd_gnss_raw = spp.classic_fde(raw_pd, alpha=alpha, sigma=sigma, pd_ephemeris=ephemeris_df,
                                               max_iter=20, verbose=verbose)

    pd_gnss_pvt = (
        pd.merge_asof(pd_gnss_pvt, nmea_pd[["time", "x_rx_m", "y_rx_m", "z_rx_m"]],
                      on="time", suffixes=("", "_gt"), direction="nearest", tolerance=pd.Timedelta(seconds=0.1)))

    pd_gnss_pvt["x_error_m"] = pd_gnss_pvt["x_rx_m"] - pd_gnss_pvt["x_rx_m_gt"]
    pd_gnss_pvt["y_error_m"] = pd_gnss_pvt["y_rx_m"] - pd_gnss_pvt["y_rx_m_gt"]
    pd_gnss_pvt["z_error_m"] = pd_gnss_pvt["z_rx_m"] - pd_gnss_pvt["z_rx_m_gt"]
    pd_gnss_pvt["error_m"] = np.sqrt(
        pd_gnss_pvt["x_error_m"] ** 2 + pd_gnss_pvt["y_error_m"] ** 2 + pd_gnss_pvt["z_error_m"] ** 2)

    pd_gnss_pvt = pd_gnss_pvt[pd_gnss_pvt["valid_estimate"] == True]

    report = (f"### Report for Classic FDE:\n"
              f"# {len(pd_gnss_pvt)} valid estimates\n"
              f"# {pd_gnss_pvt["error_m"].mean()}m mean error\n"
              f"# {pd_gnss_pvt["hpl_m"].mean()}m mean protection level.\n")

    if verbose:
        print(report)

    txt = f"FDE yield insufficient performances:\n{report}"
    assert (len(pd_gnss_pvt) > 0) and (abs(pd_gnss_pvt["error_m"]) < pd_gnss_pvt["hpl_m"]).all(), txt

    return report

def test_subset_test_fde(verbose=False):
    pd_gnss_pvt, pd_gnss_raw = spp.subset_test_fde(raw_pd, alpha=alpha, sigma=sigma, pd_ephemeris=ephemeris_df,
                                                   verbose=verbose, max_depth=2)

    pd_gnss_pvt = (
        pd.merge_asof(pd_gnss_pvt, nmea_pd[["time", "x_rx_m", "y_rx_m", "z_rx_m"]],
                      on="time", suffixes=("", "_gt"), direction="nearest", tolerance=pd.Timedelta(seconds=0.1)))

    pd_gnss_pvt["x_error_m"] = pd_gnss_pvt["x_rx_m"] - pd_gnss_pvt["x_rx_m_gt"]
    pd_gnss_pvt["y_error_m"] = pd_gnss_pvt["y_rx_m"] - pd_gnss_pvt["y_rx_m_gt"]
    pd_gnss_pvt["z_error_m"] = pd_gnss_pvt["z_rx_m"] - pd_gnss_pvt["z_rx_m_gt"]
    pd_gnss_pvt["error_m"] = np.sqrt(
        pd_gnss_pvt["x_error_m"] ** 2 + pd_gnss_pvt["y_error_m"] ** 2 + pd_gnss_pvt["z_error_m"] ** 2)

    pd_gnss_pvt = pd_gnss_pvt[pd_gnss_pvt["valid_estimate"] == True]
    pd_gnss_pvt.dropna(inplace=True)

    report = (f"### Report for subset test:\n"
              f"# {len(pd_gnss_pvt)} valid estimates\n"
              f"# {pd_gnss_pvt["error_m"].mean()}m mean error\n"
              f"# {pd_gnss_pvt["hpl_m"].mean()}m mean protection level.\n")

    if verbose:
        print(report)

    txt = f"FDE yield insufficient performances:\n{report}"
    assert (len(pd_gnss_pvt) > 0) and (abs(pd_gnss_pvt["error_m"]) < pd_gnss_pvt["hpl_m"]).all(), txt

    return report

def test_iterative_local_test_fde(verbose=False):
    pd_gnss_pvt, pd_gnss_raw = spp.iterative_local_test_fde(raw_pd, alpha=alpha, sigma=sigma,
                                                            pd_ephemeris=ephemeris_df, max_iter=20,
                                                            verbose=verbose)

    pd_gnss_pvt = (
        pd.merge_asof(pd_gnss_pvt, nmea_pd[["time", "x_rx_m", "y_rx_m", "z_rx_m"]],
                      on="time", suffixes=("", "_gt"), direction="nearest", tolerance=pd.Timedelta(seconds=0.1)))

    pd_gnss_pvt["x_error_m"] = pd_gnss_pvt["x_rx_m"] - pd_gnss_pvt["x_rx_m_gt"]
    pd_gnss_pvt["y_error_m"] = pd_gnss_pvt["y_rx_m"] - pd_gnss_pvt["y_rx_m_gt"]
    pd_gnss_pvt["z_error_m"] = pd_gnss_pvt["z_rx_m"] - pd_gnss_pvt["z_rx_m_gt"]
    pd_gnss_pvt["error_m"] = np.sqrt(
        pd_gnss_pvt["x_error_m"] ** 2 + pd_gnss_pvt["y_error_m"] ** 2 + pd_gnss_pvt["z_error_m"] ** 2)

    pd_gnss_pvt = pd_gnss_pvt[pd_gnss_pvt["valid_estimate"] == True]

    report = (f"### Report for local test:\n"
              f"# {len(pd_gnss_pvt)} valid estimates\n"
              f"# {pd_gnss_pvt["error_m"].mean()}m mean error\n"
              f"# {pd_gnss_pvt["hpl_m"].mean()}m mean protection level.\n")

    if verbose:
        print(report)

    txt = f"FDE yield insufficient performances:\n{report}"
    assert (len(pd_gnss_pvt) > 0) and (abs(pd_gnss_pvt["error_m"]) < pd_gnss_pvt["hpl_m"]).all(), txt

    return report

def test_forward_backward_fde(verbose=False):
    pd_gnss_pvt, pd_gnss_raw = spp.forward_backward_fde(raw_pd, alpha=alpha, sigma=sigma, pd_ephemeris=ephemeris_df,
                                                        max_iter=20, verbose=verbose)

    pd_gnss_pvt = (
        pd.merge_asof(pd_gnss_pvt, nmea_pd[["time", "x_rx_m", "y_rx_m", "z_rx_m"]],
                      on="time", suffixes=("", "_gt"), direction="nearest", tolerance=pd.Timedelta(seconds=0.1)))

    pd_gnss_pvt["x_error_m"] = pd_gnss_pvt["x_rx_m"] - pd_gnss_pvt["x_rx_m_gt"]
    pd_gnss_pvt["y_error_m"] = pd_gnss_pvt["y_rx_m"] - pd_gnss_pvt["y_rx_m_gt"]
    pd_gnss_pvt["z_error_m"] = pd_gnss_pvt["z_rx_m"] - pd_gnss_pvt["z_rx_m_gt"]
    pd_gnss_pvt["error_m"] = np.sqrt(
        pd_gnss_pvt["x_error_m"] ** 2 + pd_gnss_pvt["y_error_m"] ** 2 + pd_gnss_pvt["z_error_m"] ** 2)

    pd_gnss_pvt = pd_gnss_pvt[pd_gnss_pvt["valid_estimate"] == True]

    report = (f"### Report for forward-backward:\n"
              f"# {len(pd_gnss_pvt)} valid estimates\n"
              f"# {pd_gnss_pvt["error_m"].mean()}m mean error\n"
              f"# {pd_gnss_pvt["hpl_m"].mean()}m mean protection level.\n")

    if verbose:
        print(report)

    txt = f"FDE yield insufficient performances:\n{report}"
    assert (len(pd_gnss_pvt) > 0) and (abs(pd_gnss_pvt["error_m"]) < pd_gnss_pvt["hpl_m"]).all(), txt

    return report

def test_danish_fde(verbose=False):
    pd_gnss_pvt, pd_gnss_raw = spp.danish_fde(raw_pd, alpha=alpha, sigma=sigma, pd_ephemeris=ephemeris_df,
                                              max_iter=20, verbose=verbose)

    pd_gnss_pvt = (
        pd.merge_asof(pd_gnss_pvt, nmea_pd[["time", "x_rx_m", "y_rx_m", "z_rx_m"]],
                      on="time", suffixes=("", "_gt"), direction="nearest", tolerance=pd.Timedelta(seconds=0.1)))

    pd_gnss_pvt["x_error_m"] = pd_gnss_pvt["x_rx_m"] - pd_gnss_pvt["x_rx_m_gt"]
    pd_gnss_pvt["y_error_m"] = pd_gnss_pvt["y_rx_m"] - pd_gnss_pvt["y_rx_m_gt"]
    pd_gnss_pvt["z_error_m"] = pd_gnss_pvt["z_rx_m"] - pd_gnss_pvt["z_rx_m_gt"]
    pd_gnss_pvt["error_m"] = np.sqrt(
        pd_gnss_pvt["x_error_m"] ** 2 + pd_gnss_pvt["y_error_m"] ** 2 + pd_gnss_pvt["z_error_m"] ** 2)

    pd_gnss_pvt = pd_gnss_pvt[pd_gnss_pvt["valid_estimate"] == True]

    report = (f"### Report for Danish FDE:\n"
              f"# {len(pd_gnss_pvt)} valid estimates\n"
              f"# {pd_gnss_pvt["error_m"].mean()}m mean error\n"
              f"# {pd_gnss_pvt["hpl_m"].mean()}m mean protection level.\n")

    if verbose:
        print(report)

    txt = f"FDE yield insufficient performances:\n{report}"
    assert (len(pd_gnss_pvt) > 0) and (abs(pd_gnss_pvt["error_m"]) < pd_gnss_pvt["hpl_m"]).all(), txt

    return report

def test_irls(verbose=False):
    pd_gnss_pvt, pd_gnss_raw = spp.irls_fde(raw_pd, alpha=alpha, pd_ephemeris=ephemeris_df, max_iter=20,
                                            verbose=verbose)

    pd_gnss_pvt = (
        pd.merge_asof(pd_gnss_pvt, nmea_pd[["time", "x_rx_m", "y_rx_m", "z_rx_m"]],
                      on="time", suffixes=("", "_gt"), direction="nearest", tolerance=pd.Timedelta(seconds=0.1)))

    pd_gnss_pvt["x_error_m"] = pd_gnss_pvt["x_rx_m"] - pd_gnss_pvt["x_rx_m_gt"]
    pd_gnss_pvt["y_error_m"] = pd_gnss_pvt["y_rx_m"] - pd_gnss_pvt["y_rx_m_gt"]
    pd_gnss_pvt["z_error_m"] = pd_gnss_pvt["z_rx_m"] - pd_gnss_pvt["z_rx_m_gt"]
    pd_gnss_pvt["error_m"] = np.sqrt(
        pd_gnss_pvt["x_error_m"] ** 2 + pd_gnss_pvt["y_error_m"] ** 2 + pd_gnss_pvt["z_error_m"] ** 2)

    pd_gnss_pvt = pd_gnss_pvt[pd_gnss_pvt["valid_estimate"] == True]

    report = (f"### Report for IRLS:\n"
              f"# {len(pd_gnss_pvt)} valid estimates\n"
              f"# {pd_gnss_pvt["error_m"].mean()}m mean error\n"
              f"# {pd_gnss_pvt["hpl_m"].mean()}m mean protection level.\n")

    if verbose:
        print(report)

    txt = f"FDE yield insufficient performances:\n{report}"
    assert (len(pd_gnss_pvt) > 0) and (abs(pd_gnss_pvt["error_m"]) < pd_gnss_pvt["hpl_m"]).all(), txt

    return report

if __name__ == '__main__':
    reports = []
    reports.append(test_global_test(verbose=True))
    reports.append(test_classic_fde(verbose=True))
    reports.append(test_subset_test_fde(verbose=True))
    reports.append(test_iterative_local_test_fde(verbose=True))
    reports.append(test_forward_backward_fde(verbose=True))
    reports.append(test_danish_fde(verbose=True))
    reports.append(test_irls(verbose=True))

    for report in reports:
        print(report)