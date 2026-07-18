import pandas as pd
import numpy as np
from typing import Callable
from tqdm import tqdm

import bits

from bits_fde.src import test, fde, hpl

def global_test(pd_gnss_raw: pd.DataFrame, alpha:float=0.05, sigma:float|None=None, pd_ephemeris: pd.DataFrame = None,
                ephem_filepath: str = None, approx_pvt: tuple[float, float, float]=(0, 0, 0), verbose=False) \
        -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes position using BITS and performs a global test

    :param pd_gnss_raw: BITS raw dataframe
    :param alpha: Significance level
    :param sigma: Standard deviation of measurement noise set to None to use 1/weight² as sigma
    :param pd_ephemeris: BITS ephemeris dataframe
    :param ephem_filepath: Path of a rinex nav file
    :param approx_pvt: Position (ECEF meters) at initialization (default -> centre of earth)
    :param verbose: set to True for verbose output
    :return: BITS pvt dataframe, BITS raw dataframe
    """

    # Add standard deviation
    if sigma is not None:
        pd_gnss_raw["weight"] = 1/(sigma**2)

    # Compute position using bits' SPP
    pd_gnss_pvt, pd_gnss_raw = (
        bits.spp.get_position_estimate(pd_gnss_raw, pd_ephemeris=pd_ephemeris,  ephem_filepath=ephem_filepath,
                                       approx_pvt=approx_pvt, verbose=verbose))

    # Add unix_time for easier sorting
    pd_gnss_raw["unix_time"] = pd_gnss_raw["time"].apply(
        lambda gnss_timestamp: gnss_timestamp.pd_timestamp().timestamp())
    pd_gnss_pvt["unix_time"] = pd_gnss_pvt["time"].apply(
        lambda gnss_timestamp: gnss_timestamp.pd_timestamp().timestamp())

    # Clean up
    pd_gnss_raw = pd_gnss_raw.sort_values("unix_time").reset_index(drop=True)

    # Apply integrity monitoring for each timestamp group
    out_raw_pd = pd.DataFrame()
    for timestamp, group in tqdm(pd_gnss_raw.groupby("unix_time", sort=True), desc="Applying global test"):
        # Get residuals
        residuals = group["residuals_m"].to_numpy().reshape(-1, 1)

        # Get weight matrix
        W = np.diag(group["weight"])

        # Build G
        G = np.vstack([group[column].to_numpy() for column in ("e_x", "e_y", "e_z")])
        G = G.transpose()

        # Get number of unknowns
        number_of_unknown = 3 + len(pd_gnss_raw["gnss_id"].unique())

        result, chi2_stat = test.window_global_test(residuals, W, alpha, number_of_unknown)

        group["valid_estimate"] = result
        group["chi2_stat"] = float(chi2_stat.item())

        # Compute HPL
        try:
            cov = pd_gnss_pvt[pd_gnss_pvt["unix_time"] == timestamp][["cov_xx_rx_m", "cov_yy_rx_m", "cov_zz_rx_m"]].iloc[0]
            d_major = np.sqrt(cov.sum())
            noise, bias, protection = hpl.window_hpl(d_major, G, W, residuals, alpha, dof=number_of_unknown)

            group[["hpl_noise_m", "hpl_bias_m", "hpl_m"]] = float(noise.item()), float(bias.item()), float(protection.item())
        except Exception as e:
            txt = f" Could not compute HPL at timestamp {timestamp}: {e}"
            raise ValueError(txt)

        out_raw_pd = pd.concat([out_raw_pd, group], axis=0)

    # Add test results to PVT dataframe
    pd_gnss_pvt = pd.merge_asof(pd_gnss_pvt, out_raw_pd[["unix_time", "valid_estimate", "hpl_noise_m", "hpl_bias_m", "hpl_m"]],
                                on="unix_time", direction="nearest", tolerance=0.1)

    return pd_gnss_pvt, out_raw_pd

def classic_fde(pd_gnss_raw: pd.DataFrame, alpha:float=0.05, sigma:float|None=None,
                pd_ephemeris:pd.DataFrame|None=None, ephem_filepath:str|None=None, max_iter:int=20,
                weight_column:str="weight", time_column:str="unix_time", verbose=False) \
        -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes position using BITS and a classic FDE

    :param time_column:
    :param weight_column:
    :param max_iter:
    :param pd_gnss_raw: BITS raw dataframe
    :param alpha: Significance level
    :param sigma: Standard deviation of measurement noise set to None to use 1/weight² as sigma
    :param pd_ephemeris: BITS ephemeris dataframe
    :param ephem_filepath: Path of a rinex nav file
    :param verbose: set to True for verbose output
    :return: BITS pvt dataframe, BITS raw dataframe
    """
    # Add standard deviation
    if sigma is not None:
        pd_gnss_raw[weight_column] = 1/(sigma**2)

    # Compute position using bits' SPP
    pd_gnss_estimate, pd_gnss_raw = bits.spp.get_position_estimate(pd_gnss_raw, pd_ephemeris=pd_ephemeris,
                                                                   ephem_filepath=ephem_filepath, verbose=verbose)

    # Add unix_time for easier sorting
    pd_gnss_raw["unix_time"] = pd_gnss_raw["time"].apply(
        lambda gnss_timestamp: gnss_timestamp.pd_timestamp().timestamp())


    # Clean up
    pd_gnss_raw = pd_gnss_raw.sort_values("unix_time").reset_index(drop=True)

    # FDE
    pd_gnss_estimate, pd_gnss_raw = (
        fde.classic(pd_gnss_raw, bits.spp.get_approx_position_estimate, alpha=alpha, max_iter=max_iter,
                    steering_vector_column=("e_x", "e_y", "e_z"), weight_column=weight_column, time_column=time_column,
                    residuals_column="residuals_m", verbose=verbose))

    pd_gnss_estimate["unix_time"] = pd_gnss_estimate["time"].apply(
        lambda gnss_timestamp: gnss_timestamp.pd_timestamp().timestamp())

    # Compute HPL
    # Since the number of used constellations may change, we need to use window_hpl and recompute number_of_unknown at
    # each iteration.
    for timestamp, group in pd_gnss_raw.groupby("unix_time", sort=True):
        # Keep only valid estimates
        group = group[group["valid_estimate"] == True]

        # Get residuals
        residuals = group["residuals_m"].to_numpy().reshape(-1, 1)

        # Get weight matrix
        W = np.diag(group[weight_column])

        # Build G
        G = np.vstack([group[column].to_numpy() for column in ("e_x", "e_y", "e_z")])
        G = G.transpose()

        # Get number of unknowns
        number_of_unknown = 3 + len(pd_gnss_raw["gnss_id"].unique())

        try:
            cov = pd_gnss_estimate[pd_gnss_estimate["unix_time"] == timestamp][["cov_xx_rx_m", "cov_yy_rx_m", "cov_zz_rx_m"]].iloc[0]
            d_major = np.sqrt(cov.sum())
            noise, bias, protection = hpl.window_hpl(d_major, G, W, residuals, alpha, dof=number_of_unknown)

            pd_gnss_raw.loc[group.index, ["hpl_noise_m", "hpl_bias_m", "hpl_m"]] = float(noise.item()), float(bias.item()), float(protection.item())
        except Exception as e:
            txt = f" Could not compute HPL at timestamp {timestamp}: {e}"
            print(txt)

    # Add test results to PVT dataframe
    pd_gnss_estimate = pd.merge_asof(pd_gnss_estimate,  pd_gnss_raw[["unix_time", "hpl_noise_m", "hpl_bias_m", "hpl_m"]],
                                     on="unix_time", direction="nearest", tolerance=0.1)

    return pd_gnss_estimate, pd_gnss_raw

def subset_test_fde(pd_gnss_raw: pd.DataFrame, alpha:float=0.05, sigma:float|None=None,
                pd_ephemeris:pd.DataFrame|None=None, ephem_filepath:str|None=None,
                weight_column:str="weight", time_column:str="unix_time", verbose=False, max_depth:int|None=None) \
        -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes position using BITS and a Subset Test

    :param time_column:
    :param weight_column:
    :param pd_gnss_raw: BITS raw dataframe
    :param alpha: Significance level
    :param sigma: Standard deviation of measurement noise set to None to use 1/weight² as sigma
    :param pd_ephemeris: BITS ephemeris dataframe
    :param ephem_filepath: Path of a rinex nav file
    :param verbose: set to True for verbose output
    :param max_depth: Maximum allowed number of measurement size allowed, set to None for max
    :return: BITS pvt dataframe, BITS raw dataframe
    """
    # Add standard deviation
    if sigma is not None:
        pd_gnss_raw[weight_column] = 1/(sigma**2)

    # Compute position using bits' SPP
    pd_gnss_estimate, pd_gnss_raw = bits.spp.get_position_estimate(pd_gnss_raw, pd_ephemeris=pd_ephemeris,
                                                                   ephem_filepath=ephem_filepath, verbose=verbose)

    # Add unix_time for easier sorting
    pd_gnss_raw["unix_time"] = pd_gnss_raw["time"].apply(
        lambda gnss_timestamp: gnss_timestamp.pd_timestamp().timestamp())


    # Clean up
    pd_gnss_raw = pd_gnss_raw.sort_values("unix_time").reset_index(drop=True)

    # FDE
    pd_gnss_estimate, pd_gnss_raw = (
        fde.subset_test(pd_gnss_raw, bits.spp.get_approx_position_estimate, alpha=alpha, weight_column=weight_column,
                        time_column=time_column, residuals_column="residuals_m", verbose=verbose, max_depth=max_depth))

    pd_gnss_estimate["unix_time"] = pd_gnss_estimate["time"].apply(
        lambda gnss_timestamp: gnss_timestamp.pd_timestamp().timestamp())

    # Compute HPL
    # Since the number of used constellations may change, we need to use window_hpl and recompute number_of_unknown at
    # each iteration.
    for timestamp, group in pd_gnss_raw.groupby("unix_time", sort=True):
        # Keep only valid estimates
        group = group[group["valid_estimate"] == True]

        # Get residuals
        residuals = group["residuals_m"].to_numpy().reshape(-1, 1)

        # Get weight matrix
        W = np.diag(group[weight_column])

        # Build G
        G = np.vstack([group[column].to_numpy() for column in ("e_x", "e_y", "e_z")])
        G = G.transpose()

        # Get number of unknowns
        number_of_unknown = 3 + len(pd_gnss_raw["gnss_id"].unique())

        try:
            cov = pd_gnss_estimate[pd_gnss_estimate["unix_time"] == timestamp][["cov_xx_rx_m", "cov_yy_rx_m", "cov_zz_rx_m"]].iloc[0]
            d_major = np.sqrt(cov.sum())
            noise, bias, protection = hpl.window_hpl(d_major, G, W, residuals, alpha, dof=number_of_unknown)

            pd_gnss_raw.loc[group.index, ["hpl_noise_m", "hpl_bias_m", "hpl_m"]] = float(noise.item()), float(bias.item()), float(protection.item())
        except Exception as e:
            txt = f" Could not compute HPL at timestamp {timestamp}: {e}"
            print(txt)

    # Add test results to PVT dataframe
    pd_gnss_estimate = pd.merge_asof(pd_gnss_estimate,  pd_gnss_raw[["unix_time", "hpl_noise_m", "hpl_bias_m", "hpl_m"]],
                                     on="unix_time", direction="nearest", tolerance=0.1)

    return pd_gnss_estimate, pd_gnss_raw