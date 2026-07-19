import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
from typing import Callable
from tqdm import tqdm
from joblib import Parallel, delayed

from bits_fde.src import test

"""
based on: Ni Zhu. GNSS Propagation Channel Modeling in Constrained Environments: Contribution to the Improvement of the 
Geolocation Service Quality/. Engineering Sciences [physics]. Université de Lille, 2018. English. 
https://hal.science/tel-01959797v1
"""

def classic(gnss_pd:pd.DataFrame, positioning_func:Callable, sigma:float|None=None, alpha:float=0.05,
            max_iter:int=20, number_of_unknown:int|None=None, gnss_id_column:str="gnss_id",
            steering_vector_column:tuple=("e_x", "e_y", "e_z"), weight_column:str="weight",
            time_column:str="unix_time",  residuals_column:str="residuals_m", verbose:bool=False,
            *args, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs Classic method on a dataframe with multiple timestamps.

    One of the most basic FDE techniques is to simply use the GT to detect whether there is a fault. If it is the case,
    the measurement with the largest normalized residual will be excluded.

    :param gnss_pd: BITS raw dataframe
    :param positioning_func: Function to be used to estimate position
    :param sigma: Standard deviation of measurement noise set to None to use 1/weight² as sigma
    :param alpha: Significance level
    :param max_iter: Maximum allowed number of iterations
    :param number_of_unknown: Number of unknowns to solve
    :param gnss_id_column: Name of the gnss constellation ID column (used to determine number_of_unknown)
    :param steering_vector_column: Names of steering vectors columns
    :param weight_column: Name of the weight column
    :param time_column: Name of time column
    :param residuals_column: Name of pseudorange residuals column
    :param verbose: set to True for verbose output
    :param args: args to be given to positioning_func
    :param kwargs: kwargs to be given to positioning_func
    :return: BITS pvt dataframe, BITS raw dataframe
    """
    out_raw_pd = pd.DataFrame()
    out_estimate_pd = pd.DataFrame()

    groups = gnss_pd.groupby(time_column, sort=True)
    iterator = tqdm(groups, desc="Applying classic FDE") if verbose else groups

    for _, group in iterator:
        estimate_pd, raw_pd = (
            window_iterative_local_test(group, positioning_func, sigma=sigma, alpha=alpha, max_iter=max_iter,
                                        number_of_unknown=number_of_unknown, gnss_id_column=gnss_id_column,
                                        steering_vector_column=steering_vector_column, weight_column=weight_column,
                                        time_column=time_column,  residuals_column=residuals_column,
                                        redundancy_check=False, *args, **kwargs))

        out_raw_pd = pd.concat([out_raw_pd, raw_pd], axis=0)
        out_estimate_pd = pd.concat([out_estimate_pd, estimate_pd], axis=0)

    return out_estimate_pd, out_raw_pd


def subset_test(gnss_pd:pd.DataFrame, positioning_func:Callable, sigma:float|None=None, alpha:float=0.05,
                number_of_unknown:int|None=None, gnss_id_column:str="gnss_id", weight_column:str="weight",
                time_column:str="unix_time",  residuals_column:str="residuals_m", verbose:bool=False,
                max_depth:int|None=None, *args, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs Subset Test on a dataframe with multiple timestamps.

    The principle of the Subset Test (ST) is to perform Global Test (GT) several times while taking out one or several
    measurements at a time in order to find the right measurement set excluding the huge errors.

    After the initial failure of the GT with all the measurements, the ST will begin. The test statistics will be
    calculated for all the possible subsets that include n + 1 to m − 1 measurement, where n is the number of unknown to
    be estimated and m is the number of measurements. Then, the subset that has the smallest test statistic below the
    threshold, and at the same time, the largest number of measurement, will be chosen to calculate the position
    solution.

    :param gnss_pd: BITS raw dataframe
    :param positioning_func: Function to be used to estimate position
    :param sigma: Standard deviation of measurement noise set to None to use 1/weight² as sigma
    :param alpha: Significance level
    :param number_of_unknown: Number of unknowns to solve
    :param gnss_id_column: Name of the gnss constellation ID column (used to determine number_of_unknown)
    :param weight_column: Name of the weight column
    :param time_column: Name of time column
    :param residuals_column: Name of pseudorange residuals column
    :param verbose: set to True for verbose output
    :param max_depth: Maximum allowed number of measurement size allowed, set to None for max
    :param args: args to be given to positioning_func
    :param kwargs: kwargs to be given to positioning_func
    :return: BITS pvt dataframe, BITS raw dataframe
    """
    groups = gnss_pd.groupby(time_column, sort=True)

    results = Parallel(n_jobs=-1)(
        delayed(window_subset_test)(
            group, positioning_func, sigma=sigma, alpha=alpha,
            number_of_unknown=number_of_unknown, gnss_id_column=gnss_id_column,
            weight_column=weight_column, time_column=time_column,
            residuals_column=residuals_column, max_depth=max_depth, *args, **kwargs
        )
        for _, group in tqdm(groups, desc="Applying Subset Test", disable=not verbose)
    )

    estimate_list, raw_list = zip(*results) if results else ([], [])
    out_estimate_pd = pd.concat(estimate_list, axis=0) if estimate_list else pd.DataFrame()
    out_raw_pd = pd.concat(raw_list, axis=0) if raw_list else pd.DataFrame()

    return out_estimate_pd, out_raw_pd


def window_subset_test(window_gnss_pd:pd.DataFrame, positioning_func:Callable, sigma:float|None=None, alpha:float=0.05,
                       number_of_unknown:int|None=None, gnss_id_column:str="gnss_id", weight_column:str="weight",
                       time_column:str="unix_time",  residuals_column:str="residuals_m", max_depth:int|None=None, *args, **kwargs) \
        -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs Subset Test on a single timestamp.

    The principle of the Subset Test (ST) is to perform Global Test (GT) several times while taking out one or several
    measurements at a time in order to find the right measurement set excluding the huge errors.

    After the initial failure of the GT with all the measurements, the ST will begin. The test statistics will be
    calculated for all the possible subsets that include n + 1 to m − 1 measurement, where n is the number of unknown to
    be estimated and m is the number of measurements. Then, the subset that has the smallest test statistic below the
    threshold, and at the same time, the largest number of measurement, will be chosen to calculate the position
    solution.

    :param window_gnss_pd: BITS raw dataframe
    :param positioning_func: Function to be used to estimate position
    :param sigma: Standard deviation of measurement noise set to None to use 1/weight² as sigma
    :param alpha: Significance level
    :param number_of_unknown: Number of unknowns to solve
    :param gnss_id_column: Name of the gnss constellation ID column (used to determine number_of_unknown)
    :param weight_column: Name of the weight column
    :param time_column: Name of time column
    :param residuals_column: Name of pseudorange residuals column
    :param max_depth: Maximum allowed number of measurement size allowed, set to None for max
    :param args: args to be given to positioning_func
    :param kwargs: kwargs to be given to positioning_func
    :return: BITS pvt dataframe, BITS raw dataframe
    """
    # 1. Find every possible combinations
    n = len(window_gnss_pd)
    test_statistic_list = []
    estimate_list = []
    index_list = []
    found_valid_group = False
    if max_depth is not None and (n - max_depth) >= 1:
        min_size = n - max_depth
    else:
        min_size = 1
    # Start with the biggest group and iterate until a size with at least one valid group is found
    for size in range(n, min_size, -1):
        for combo_index in combinations(window_gnss_pd.index, size):
            if number_of_unknown is None:
                local_number_of_unknown = 3 + len(window_gnss_pd[gnss_id_column].unique())
            else:
                local_number_of_unknown = number_of_unknown

            if size < local_number_of_unknown + 1:
                continue

            sub_window = window_gnss_pd.loc[list(combo_index)]

            # 2. Compute position
            try:
                sub_estimate_pd, sub_window = positioning_func(sub_window, *args, **kwargs)
            except:
                continue

            # 3. Apply global test
            sub_window = test.global_test(sub_window, sigma=sigma, alpha=alpha,
                                              number_of_unknown=local_number_of_unknown, weight_column=weight_column,
                                              time_column=time_column, residuals_column=residuals_column)
            # Keep window if global test passed
            if sub_window["valid_estimate"].all():
                sub_estimate_pd["valid_estimate"] = True

                test_statistic_list.append(sub_window["test_statistic"].iloc[0])
                estimate_list.append(sub_estimate_pd)
                index_list.append(combo_index)
                found_valid_group = True
        if found_valid_group:
            break

    # if no valid group found, return initial estimate
    if not found_valid_group:
        # Compute position
        try:
            estimate_pd, window_gnss_pd = positioning_func(window_gnss_pd, *args, **kwargs)
        except:
            return pd.DataFrame(), pd.DataFrame()
        estimate_pd["valid_estimate"] = False
        window_gnss_pd["valid_estimate"] = False
        return estimate_pd, window_gnss_pd

    # 4. Keep group with best test statistic
    min_test_statistic = np.nanmin(test_statistic_list)
    best_group_index = [i for i, value in enumerate(test_statistic_list) if value == min_test_statistic][0]

    best_estimate_pd = estimate_list[best_group_index]
    best_group_window_index = index_list[best_group_index]

    # Update valid_estimate and test_statistic at kept SV lines
    window_gnss_pd.loc[best_group_window_index, "valid_estimate"] = True
    window_gnss_pd.loc[best_group_window_index, "test_statistic"] = float(min_test_statistic)

    return best_estimate_pd, window_gnss_pd

def iterative_local_test(gnss_pd:pd.DataFrame, positioning_func:Callable, sigma:float|None=None, alpha:float=0.05,
            max_iter:int=20, number_of_unknown:int|None=None, gnss_id_column:str="gnss_id",
            steering_vector_column:tuple=("e_x", "e_y", "e_z"), weight_column:str="weight",
            time_column:str="unix_time",  residuals_column:str="residuals_m", verbose:bool=False,
            *args, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs Iterative Local Test on a dataframe with multiple timestamps.

    The GT is used to detect the existence of fault. If the GT is failed and there is enough redundancy, then the LT
    will be performed individually for each measurement. The measurement with the largest test statistic which is also
    above the threshold will be examined with the redundancy check. If the redundancy check passes, the measurement with
    the largest test statistic will be excluded.

    :param gnss_pd: BITS raw dataframe
    :param positioning_func: Function to be used to estimate position
    :param sigma: Standard deviation of measurement noise set to None to use 1/weight² as sigma
    :param alpha: Significance level
    :param max_iter: Maximum allowed number of iterations
    :param number_of_unknown: Number of unknowns to solve
    :param gnss_id_column: Name of the gnss constellation ID column (used to determine number_of_unknown)
    :param steering_vector_column: Names of steering vectors columns
    :param weight_column: Name of the weight column
    :param time_column: Name of time column
    :param residuals_column: Name of pseudorange residuals column
    :param verbose: set to True for verbose output
    :param args: args to be given to positioning_func
    :param kwargs: kwargs to be given to positioning_func
    :return: BITS pvt dataframe, BITS raw dataframe
    """
    out_raw_pd = pd.DataFrame()
    out_estimate_pd = pd.DataFrame()

    groups = gnss_pd.groupby(time_column, sort=True)
    iterator = tqdm(groups, desc="Applying Iterative Local Test FDE") if verbose else groups

    for _, group in iterator:
        estimate_pd, raw_pd = (
            window_iterative_local_test(group, positioning_func, sigma=sigma, alpha=alpha,  max_iter=max_iter,
                                        number_of_unknown=number_of_unknown, gnss_id_column=gnss_id_column,
                                        steering_vector_column=steering_vector_column, weight_column=weight_column,
                                        time_column=time_column,  residuals_column=residuals_column, *args, **kwargs))

        out_raw_pd = pd.concat([out_raw_pd, raw_pd], axis=0)
        out_estimate_pd = pd.concat([out_estimate_pd, estimate_pd], axis=0)

    return out_estimate_pd, out_raw_pd


def window_iterative_local_test(window_gnss_pd:pd.DataFrame, positioning_func:Callable, sigma:float|None=None,
                                alpha:float=0.05, max_iter:int=20, number_of_unknown:int|None=None,
                                gnss_id_column:str="gnss_id", steering_vector_column:tuple=("e_x", "e_y", "e_z"),
                                weight_column:str="weight", time_column:str="unix_time",
                                residuals_column:str="residuals_m", redundancy_check:bool=True, *args, **kwargs) \
        -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs Iterative Local Test on a single timestamp.

    The GT is used to detect the existence of fault. If the GT is failed and there is enough redundancy, then the LT
    will be performed individually for each measurement. The measurement with the largest test statistic which is also
    above the threshold will be examined with the redundancy check. If the redundancy check passes, the measurement with
    the largest test statistic will be excluded.

    :param window_gnss_pd: BITS raw dataframe
    :param positioning_func: Function to be used to estimate position
    :param sigma: Standard deviation of measurement noise set to None to use 1/weight² as sigma
    :param alpha: Significance level
    :param max_iter: Maximum allowed number of iterations
    :param number_of_unknown: Number of unknowns to solve
    :param gnss_id_column: Name of the gnss constellation ID column (used to determine number_of_unknown)
    :param steering_vector_column: Names of steering vectors columns
    :param weight_column: Name of the weight column
    :param time_column: Name of time column
    :param residuals_column: Name of pseudorange residuals column
    :param redundancy_check: Set to False for Classic FDE
    :param args: args to be given to positioning_func
    :param kwargs: kwargs to be given to positioning_func
    :return: BITS pvt dataframe, BITS raw dataframe
    """
    out_pd = pd.DataFrame()
    estimate_pd = pd.DataFrame()
    for index in range(max_iter):
        # 1. Compute position
        try:
            estimate_pd, window_gnss_pd = positioning_func(window_gnss_pd, *args, **kwargs)
        except:
            break
        estimate_pd["valid_estimate"] = False
        # Check if FDE is finished
        remaining_measurements_count = len(window_gnss_pd)
        if remaining_measurements_count == 0:
            break

        # 2. Apply global test
        if number_of_unknown is None:
            local_number_of_unknown = 3 + len(window_gnss_pd[gnss_id_column].unique())
        else:
            local_number_of_unknown = number_of_unknown
        window_gnss_pd = test.global_test(window_gnss_pd, sigma=sigma, alpha=alpha,
                                          number_of_unknown=local_number_of_unknown, weight_column=weight_column,
                                          time_column=time_column,  residuals_column=residuals_column)
        # Check if FDE is finished
        if window_gnss_pd["valid_estimate"].all():
            estimate_pd["valid_estimate"] = True
            break

        # 3. Apply local test
        window_gnss_pd = test.local_test(window_gnss_pd, sigma=sigma, alpha=alpha, weight_column=weight_column,
                                         time_column=time_column,  residuals_column=residuals_column,
                                         steering_vector_column=steering_vector_column)
        # Exclude max normalized residual
        idx = window_gnss_pd["test_statistic"].idxmax()
        # Check redundancy and LT result
        if window_gnss_pd.loc[[idx], "valid_estimate"].iloc[0] is not False and redundancy_check:
            break
        out_pd = pd.concat([out_pd, window_gnss_pd.loc[[idx]]], ignore_index=True)
        window_gnss_pd = window_gnss_pd.drop(idx)
        # Check if FDE is finished
        remaining_measurements_count = len(window_gnss_pd)
        if remaining_measurements_count == 0:
            break

    out_pd = pd.concat([out_pd, window_gnss_pd], ignore_index=True)

    return estimate_pd, out_pd


def forward_backward(gnss_pd:pd.DataFrame, positioning_func:Callable, sigma:float|None=None, alpha:float=0.05,
                     max_iter:int=20, number_of_unknown:int|None=None, gnss_id_column:str="gnss_id",
                     steering_vector_column:tuple=("e_x", "e_y", "e_z"), weight_column:str="weight",
                     time_column:str="unix_time",  residuals_column:str="residuals_m", verbose:bool=False,
                     *args, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs Forward-Backward on a dataframe with multiple timestamps.

    There are two main parts in the FB testing algorithm: the forward part and the backward part. In the forward part,
    the GT is firstly carried out to check the measurement consistency. If the GT fails, the LT will be performed in
    order to identify and exclude the outliers. This forward part will be conducted recursively until no more erroneous
    measurements are detected and the solution is declared reliable or unreliable. If the GT in the forward part is
    passed and more than one measurement is excluded, then the backward part will begin, where the excluded measurements
    will be reintroduced into the measurement set.

    :param gnss_pd: BITS raw dataframe
    :param positioning_func: Function to be used to estimate position
    :param sigma: Standard deviation of measurement noise set to None to use 1/weight² as sigma
    :param alpha: Significance level
    :param max_iter: Maximum allowed number of iterations
    :param number_of_unknown: Number of unknowns to solve
    :param gnss_id_column: Name of the gnss constellation ID column (used to determine number_of_unknown)
    :param steering_vector_column: Names of steering vectors columns
    :param weight_column: Name of the weight column
    :param time_column: Name of time column
    :param residuals_column: Name of pseudorange residuals column
    :param verbose: set to True for verbose output
    :param args: args to be given to positioning_func
    :param kwargs: kwargs to be given to positioning_func
    :return: BITS pvt dataframe, BITS raw dataframe
    """
    out_raw_pd = pd.DataFrame()
    out_estimate_pd = pd.DataFrame()

    groups = gnss_pd.groupby(time_column, sort=True)
    iterator = tqdm(groups, desc="Applying Forward-Backward FDE") if verbose else groups

    for _, group in iterator:
        estimate_pd, raw_pd = (
            window_forward_backward(group, positioning_func, sigma=sigma, alpha=alpha,  max_iter=max_iter,
                                    number_of_unknown=number_of_unknown, gnss_id_column=gnss_id_column,
                                    steering_vector_column=steering_vector_column, weight_column=weight_column,
                                    time_column=time_column,  residuals_column=residuals_column, *args, **kwargs))

        out_raw_pd = pd.concat([out_raw_pd, raw_pd], axis=0)
        out_estimate_pd = pd.concat([out_estimate_pd, estimate_pd], axis=0)

    return out_estimate_pd, out_raw_pd


def window_forward_backward(window_gnss_pd:pd.DataFrame, positioning_func:Callable, sigma:float|None=None,
                            alpha:float=0.05, max_iter:int=20, number_of_unknown:int|None=None,
                            gnss_id_column:str="gnss_id", steering_vector_column:tuple=("e_x", "e_y", "e_z"),
                            weight_column:str="weight", time_column:str="unix_time",
                            residuals_column:str="residuals_m", *args, **kwargs) \
        -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs Forward-Backward on a single timestamp.

    There are two main parts in the FB testing algorithm: the forward part and the backward part. In the forward part,
    the GT is firstly carried out to check the measurement consistency. If the GT fails, the LT will be performed in
    order to identify and exclude the outliers. This forward part will be conducted recursively until no more erroneous
    measurements are detected and the solution is declared reliable or unreliable. If the GT in the forward part is
    passed and more than one measurement is excluded, then the backward part will begin, where the excluded measurements
    will be reintroduced into the measurement set.

    :param window_gnss_pd: BITS raw dataframe
    :param positioning_func: Function to be used to estimate position
    :param sigma: Standard deviation of measurement noise set to None to use 1/weight² as sigma
    :param alpha: Significance level
    :param max_iter: Maximum allowed number of iterations
    :param number_of_unknown: Number of unknowns to solve
    :param gnss_id_column: Name of the gnss constellation ID column (used to determine number_of_unknown)
    :param steering_vector_column: Names of steering vectors columns
    :param weight_column: Name of the weight column
    :param time_column: Name of time column
    :param residuals_column: Name of pseudorange residuals column
    :param args: args to be given to positioning_func
    :param kwargs: kwargs to be given to positioning_func
    :return: BITS pvt dataframe, BITS raw dataframe
    """
    # 1 Forward: iterative local test
    pd_gnss_estimate, window_gnss_pd = (
        window_iterative_local_test(window_gnss_pd, positioning_func, sigma=sigma, alpha=alpha, max_iter=max_iter,
                                    number_of_unknown=number_of_unknown, gnss_id_column=gnss_id_column,
                                    steering_vector_column=steering_vector_column, weight_column=weight_column,
                                    time_column=time_column, residuals_column=residuals_column, *args, **kwargs))

    # Check if Iterative Local Test passed
    if not window_gnss_pd["valid_estimate"].any():
        return pd_gnss_estimate, window_gnss_pd

    # 2 Backward: excluded measurements reintroduction
    # A. Find valid/excluded measurements
    excluded_group = window_gnss_pd[window_gnss_pd["valid_estimate"] == False]
    valid_group = window_gnss_pd[window_gnss_pd["valid_estimate"] == True]

    # B find every possible combinations
    excluded_size = len(excluded_group)
    test_statistic_list = []
    estimate_list = []
    index_list = []
    found_valid_group = False

    for  size in range(excluded_size, 0, -1):
        for combo_index in combinations(excluded_group.index, size):
            sub_group = pd.concat([valid_group, window_gnss_pd.loc[list(combo_index)]])
            # C. Compute position
            try:
                sub_estimate_pd, sub_group = positioning_func(sub_group, *args, **kwargs)
            except:
                continue

            # D. Apply global test
            if number_of_unknown is None:
                local_number_of_unknown = 3 + len(sub_group[gnss_id_column].unique())
            else:
                local_number_of_unknown = number_of_unknown
            sub_group = test.global_test(sub_group, sigma=sigma, alpha=alpha, number_of_unknown=local_number_of_unknown,
                                         weight_column=weight_column, time_column=time_column,
                                         residuals_column=residuals_column)

            # Keep window if global test passed
            if window_gnss_pd["valid_estimate"].all():
                sub_estimate_pd["valid_estimate"] = True
                test_statistic_list.append(sub_group["test_statistic"].iloc[0])
                estimate_list.append(sub_estimate_pd)
                index_list.append(combo_index)
                found_valid_group = True

    # if no valid group found, return forward estimate
    if not found_valid_group:
        return pd_gnss_estimate, window_gnss_pd

    # E. Keep group with best test statistic
    min_test_statistic = np.nanmin(test_statistic_list)
    best_group_index = [i for i, value in enumerate(test_statistic_list) if value == min_test_statistic][0]

    best_estimate_pd = estimate_list[best_group_index]
    best_group_window_index = index_list[best_group_index]

    # Update valid_estimate and test_statistic at kept SV lines
    window_gnss_pd.loc[best_group_window_index, "valid_estimate"] = True
    window_gnss_pd.loc[best_group_window_index, "test_statistic"] = float(min_test_statistic)

    return best_estimate_pd, window_gnss_pd


def danish(gnss_pd: pd.DataFrame, positioning_func:Callable, positioning_func_args: tuple = (), sigma: float = 0.3,
                    alpha: float = 0.05, num_min_meas: int = 5, max_iter: int = 10,
                    steering_vector_column_name: tuple = ("steering_vector_x", "steering_vector_y", "steering_vector_z"),
                    weight_column_name: str = "weight", residuals_column_name: str = "residuals") -> pd.DataFrame:
    """
    an iteratively reweighting procedure on the pre-estimated measurement variance will be carried out according to the
    ratio between local test statistic wi and the local threshold th. What should be highlighted is that there is no
    exclusion step in this method, which can well keep the initial satellite geometry. If the algorithm cannot converge
    after several iteration (here, we fix it as 10), the solution will be declared as unreliable.

    Args:
        gnss_pd:
        positioning_func:
        positioning_func_args:
        sigma:
        alpha:
        num_min_meas:
        max_iter:
        steering_vector_column_name:
        weight_column_name:
        residuals_column_name:

    Returns:

    """
    sub_pd = gnss_pd.copy()
    sub_pd["chi_2_passed"] = False
    out_pd = pd.DataFrame()
    for index in range(max_iter):
        print(f"\n\nIteration {index}")

        # 1. Compute position
        sub_pd = positioning_func(sub_pd, *positioning_func_args)
        # get rid of non-valid baselines
        mask = sub_pd["baseline"].isna()
        out_pd = pd.concat([out_pd, sub_pd[mask]], ignore_index=True)
        sub_pd = sub_pd[~mask]
        # Check if FDE is finished
        if len(sub_pd) == 0:
            break

        # 2. Apply global test
        sub_pd = global_test(sub_pd, sigma=sigma, alpha=alpha, num_min_meas=num_min_meas)
        # Get rid of baseline estimates that passes global test
        try:
            mask = (sub_pd["chi_2_passed"] == True) | (sub_pd["chi_2_passed"].isna())
        except:
            pass
        out_pd = pd.concat([out_pd, sub_pd[mask]], ignore_index=True)
        sub_pd = sub_pd[~mask]
        # Check if FDE is finished
        if len(sub_pd) == 0:
            break

        # 3 Get local test statistics
        sub_pd = local_test(sub_pd, sigma=sigma, alpha=alpha, num_min_meas=num_min_meas,
                            steering_vector_column_name=steering_vector_column_name,
                            weight_column_name=weight_column_name, residuals_column_name=residuals_column_name)
        chi2_threshold = stats.chi2.ppf(1 - alpha, df=1)

        # 4 Reweighting
        mask = sub_pd["normalized_residual"] > chi2_threshold
        sub_pd.loc[mask, weight_column_name] = sub_pd.loc[mask, weight_column_name] / np.exp(sub_pd.loc[mask, "normalized_residual"]/chi2_threshold)

    out_pd = pd.concat([out_pd, sub_pd], ignore_index=True)
    out_pd.sort_values("unix_time", inplace=True)
    out_pd.reset_index(drop=True, inplace=True)
    return out_pd


def irls(gnss_pd: pd.DataFrame, positioning_func:Callable, a:float=1.345, alpha:float=0.05, max_iter:int= 20,
         delta:float=1e-7,  steering_vector_column:tuple=("e_x", "e_y", "e_z"),
         clock_bias_vector_column:tuple|None=("e_bbei", "e_bgal", "e_bglo", "e_bgps"),
         estimate_column:tuple=("x_rx_m", "y_rx_m", "z_rx_m"), weight_column:str="weight",
         residuals_column:str="residuals_m", time_column:str="unix_time",
         covariance_column:tuple=("cov_xx_rx_m", "cov_yy_rx_m", "cov_zz_rx_m", "cov_bb1_rx_m", "cov_bb2_rx_m",
                                  "cov_bb3_rx_m", "cov_bb4_rx_m"), verbose:bool=False, *args, **kwargs) \
        -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Iterative Reweighted Least Square (IRLS)
    ref: D. Medina et. al. Robust Statistics for GNSS Positioning under Harsh Conditions: A Useful Tool ?

    Cov(x̂) = σ² · (Hᵀ W H)⁻¹ · (Hᵀ W² H) · (Hᵀ W H)⁻¹

    :param gnss_pd:
    :param positioning_func:
    :param a:
    :param alpha:
    :param max_iter:
    :param delta:
    :param steering_vector_column:
    :param clock_bias_vector_column:
    :param estimate_column:
    :param weight_column:
    :param residuals_column:
    :param time_column:
    :param covariance_column:
    :param verbose:
    :param args:
    :param kwargs:
    :return:
    """
    out_raw_list = []
    out_estimate_list = []

    groups = gnss_pd.groupby(time_column, sort=True)
    iterator = tqdm(groups, desc="Applying Iterative Reweighted Least Square") if verbose else groups

    for _, group in iterator:
        estimate_pd, raw_pd = (
            window_irls(group, positioning_func, a=a, alpha=alpha,  max_iter=max_iter, delta=delta,
                        steering_vector_column=steering_vector_column,
                        clock_bias_vector_column=clock_bias_vector_column,  estimate_column=estimate_column,
                        weight_column=weight_column, residuals_column=residuals_column, time_column=time_column,
                        covariance_column=covariance_column, *args, **kwargs))

        out_raw_list.append(raw_pd)
        out_estimate_list.append(estimate_pd)

    out_raw_pd = pd.concat(out_raw_list, axis=0, ignore_index=True)
    out_estimate_pd = pd.concat(out_estimate_list, axis=0, ignore_index=True)

    return out_estimate_pd, out_raw_pd


def window_irls(window_gnss_pd: pd.DataFrame, positioning_func:Callable, a:float=1.345, alpha:float=0.05,
                max_iter:int= 20, delta:float=1e-7, steering_vector_column:tuple=("e_x", "e_y", "e_z"),
                clock_bias_vector_column:tuple|None=("e_bbei", "e_bgal", "e_bglo", "e_bgps"),
                estimate_column:tuple=("x_rx_m", "y_rx_m", "z_rx_m"), weight_column:str="weight",
                residuals_column:str="residuals_m", time_column:str="unix_time",
                covariance_column:tuple=("cov_xx_rx_m", "cov_yy_rx_m", "cov_zz_rx_m", "cov_bb1_rx_m", "cov_bb2_rx_m",
                                         "cov_bb3_rx_m", "cov_bb4_rx_m"), *args, **kwargs) \
        -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Iterative Reweighted Least Square (IRLS)
    ref: D. Medina et. al. Robust Statistics for GNSS Positioning under Harsh Conditions: A Useful Tool ?

    Cov(x̂) = σ² · (Hᵀ W H)⁻¹ · (Hᵀ W² H) · (Hᵀ W H)⁻¹

    :param window_gnss_pd:
    :param positioning_func:
    :param a:
    :param alpha:
    :param max_iter:
    :param delta:
    :param steering_vector_column:
    :param clock_bias_vector_column:
    :param estimate_column:
    :param weight_column:
    :param residuals_column:
    :param time_column:
    :param covariance_column:
    :param args:
    :param kwargs:
    :return:
    """
    last_estimate = None
    converged = False
    for iteration in range(max_iter):
        # 1. Compute position
        try:
            estimate_pd, window_gnss_pd = positioning_func(window_gnss_pd, *args, **kwargs)
        except:
            return pd.DataFrame(), pd.DataFrame()
        estimate = np.vstack([estimate_pd[column].to_numpy() for column in estimate_column])

        if pd.isna(estimate).any():
            break

        # 2. Check convergence
        if last_estimate is not None:
            converged = float(np.linalg.norm(estimate - last_estimate)) < delta
        last_estimate = estimate
        if converged:
            break

        # 3. Update scale
        residuals = window_gnss_pd[residuals_column].to_numpy()
        sigma = mad(residuals)

        # 4. Update weights
        normed_residuals = residuals / sigma
        # Weight = loss_function/normed_residuals; if normed_residuals==0 -> weight = 1 (best possible weight)
        weight_matrix = np.divide(
            huber_psi(normed_residuals, a),
            normed_residuals,
            out=np.ones_like(normed_residuals, dtype=float),
            where=(normed_residuals != 0)
        )
        window_gnss_pd[weight_column] = weight_matrix

    if pd.isna(estimate).any():
        estimate_pd["valid_estimate"] = False
        window_gnss_pd["valid_estimate"] = False
        return estimate_pd, window_gnss_pd

    # 5. Compute covariance
    # Build geometry matrix
    G = np.vstack([window_gnss_pd[column].to_numpy() for column in steering_vector_column])
    found_clock_bias_vector = False
    if clock_bias_vector_column is not None:
        for column in clock_bias_vector_column:
            if column in window_gnss_pd.columns:
                found_clock_bias_vector = True
                G = np.vstack([G, window_gnss_pd[column].to_numpy()])
    if not found_clock_bias_vector:
        G = np.vstack([G, np.ones((1, G.shape[1]))])
    G = G.transpose()

    # Build weights
    W = np.diag(window_gnss_pd[weight_column])

    sandwich_bread = np.linalg.inv(G.transpose() @ W @ G)
    sandwich_vegan_ham = (G.transpose() @ W ** 2 @ G)  # I'm a vegetarian
    cov = sigma ** 2 * sandwich_bread @ sandwich_vegan_ham @ sandwich_bread

    number_of_unknowns = G.shape[1]
    for i in range(number_of_unknowns):
        window_gnss_pd[covariance_column[i]] = float(cov[i][i])

    window_gnss_pd[weight_column] = window_gnss_pd[weight_column] / sigma**2

    # 6. Perform global test
    window_gnss_pd = test.global_test(window_gnss_pd, alpha=alpha, number_of_unknown=number_of_unknowns,
                                 weight_column=weight_column, time_column=time_column,
                                 residuals_column=residuals_column)
    estimate_pd["valid_estimate"] = window_gnss_pd["valid_estimate"].iloc[0]

    return estimate_pd, window_gnss_pd

def mad(x: np.ndarray, c_m:float=1.4826) -> float:
    """
    Median Absolute Deviation (MAD) used as standard deviation estimator

    σ̂_MAD(x) = c_m * Med(|x - Med(x)|)

    with c_m = 1.4826  for coherence with Gaussian distribution

    :param x: Normalized residuals
    :param c_m:
    :return:
    """
    x = np.asarray(x, dtype=float).ravel()
    return c_m * np.median(np.abs(x - np.median(x)))

def huber_psi(x: np.ndarray, a: float=1.345) -> np.ndarray:
    """
    Huber's loss function

    :param x: Normalized residuals
    :param a: Huber threshold
    :return:
    """
    return np.where(np.abs(x) <= a, x, a * np.sign(x))


