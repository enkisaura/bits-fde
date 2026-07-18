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
        estimate_pd, raw_pd = window_classic(group, positioning_func, sigma=sigma, alpha=alpha, max_iter=max_iter,
                                             number_of_unknown=number_of_unknown, gnss_id_column=gnss_id_column,
                                             steering_vector_column=steering_vector_column, weight_column=weight_column,
                                             time_column=time_column,  residuals_column=residuals_column,
                                             *args, **kwargs)

        out_raw_pd = pd.concat([out_raw_pd, raw_pd], axis=0)
        out_estimate_pd = pd.concat([out_estimate_pd, estimate_pd], axis=0)

    return out_estimate_pd, out_raw_pd



def window_classic(window_gnss_pd:pd.DataFrame, positioning_func:Callable, sigma:float|None=None, alpha:float=0.05,
                   max_iter:int=20, number_of_unknown:int|None=None, gnss_id_column:str="gnss_id",
                   steering_vector_column:tuple=("e_x", "e_y", "e_z"), weight_column:str="weight",
                   time_column:str="unix_time",  residuals_column:str="residuals_m", *args, **kwargs) \
        -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs Classic method on a single timestamp.

    One of the most basic FDE techniques is to simply use the GT to detect whether there is a fault. If it is the case,
    the measurement with the largest normalized residual will be excluded.

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
        out_pd = pd.concat([out_pd, window_gnss_pd.loc[[idx]]], ignore_index=True)
        window_gnss_pd = window_gnss_pd.drop(idx)
        # Check if FDE is finished
        remaining_measurements_count = len(window_gnss_pd)
        if remaining_measurements_count == 0:
            break

    out_pd = pd.concat([out_pd, window_gnss_pd], ignore_index=True)

    return estimate_pd, out_pd

def old_subset_test(gnss_pd:pd.DataFrame, positioning_func:Callable, sigma:float|None=None, alpha:float=0.05,
                number_of_unknown:int|None=None, gnss_id_column:str="gnss_id", weight_column:str="weight",
                time_column:str="unix_time",  residuals_column:str="residuals_m", verbose:bool=False, *args, **kwargs) \
        -> tuple[pd.DataFrame, pd.DataFrame]:
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
    :param args: args to be given to positioning_func
    :param kwargs: kwargs to be given to positioning_func
    :return: BITS pvt dataframe, BITS raw dataframe
    """
    out_raw_pd = pd.DataFrame()
    out_estimate_pd = pd.DataFrame()

    groups = gnss_pd.groupby(time_column, sort=True)
    iterator = tqdm(groups, desc="Applying Subset Test") if verbose else groups

    for _, group in iterator:
        estimate_pd, raw_pd = window_subset_test(group, positioning_func, sigma=sigma, alpha=alpha,
                                                 number_of_unknown=number_of_unknown, gnss_id_column=gnss_id_column,
                                                 weight_column=weight_column, time_column=time_column,
                                                 residuals_column=residuals_column, *args, **kwargs)

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


def sequential_local_test(gnss_pd: pd.DataFrame, positioning_func:Callable, positioning_func_args: tuple = (), sigma: float = 0.3,
                    alpha: float = 0.05, num_min_meas: int = 5, max_iter: int = 20,
                    steering_vector_column_name: tuple = ("steering_vector_x", "steering_vector_y", "steering_vector_z"),
                    weight_column_name: str = "weight", residuals_column_name: str = "residuals") -> pd.DataFrame:
    """
    each measurement will be examined by LT. In each iteration, the measurement with the biggest local test statistic
    exceeding the threshold will be eliminated as outlier. The procedure stops until no outlier exists or lack of
    redundancy

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
    sub_pd["exclude_measurement"] = False
    out_pd = pd.DataFrame()
    print("1", len(sub_pd))

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


        # 2. Exclude largest test statistics
        sub_pd = local_test(sub_pd, sigma=sigma, alpha=alpha, num_min_meas=num_min_meas,
                            steering_vector_column_name=steering_vector_column_name,
                            weight_column_name=weight_column_name, residuals_column_name=residuals_column_name)

        # Exclude baseline estimate that passes global test
        mask = (sub_pd["chi_2_passed"] == True) | (sub_pd["chi_2_passed"].isna())
        out_pd = pd.concat([out_pd, sub_pd[mask]], ignore_index=True)
        sub_pd = sub_pd[~mask]

        # Get rid of largest normalized residual
        idx_max_norm_res = sub_pd.groupby("unix_time")["normalized_residual"].idxmax()
        mask = sub_pd.index.isin(idx_max_norm_res)
        sub_pd.loc[mask, "exclude_measurement"] = True
        out_pd = pd.concat([out_pd, sub_pd[mask]], ignore_index=True)
        sub_pd = sub_pd[~mask]

        # Check if FDE is finished
        print("4", len(sub_pd))
        if len(sub_pd) == 0:
            break


    out_pd.sort_values("unix_time", inplace=True)
    out_pd.reset_index(drop=True, inplace=True)

    return out_pd

def forward_backward(gnss_pd: pd.DataFrame, positioning_func:Callable, window_positioning_func:Callable,
                     positioning_func_args: tuple = (), window_positioning_func_args: tuple = (), sigma: float = 0.3,
                     alpha: float = 0.05, num_min_meas: int = 5, max_iter: int = 20,
                     steering_vector_column_name: tuple = ("steering_vector_x", "steering_vector_y", "steering_vector_z"),
                     weight_column_name: str = "weight", residuals_column_name: str = "residuals") -> pd.DataFrame:
    """
    the forward loop will be conducted as the sequential LT described previously. Then in the backward loop, the
    eliminated satellites in the forward loop will be reintroduced with all the possible combination until the optimal
    measurement set is found. The main advantage of the FB technique is, on the one hand, to avoid the erroneous
    rejection of a good measurement since a huge measurement error can sometimes distribute and hide in other
    measurements’ residuals due to special satellite geometry; on the other hand, the effect of  re-introduction of
    previous excluded satellite can enhance the satellite geometry which is usually poor in urban canyon.

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
    # 1 Apply sequential local test
    gnss_pd = sequential_local_test(gnss_pd, positioning_func=positioning_func,
                                    positioning_func_args=positioning_func_args, sigma=sigma, alpha=alpha,
                                    num_min_meas=num_min_meas, max_iter=max_iter,
                                    steering_vector_column_name=steering_vector_column_name,
                                    weight_column_name=weight_column_name, residuals_column_name=residuals_column_name)

    # 2 Add satellites back
    out_pd = gnss_pd.copy()
    # Iterate over each timestamp
    tqdm_desc = "Computing position using forward/backward testing"
    for _, group in tqdm(gnss_pd.groupby("unix_time"), total=len(gnss_pd["unix_time"].unique()), desc=tqdm_desc):
        excluded_group = group[group["exclude_measurement"]==True]
        kept_group = group[group["exclude_measurement"]==False]
        if len(excluded_group) == 0:
            out_pd = pd.concat([out_pd, kept_group[kept_group["chi_2_passed"]==True]], ignore_index=True)
            continue

        excluded_size = len(excluded_group)
        test_statistic_list = []
        size_list = []
        group_list = []
        for size in range(0, excluded_size+1):
            for combo_index in combinations(excluded_group.index, size):
                sub_group = pd.concat([kept_group, group.loc[list(combo_index)]])
    
                # 2. Compute position for each combination
                sub_group = window_positioning_func(sub_group, *window_positioning_func_args)
    
                # 3. Apply global test
                chi2_stat = np.sum((sub_group[residuals_column_name] / sigma) ** 2)  # Compute test statistic
                chi2_threshold = stats.chi2.ppf(1 - alpha, df=len(sub_group))  # Compute threshold
                passed = chi2_stat < chi2_threshold
    
                # 4. Keep only groups that passes GT
                if passed:
                    test_statistic_list.append(chi2_stat)
                    size_list.append(size)
                    group_list.append(sub_group)

        if len(test_statistic_list) > 0:
            # 3. Keep group with the largest number of measurements
            max_meas = np.nanmax(size_list)
            max_meas_index = [i for i, value in enumerate(size_list) if value == max_meas]

            # 4. Keep group with best test statistic
            kept_test_statistic_list = [test_statistic_list[i] for i in max_meas_index]
            min_test_statistic = np.nanmin(kept_test_statistic_list)
            best_sv_statistic_index = [i for i, value in enumerate(kept_test_statistic_list) if value == min_test_statistic]
            best_group_index = max_meas_index[best_sv_statistic_index[0]]

            best_group = group_list[best_group_index]
            best_group["test_statistic"] = min_test_statistic
            out_pd = pd.concat([out_pd, best_group], ignore_index=True)

    out_pd.sort_values("unix_time", inplace=True)
    out_pd.reset_index(drop=True, inplace=True)
    return out_pd
    


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


def irls(gnss_pd: pd.DataFrame, window_positioning_func:Callable, window_positioning_func_args: tuple = (),
         a:float=1.345, max_iter:int= 50, delta:float=1e-7) -> pd.DataFrame:
    """
    Iterative Reweighted Least Square (IRLS)
    ref: D. Medina et. al. Robust Statistics for GNSS Positioning under Harsh Conditions: A Useful Tool ?

    Cov(x̂) = σ² · (Hᵀ W H)⁻¹ · (Hᵀ W² H) · (Hᵀ W H)⁻¹
    """
    out_pd = pd.DataFrame()

    # Iterate over each timestamp
    tqdm_desc = "Computing position using IRLS"
    for _, sub_pd in tqdm(gnss_pd.groupby("unix_time"), total=len(gnss_pd["unix_time"].unique()), desc=tqdm_desc):
            sub_pd = window_irls(sub_pd, window_positioning_func,
                                 window_positioning_func_args=window_positioning_func_args,
                                 a=a, max_iter=max_iter, delta=delta)

            # Save result
            out_pd = pd.concat([out_pd, sub_pd], ignore_index=True)

    out_pd = global_test(out_pd, alpha = 0.05, num_min_meas = 5, weight_column = "weight")
    out_pd.sort_values("unix_time", inplace=True)
    out_pd.reset_index(drop=True, inplace=True)
    return out_pd


def window_irls(window: pd.DataFrame, window_positioning_func:Callable, window_positioning_func_args: tuple = (),
                a:float=1.345, max_iter:int= 50, delta:float=1e-7) -> pd.DataFrame:
    last_estimate = None
    converged = False
    for iteration in range(max_iter):
        # 1. Estimate position
        window = window_positioning_func(window, *window_positioning_func_args)
        estimate = window[["baseline_x", "baseline_y", "baseline_z"]].iloc[0].to_numpy()
        if pd.isna(estimate).any():
            break

        # 2. Check convergence
        if last_estimate is not None:
            converged = float(np.linalg.norm(estimate - last_estimate)) < delta
        last_estimate = estimate
        if converged:
            break

        # 3. Update scale
        residuals = window["residuals"].to_numpy()
        sigma = mad(residuals)

        # 4. Update weights
        normed_residuals = residuals / sigma
        window["weight"] = huber_psi(normed_residuals, a) / normed_residuals

    if not pd.isna(estimate).any():
        # Compute covariance matrix
        # Build geometry matrix
        ex = window["steering_vector_x_rx1"].to_numpy()
        ey = window["steering_vector_y_rx1"].to_numpy()
        ez = window["steering_vector_z_rx1"].to_numpy()
        H = np.vstack((ex, ey, ez, np.ones_like(ex))).transpose()

        # Build weights
        W = np.diag(window["weight"])

        sandwich_bread = np.linalg.inv(H.transpose() @ W @ H)
        sandwich_vegan_ham = (H.transpose() @ W ** 2 @ H)  # I'm a vegetarian
        cov = sigma ** 2 * sandwich_bread @ sandwich_vegan_ham @ sandwich_bread

        window["sigma_0"] = sigma
        window["covariance_x"] = float(cov[0][0])
        window["covariance_y"] = float(cov[1][1])
        window["covariance_z"] = float(cov[2][2])
        window["covariance_b"] = float(cov[3][3])
        window["uncertainty"] = float(np.sqrt(np.trace(cov[:3, :3])))

        window["weight"] = window["weight"] / sigma**2
        window["converged"] = converged

        return window

def mad(x: np.ndarray, c_m:float=1.4826) -> float:
    """
    Estimation robuste de l'écart-type par la MAD normalisée.

        σ̂_MAD(x) = c_m * Med(|x - Med(x)|)

    avec c_m = 1.4826  pour cohérence avec σ sous gaussienne.

    Paramètres
    ----------
    x : (n,)  vecteur d'observations

    Retourne
    --------
    σ̂_MAD : float  estimation robuste de σ
    """
    x = np.asarray(x, dtype=float).ravel()
    return c_m * np.median(np.abs(x - np.median(x)))

def huber_psi(x: np.ndarray, a: float=1.345) -> np.ndarray:
    """
    Fonction d'influence de Huber ψ_a^H(x).

        ψ_a^H(x) = x           si |x| ≤ a
                   a * sign(x) si |x| > a

    Paramètres
    ----------
    x : (n,)  résidus normalisés
    a : float  seuil de Huber (typiquement 1.345 → 95% efficacité)

    Retourne
    --------
    psi : (n,)
    """
    #x = np.asarray(x, dtype=float).ravel()
    return np.where(np.abs(x) <= a, x, a * np.sign(x))


