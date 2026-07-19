import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm

"""
based on: Ni Zhu. GNSS Propagation Channel Modeling in Constrained Environments: Contribution to the Improvement of the 
Geolocation Service Quality/. Engineering Sciences [physics]. Université de Lille, 2018. English. 
https://hal.science/tel-01959797v1
"""

def global_test(gnss_pd: pd.DataFrame, sigma: float|None=None, alpha: float = 0.05, number_of_unknown:int=4,
                weight_column:str="weight", time_column:str="unix_time",  residuals_column:str="residuals_m",
                verbose:bool=False) -> pd.DataFrame:
    """
    Performs global test on a dataframe with multiple timestamps.

    Global Test (GT) is generally implemented as the primary stage in the whole FDE scheme. It uses the Normalized Sum
    of Squared Error (NSSE) as the test statistic.

    :param gnss_pd: BITS raw dataframe
    :param sigma: Standard deviation of measurement noise set to None to use 1/weight² as sigma
    :param alpha: Significance level
    :param number_of_unknown: Number of unknowns to solve
    :param weight_column: Name of the weight column
    :param time_column: Name of time column
    :param residuals_column: Name of pseudorange residuals column
    :param verbose: set to True for verbose output
    :return: BITS raw dataframe with test results in the "valid_estimate" column
    """

    # Clean up
    gnss_pd = gnss_pd.sort_values(time_column).reset_index(drop=True)

    gnss_pd["valid_estimate"] = None
    gnss_pd["test_statistic"] = None

    # Apply integrity monitoring for each timestamp group
    groups = gnss_pd.groupby(time_column, sort=True)
    iterator = tqdm(groups, desc="Applying global test") if verbose else groups

    for _, group in iterator:
        # Get residuals
        residuals = group[residuals_column].to_numpy().reshape(-1, 1)

        # Get weight matrix
        if sigma is not None:
            W = np.diag(np.full(residuals.shape[0], 1/(sigma**2)))
        else:
            W = np.diag(group[weight_column])

        result, chi2_stat, chi2_threshold = window_global_test(residuals, W, alpha, number_of_unknown)

        gnss_pd.loc[group.index, "valid_estimate"] = result
        gnss_pd.loc[group.index, "test_statistic"] = float(chi2_stat)
        gnss_pd.loc[group.index, "test_threshold"] = float(chi2_threshold)

    return gnss_pd

def window_global_test(residuals:np.ndarray, W:np.ndarray, alpha:float, number_of_unknown:int) \
        -> tuple[bool, np.ndarray, np.ndarray]:
    """
    Performs global test on a single timestamp.

    Global Test (GT) is generally implemented as the primary stage in the whole FDE scheme. It uses the Normalized Sum
    of Squared Error (NSSE) as the test statistic.

    :param residuals: Pseudorange residuals matrix
    :param W: Weight matrix
    :param alpha: Significance level
    :param number_of_unknown: Number of unknowns to solve
    :return: True if passed, else False, chi2 test statistic
    """
    # Checking minimal number of measurement
    if len(residuals) < number_of_unknown:
        return False, np.array([np.nan])

    # chi2 statistic: Normalized Sum of Squared Error
    chi2_stat = residuals.transpose() @ W @ residuals

    # Threshold
    chi2_threshold = stats.chi2.ppf(1 - alpha, df=len(residuals) - number_of_unknown)

    return bool(chi2_stat < chi2_threshold), chi2_stat, chi2_threshold


def local_test(gnss_pd: pd.DataFrame, sigma: float|None=None, alpha: float = 0.05,
               weight_column:str="weight", time_column:str="unix_time",  residuals_column:str="residuals_m",
               steering_vector_column:tuple=("e_x", "e_y", "e_z"), verbose:bool=False) -> pd.DataFrame:
    """
    Performs local test on a dataframe with multiple timestamps.

    If a fault is detected by Global Test (GT), that means an outlier exists in one or several measurements. A Local
    Test (LT) can be carried out so as to identify the outlier. The LT uses the normalized residuals as test statistic.

    :param gnss_pd: BITS raw dataframe
    :param sigma: Standard deviation of measurement noise set to None to use 1/weight² as sigma
    :param alpha: Significance level
    :param weight_column: Name of the weight column
    :param time_column: Name of time column
    :param residuals_column: Name of pseudorange residuals column
    :param steering_vector_column: Names of steering vectors columns
    :param verbose: set to True for verbose output
    :return: BITS raw dataframe with test results in the "valid_estimate" column
    """
    # Clean up
    gnss_pd = gnss_pd.sort_values(time_column).reset_index(drop=True)

    gnss_pd["valid_estimate"] = None
    gnss_pd["test_statistic"] = None

    # Apply integrity monitoring for each timestamp group
    groups = gnss_pd.groupby(time_column, sort=True)
    iterator = tqdm(groups, desc="Applying local test") if verbose else groups

    for _, group in iterator:
        # Build residuals
        residuals = group[residuals_column].to_numpy().reshape(-1, 1)

        # Build weight matrix
        if sigma is not None:
            W = np.diag(np.full(residuals.shape[0], 1 / (sigma ** 2)))
        else:
            W = np.diag(group[weight_column])

        # Build G
        G = np.vstack([group[column].to_numpy() for column in steering_vector_column])
        G = G.transpose()

        valid_estimate, normalized_residuals, chi2_threshold  = window_local_test(residuals, W, G, alpha)

        gnss_pd.loc[group.index, "valid_estimate"] = valid_estimate
        gnss_pd.loc[group.index, "test_statistic"] = normalized_residuals
        gnss_pd.loc[group.index, "test_threshold"] = chi2_threshold

    return gnss_pd

def window_local_test(residuals:np.ndarray, W:np.ndarray, G:np.ndarray, alpha:float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs local test on a single timestamp.

    If a fault is detected by Global Test (GT), that means an outlier exists in one or several measurements. A Local
    Test (LT) can be carried out so as to identify the outlier. The LT uses the normalized residuals as test statistic.

    :param residuals: Pseudorange residuals matrix
    :param W: Weight matrix
    :param G: Geometry matrix
    :param alpha: Significance level
    :return: True if passed, else False, chi2 test statistic
    """
    # Threshold with Bonferroni correction
    chi2_threshold = stats.norm.ppf(1 - alpha / (2 * len(residuals)))

    # chi2 statistic: normalized residuals
    try:
        Cr = np.linalg.inv(W) - G @ np.linalg.inv(G.transpose() @ W @ G) @ G.transpose()
    except np.linalg.LinAlgError:
        return np.full(len(residuals), False), np.full(len(residuals), np.nan)

    chi2_stat = np.abs(residuals.flatten() / np.sqrt(np.diag(Cr)))

    valid_estimate = chi2_stat < chi2_threshold

    # Redundancy check
    R = Cr @ W
    r_diag = np.diag(R)

    for i in np.where(~valid_estimate)[0]:
        off_diag = np.abs(np.delete(R[i, :], i))
        if off_diag.size > 0 and r_diag[i] <= off_diag.max():
            valid_estimate[i] = np.nan

    return valid_estimate, chi2_stat, chi2_threshold