import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm

"""
based on: Ni Zhu, David Betaille, Juliette Marais, Marion Berbineau. GNSS Integrity Enhancement for Urban Transport 
Applications by Error Characterization and Fault Detection and Exclusion (FDE). Géolocalisation et Navigation dans 
l'Espace et le Temps, Journées Scientifiques 2018 de l'URSI, Mar 2018, Paris, France. 11p. 
https://hal.science/hal-01757326v2
"""

def hpl(gnss_pd: pd.DataFrame, sigma:float|None=None, alpha: float = 0.05, dof:int=4,
        uncertainty_column:str="uncertainty_m", steering_vector_column:tuple=("e_x", "e_y", "e_z"),
        weight_column:str="weight", time_column:str="unix_time",  residuals_column:str="residuals_m") -> pd.DataFrame:
    """
    Computes protection level for every timestamp of a dataframe

    HPL = HPL_noise + HPL_bias

    HPL_noise = K(Pmd)*dmajor
    where K is an inflation factor, dmajor is the error uncertainty

    HPL_bias = max(SLOPE*sigma) * pbias
    SLOPE = sqrt[((Gn+)2 + (Ge+)2)/S]
    G+ = (Gt W G)-1 Gt W
    S = I - G G+
    pbias = sqrt(rt W r)

    :param gnss_pd: BITS raw dataframe
    :param sigma: Standard deviation of measurement noise
    :param alpha: Significance level
    :param dof: degree of freedom
    :param uncertainty_column: Name of the column containing d_major
    :param steering_vector_column: Names of steering vectors columns
    :param weight_column: Name of the weight column
    :param time_column: Name of time column
    :param residuals_column: Name of pseudorange residuals column
    :return: BITS raw dataframe with computed hpl
    """
    gnss_pd = gnss_pd.sort_values("unix_time").reset_index(drop=True) # Clean up

    # Apply integrity monitoring for each timestamp group
    out_pd = pd.DataFrame()
    for _, group in tqdm(gnss_pd.groupby(time_column, sort=True), desc="Applying global test"):
        # Build d_major
        d_major = group[uncertainty_column].to_numpy().ravel()

        # Build G
        G = np.vstack([group[column].to_numpy() for column in steering_vector_column])
        G = G.transpose()

        # Build residuals
        residuals = group[residuals_column].to_numpy().ravel().reshape(-1, 1)

        # Get weight matrix
        if sigma is not None:
            W = np.diag(np.full(residuals.shape[0], 1 / (sigma ** 2)))
        else:
            W = np.diag(group[weight_column])

        # Compute HPL
        noise, bias, protection = window_hpl(d_major, G, W, residuals, alpha, dof)

        group["hpl_noise_m"] = noise
        group["hpl_bias_m"] = float(bias)
        group["hpl_m"] = protection

        out_pd = pd.concat([out_pd, group], axis=0)

    return out_pd

def window_hpl(dmajor:np.ndarray, G:np.ndarray, W:np.ndarray, residuals:np.ndarray, alpha:float=0.05, dof:int=4) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes protection level at a single epoch

    :param dmajor: Error uncertainty along the semi-major axis of the error ellipse
    :param G: Geometry matrix
    :param W: Weight matrix
    :param residuals: Pseudorange residuals matrix
    :param alpha: Significance level
    :param dof: Degree of freedom
    :return: hpl_noise, hpl_bias, hpl
    """
    # 1 Compute impact of measurement noise
    noise = hpl_noise(dmajor, alpha=alpha, dof=dof)

    # 2 Compute impact of measurement bias
    bias = hpl_bias(G, W, residuals)

    # Compute protection level
    protection = noise + bias

    return noise, bias, protection


def hpl_noise(dmajor:np.ndarray, alpha:float=0.05, dof:int=4) -> np.ndarray:
    """
    HPL_noise = K(Pmd)*dmajor, where K is an inflation factor, dmajor is the error uncertainty

    :param dmajor: Error uncertainty along the semi-major axis of the error ellipse
    :param alpha: Significance level
    :param dof: Degree of freedom
    :return: hpl_noise
    """
    K = np.sqrt(stats.chi2.ppf(1 - alpha, df=dof))

    return K*dmajor


def hpl_bias(G: np.ndarray, W: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """
    HPL_bias = max(SLOPE*sigma) * pbias
    SLOPE = sqrt[((Gn+)2 + (Ge+)2)/S]
    G+ = (Gt W G)-1 Gt W
    S = I - G G+
    pbias = sqrt(rt W r)

    :param G: Geometry matrix
    :param W: Weight matrix
    :param residuals: Pseudorange residuals matrix
    :return: hpl_bias
    """
    sigma = np.diag(W).ravel()
    GtW = G.T @ W
    GtWG = GtW @ G
    try:
        G_plus = np.linalg.inv(GtWG) @ GtW
    except np.linalg.LinAlgError:
        print("Could not compute hpl_bias")
        return np.array([np.nan])

    S = np.eye(len(residuals)) - G @ G_plus

    slope = np.sqrt(np.sum(G_plus**2, axis=0)/np.diag(S))

    p_bias = np.sqrt(residuals.transpose() @ W @ residuals)

    return (np.max(slope * sigma) * p_bias).ravel()