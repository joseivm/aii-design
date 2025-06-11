import numpy as np
import pandas as pd
from scipy.stats import genpareto, t
from bisect import bisect_left

import os

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

# Input files/dirs
DATA_DIR = os.path.join(PROJECT_DIR,'data')

# Output files/dirs
EXPERIMENTS_DIR = os.path.join(PROJECT_DIR,'experiments')
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR,'evaluation','Thailand')
PREDICTIONS_DIR = os.path.join(EXPERIMENTS_DIR,'prediction','Thailand')
MODELS_DIR = os.path.join(PREDICTIONS_DIR, 'model-preds')

# Global Variables
ZONES = ['C1','C2','C3','N1','N2','N3','NE1','NE2','NE3','S1','S2']
TEST_YEARS = np.arange(2015, 2023)

##### Data Loading #####
def load_all_zone_preds(zones, params):
    c_k, w_0, alpha = params['c_k'], params['w_0'], params['risk_coef']
    zdfs = []
    for zone in zones:
        zdf = load_zone_preds(zone, c_k, w_0, alpha)
        zdfs.append(zdf)

    df = pd.concat(zdfs, ignore_index=True)
    df = df.loc[df.Set == 'Test',:]
    df.loc[df.PredLoss < 0,'PredLoss'] = 0
    df.loc[df.PredLoss > 1,'PredLoss'] = 1
    return df

def load_zone_preds(zone, c_k, w_0, alpha, method='VMX'):
    model_name = get_best_model(zone, c_k, w_0, alpha, method)
    dfs = []
    for year in TEST_YEARS:
        df = load_model_predictions(model_name, zone, year)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    df['Province'] = df.Idx.apply(lambda x: x[1:3])
    return df

def load_model_predictions(model_name, zone, test_year):
    pred_dir = os.path.join(PREDICTIONS_DIR,'model-preds',model_name)
    pred_file = os.path.join(pred_dir,f"{zone}_{test_year}_preds.csv")
    pdf = pd.read_csv(pred_file)
    return pdf

def get_best_model(zone, c_k, w_0, alpha, method='VMX'):
    res_dir = os.path.join(RESULTS_DIR,'Val','full-results')
    results_fname = os.path.join(res_dir,f"{zone}.csv")
    rdf = pd.read_csv(results_fname)
    rdf = rdf.loc[(rdf['c_k'] == c_k) & (rdf.Zone == zone) & (rdf.w_0 == w_0) & (rdf.alpha == alpha),:]
    rdf = rdf.loc[rdf['Method'] == method,:]
    rdf = rdf.groupby('EvalName')['DeltaCE'].mean().reset_index(name='DeltaCE')
    idx = rdf['DeltaCE'].idxmax()
    best_model = rdf.loc[idx, 'EvalName']
    return '_'.join(best_model.split('_')[1:-3])

def create_param_dict(zones, c_k, test_year, w_0=0.1, alpha=2):
    params = {'epsilon_p': 0.01, 'c_k': c_k, 'subsidy': 0, 'w_0': w_0, 'premium_ub':0.25,  
            'risk_coef': alpha, 'S': 1, 'market_loading': 1, 'eval_set': 'Test',
            'test_year': test_year, 'zones': zones}
    return params


##### Copula things ##### 
def fit_spliced_severity(ldf: pd.DataFrame,
                         tail_q: float = 0.95,
                         loss_col: str = 'PredLoss') -> dict:
    """
    Fit an empirical+GPD splice per zone, but if a zone's data are all
    <= threshold (i.e. no tail) or all zeros, fall back to pure empirical.
    """
    sev_params = {}
    for zone, grp in ldf.groupby('Zone'):
        data = grp[loss_col].to_numpy()
        # threshold at tail_q
        u = np.quantile(data, tail_q)
        # split body/tail
        body = np.sort(data[data <= u])
        tail = data[data > u]

        # if no tail (or zone is constant), fall back to pure empirical
        if tail.size < 1 or body.size < 1 or np.all(data == data[0]):
            sev_params[zone] = {
                'u': u,
                'body': np.sort(data),
                'p_body': 1.0,
                'gpd_c': 0.0,
                'gpd_s': 1.0
            }
        else:
            # fit GPD to exceedances
            c, loc, s = genpareto.fit(tail, floc=u)
            sev_params[zone] = {
                'u': u,
                'body': body,
                'p_body': len(body) / len(data),
                'gpd_c': c,
                'gpd_s': s
            }
    return sev_params

def nearest_positive_definite(A: np.ndarray) -> np.ndarray:
    """
    Higham's algorithm to find the nearest positive-definite matrix to A.
    """
    # Step 1: symmetrize
    B = (A + A.T) / 2
    # Step 2: SVD of B
    U, s, Vt = np.linalg.svd(B)
    H = Vt.T @ np.diag(s) @ Vt
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    # Step 3: ensure positive definiteness
    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while True:
        try:
            np.linalg.cholesky(A3)
            return A3
        except np.linalg.LinAlgError:
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1

def force_spd_jitter(A: np.ndarray,
                     initial_jitter: float = 1e-8,
                     max_tries: int = 10) -> np.ndarray:
    """
    Make A symmetric positive-definite by adding jitter*I until Cholesky works.
    """
    # 1) Symmetrize
    B = (A + A.T) / 2.0

    # 2) Try Cholesky with increasing jitter
    jitter = initial_jitter
    I      = np.eye(A.shape[0])
    for _ in range(max_tries):
        try:
            # if this succeeds, B_j is SPD
            _ = np.linalg.cholesky(B + jitter * I)
            return B + jitter * I
        except np.linalg.LinAlgError:
            jitter *= 10.0

    # If we still fail, raise (should be extremely rare)
    raise np.linalg.LinAlgError(
        f"force_spd_jitter: failed to SPD after {max_tries} tries."
    )

def fit_t_copula_corr(U: pd.DataFrame, nu: float = 4) -> np.ndarray:
    """
    Estimate the correlation matrix for a Student-t copula given uniform data U (DataFrame),
    handling any columns that are effectively constant by giving them corr=1 with themselves
    and corr=0 with everything else.
    """
    eps = 1e-6
    # 1) clamp into (0,1) so t.ppf never returns infinities
    U_clamped = U.clip(eps, 1 - eps)

    # 2) invert via Student-t quantile
    X = t.ppf(U_clamped.values, df=nu)  # shape (n_obs, d)

    # 3) detect constant columns (zero sample std)
    stds = X.std(axis=0)
    idx_const = np.where(stds < 1e-12)[0]     # or == 0
    idx_var   = np.where(stds >= 1e-12)[0]

    d = X.shape[1]
    corr = np.eye(d, dtype=float)

    # 4) if at least two non-constant, fill that submatrix
    if len(idx_var) > 1:
        sub = np.corrcoef(X[:, idx_var], rowvar=False)
        for ii, i in enumerate(idx_var):
            for jj, j in enumerate(idx_var):
                corr[i, j] = sub[ii, jj]
    # 5) the rows/cols for idx_const remain all zeros except diag=1

    return corr

def simulate_t_copula_uniforms(corr: np.ndarray, nu: float, n_sim: int, random_state=None) -> np.ndarray:
    """
    Draw samples from a Student-t copula, ensuring corr is positive-definite.
    """
    # Ensure positive definiteness
    corr_pd = force_spd_jitter(corr)
    # proceed with simulation
    rng = np.random.default_rng(random_state)
    d = corr_pd.shape[0]
    L = np.linalg.cholesky(corr_pd)
    Z = rng.standard_normal((n_sim, d))
    chi2 = rng.chisquare(nu, size=n_sim) / nu
    T = (Z / np.sqrt(chi2)[:, None]) @ L.T
    return t.cdf(T, df=nu)

def invert_spliced(z: str, u_vals: np.ndarray, sev_params: dict) -> np.ndarray:
    """
    Vectorized inverse CDF for zone z given uniforms u_vals.
    """
    sp = sev_params[z]
    u = sp['u']
    body = sp['body']
    p_body = sp['p_body']
    n_body = len(body)
    # allocate output
    out = np.empty_like(u_vals)
    # body mask
    mask_body = u_vals < p_body
    if mask_body.any():
        # indices into body
        idx = np.floor((u_vals[mask_body] / p_body) * (n_body - 1)).astype(int)
        out[mask_body] = body[idx]
    if (~mask_body).any():
        # tail uniforms
        gq = (u_vals[~mask_body] - p_body) / (1 - p_body)
        out[~mask_body] = genpareto.ppf(gq,
                                        c=sp['gpd_c'],
                                        loc=u,
                                        scale=sp['gpd_s'])
    return out

def simulate_single_zone_payouts(
    ldf: pd.DataFrame,
    zone: str,
    payout_col: str = 'PredLoss',
    tail_q: float = 0.90,
    n_sim: int = 20000,
    random_state: int = None
) -> pd.DataFrame:
    """
    Simulate synthetic payout shares for a single zone based on its spliced-severity marginal.

    Parameters
    ----------
    ldf : pd.DataFrame
        DataFrame containing columns ['Zone', 'Year', payout_col].
    zone : str
        The zone identifier to simulate.
    payout_col : str
        Column name in ldf for the historical payout shares.
    tail_q : float
        Quantile threshold for splicing body/tail (e.g. 0.90).
    n_sim : int
        Number of synthetic draws.
    random_state : int
        RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Year', 'Zone', 'Payout'], where 'Year' is the
        simulation index (0..n_sim-1) and 'Payout' are simulated shares.
    """
    # 1) Fit only this zone's spliced severity
    zone_df = ldf.loc[ldf['Zone'] == zone, :]
    sev = fit_spliced_severity(zone_df, tail_q, payout_col)[zone]
    u_thr = sev['u']
    body = sev['body']
    p_body = sev['p_body']
    c, s = sev['gpd_c'], sev['gpd_s']
    n_body = len(body)

    # 2) Simulate uniforms
    rng = np.random.default_rng(random_state)
    U = rng.random(n_sim)

    # 3) Invert via spliced marginal
    payouts = np.empty(n_sim, dtype=float)
    # body draws
    mask_body = U < p_body
    if n_body > 0:
        idx = np.floor((U[mask_body] / p_body) * n_body).astype(int)
        idx = np.clip(idx, 0, n_body - 1)
        payouts[mask_body] = body[idx]
    else:
        # no body: all mass in tail, set to threshold
        payouts[mask_body] = u_thr
    # tail draws
    mask_tail = ~mask_body
    if mask_tail.any() and p_body < 1.0:
        gq = (U[mask_tail] - p_body) / (1 - p_body)
        payouts[mask_tail] = genpareto.ppf(gq, c=c, loc=u_thr, scale=s)

    # 4) Build result
    sim_df = pd.DataFrame({
        'Year': np.arange(n_sim),
        'Zone': zone,
        payout_col: payouts
    })
    sim_df.loc[sim_df[payout_col] < 0, payout_col] = 0
    return sim_df

def simulate_zone_payouts(ldf: pd.DataFrame,
                          payout_col='PredLoss',
                          tail_q: float = 0.90,
                          nu: float = 4,
                          n_sim: int = 20000,
                          random_state: int = None) -> pd.DataFrame:
    """
    Simulate correlated zone-level payouts.
    Returns a DataFrame of shape (n_sim x n_zones) with total payouts (size-adjusted).
    """
    # 1) Fit spliced severity per zone
    sev_params = fit_spliced_severity(ldf, tail_q, payout_col)
    zones = sorted(sev_params.keys())

    # 2) Build historical uniform matrix U_hist (one row per TestYear)
    annual = (ldf.groupby(['Year', 'Zone'])[payout_col]
                .mean()
                .unstack('Zone')
                .loc[:, zones]
                .dropna()
                .sort_index())
    U_hist = annual.copy().astype(float)

    for z in zones:
        sp     = sev_params[z]
        u_thr  = sp['u']
        body   = sp['body']              # sorted 1D array
        p_body = sp['p_body']
        c, s   = sp['gpd_c'], sp['gpd_s']
        n_body = len(body)

        # Build an xp/yp for interp
        if n_body > 1:
            xp = body
            yp = np.linspace(0, p_body, n_body)
        else:
            # degenerate: everything in the body is the same
            xp = np.array([body[0] - 1e-8, body[0], body[0] + 1e-8])
            yp = np.array([0.0, p_body, p_body])

        for yr in annual.index:
            x = annual.at[yr, z]
            if x <= u_thr or p_body == 1.0:
                # body‐only: linearly interpolate in [0, p_body]
                U_hist.at[yr, z] = np.interp(x, xp, yp)
            else:
                # tail CDF
                tail_u = genpareto.cdf(x, c=c, loc=u_thr, scale=s)
                U_hist.at[yr, z] = p_body + (1 - p_body) * tail_u

    # 3) Fit t-copula correlation
    corr = fit_t_copula_corr(U_hist, nu)

    # 4) Simulate uniforms from t-copula
    U_sim = simulate_t_copula_uniforms(corr, nu, n_sim, random_state)

    # 5) Invert uniforms to payout shares, then scale by zone size
    payouts = {}
    for j, z in enumerate(zones):
        shares = invert_spliced(z, U_sim[:, j], sev_params)
        payouts[z] = shares

    # 6) Build DataFrame
    sim_df = pd.DataFrame(payouts, index=pd.RangeIndex(n_sim, name='Year'))
    return sim_df.reset_index()

# ─────────────────────────────────────────────────────────────────────────────
# Example usage:
# params = create_param_dict(ZONES,0.13,2017)
# ldf = load_all_zone_preds(ZONES,params)
# sim_payouts = simulate_zone_payouts(ldf, tail_q=0.9, nu=4,
#                                     n_sim=20000, random_state=42)
# sim_payouts.head()
# ─────────────────────────────────────────────────────────────────────────────
