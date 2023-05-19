import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import scipy.stats
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm
import time
import scipy.linalg as scilinalg
import seaborn as sns
from scipy.stats import ortho_group
import pandas as pd

import multiprocessing as mp
from joblib import Parallel, delayed
from utils import *


def gen_(d1, d2, het, sd, tail, pr, M_mean, mis_set, k_star):
    '''
    The code defines a function called gen_ that generates a synthetic matrix with missing values. The function takes several parameters:
    
    d1: Number of rows in the matrix.
    d2: Number of columns in the matrix.
    het: Type of missingness pattern ('rank1', 'logis1', 'logis', 'logis2', 'homo').
    sd: Standard deviation of the noise.
    tail: Type of noise distribution ('exp', 'cauchy', 'gaussian', 'poisson', 't', 'het', 'het1').
    pr: Probability of missingness for homogeneous missingness.
    M_mean: Mean value of the true matrix.
    mis_set: Flag to indicate whether missing values should be generated according to the missingness pattern.
    k_star: Rank of the true matrix.
    '''

    N = d1*d2
    d = d1+d2

    def gen_P(het):
        # missingness pattern
        if het=='rank1':
            # rank-1 structure
            lo_p = 0.3
            up_p = 0.9
            u = np.random.uniform(lo_p,up_p,d1).reshape((d1,1))
            v = np.random.uniform(lo_p,up_p,d2).reshape((d2,1))
            P = u @ v.T
        elif het == 'logis1':
            L = 0.5
            k_p = 5
            u = L*np.random.uniform(0,2,d1*k_p).reshape((d1,k_p))
            v = L*np.random.uniform(-1,1,d2*k_p).reshape((d2,k_p))
            AA = u @ v.T
            P = 1/(1+np.exp(-AA))
        elif het == 'logis2':
            L = 0.5
            k_l = 1
            u = L*np.random.uniform(0,2,d1*k_l).reshape((d1,k_l))
            v = L*np.random.uniform(-1,1,d2*k_l).reshape((d2,k_l))
            AA = u@v.T
            P = 1/(1+np.exp(-AA))
        elif het == 'homo':
            
            P = pr * np.ones((d1,d2))
        return P
    P = gen_P(het)

    if mis_set == 1:
        M_star, M_obs, S = mat_gen_mis(d1,d2,P,k_star,M_mean)
    else:
        M_star, M_obs, S = mat_gen(d1,d2,P,k_star,M_mean)
    
    # tails

    if tail == 'gaussian':
        E = np.random.normal(0,sd,d1*d2).reshape((d1,d2))
    if tail == 't':
        E = sd * np.random.standard_t(1.2, size=d1*d2).reshape((d1,d2))
    if tail == 'het':
        E = np.random.normal(0,(0.5/P).ravel(),d1*d2).reshape((d1,d2))
    if tail == 'het1':
        Q = gen_P(het)
        E = np.random.normal(0,(0.5/Q).ravel(),d1*d2).reshape((d1,d2))
        

    M_obs = M_obs + E * S
    M_star += E
    assert (M_star * S == M_obs).all()
    S = S.astype(dtype=bool)
    return M_star, M_obs, P, S


def model_based(rk, M_obs, S, M_star, alpha, base):
    d1, d2 = M_obs.shape
    p_est = np.mean(S)
    u, s, vh = svds_(M_obs/p_est, rk)
    M_spectral = u @ np.diag(s) @ vh
    sigma_est_spec = np.sqrt(np.sum((( M_obs - M_spectral)*S)**2)/(d1*d2*p_est))

    if base == "als":
        Mhat, sigma_est, sigmaS = ALS_solve(M_obs, S, rk, 0.0)
        itn = 0
        while (np.linalg.norm((Mhat - M_star) * S) / np.linalg.norm(M_star * S) > 1) and (itn <= 5):
            Mhat, sigma_est, sigmaS = ALS_solve(M_obs, S, rk, 0.0)
            itn += 1
        s = np.sqrt(sigmaS**2 + sigma_est**2)
    elif base == "cvx":
        Mhat, X_d, Y_d, sigma_est, sigmaS = cvx_mc(M_obs, S, p_est, rk, sigma_est_spec, eta=1)

        s = np.sqrt(sigmaS**2 + sigma_est**2)

    mul = norm.ppf(1-alpha/2)
    lo_uq_mat = Mhat - s * mul
    up_uq_mat = Mhat + s * mul
    lo_uq = lo_uq_mat[~S].reshape(-1)
    up_uq = up_uq_mat[~S].reshape(-1)
    return lo_uq, up_uq, Mhat, s
        

def cfmc_simu(alpha, rk, M_obs, S, M_star, P, het, full_exp=False):
    
    d1, d2 = M_obs.shape
        
    # M_star: underlying true matrix
    # M_obs: partially observed matrix

    # unobserved indices
    ind_test = np.transpose(np.nonzero(S==0))
    n0 = ind_test.shape[0]
    
    # construct lower & upper bnds
    q = 0.8
    lo_als, up_als, r, qvals, M_cf_als, s_cf_als = cmc_alg(M_obs, S, alpha, q, rk, P, missing_model="homo", base="als")

    # model-based: alternating least squares
    lo_uq_als, up_uq_als, Mhat_als, s_als = model_based(rk,M_obs,S,M_star,alpha,base = "als")

    
    # evaluation
    m_star = []
    for i in range(ind_test.shape[0]):
        m_star = np.append(m_star, M_star[ind_test[i,0],ind_test[i,1]])

    # compute coverage rate and average length
    coverage_cmc_als = np.mean((lo_als <= m_star) & (up_als >= m_star))
    coverage_als = np.mean((lo_uq_als <= m_star) & (up_uq_als >= m_star))
    length_cmc_als = np.round(np.mean((up_als - lo_als)),4)
    length_als = np.round(np.mean((up_uq_als - lo_uq_als)),4)

    
    if full_exp:
        lo_cvx, up_cvx, r_, qvals_, M_cf_cvx, s_cf_cvx = cmc_alg(M_obs, S, alpha, q, rk, P, missing_model="homo", base="cvx")

        # model-based: convex
        lo_uq_cvx, up_uq_cvx, Mhat_cvx, s_cvx = model_based(rk,M_obs,S,M_star,alpha,base="cvx")
        
        coverage_cmc_cvx = np.mean((lo_cvx <= m_star) & (up_cvx >= m_star))
        coverage_cvx = np.mean((lo_uq_cvx <= m_star) & (up_uq_cvx >= m_star))
        length_cmc_cvx = np.round(np.mean((up_cvx - lo_cvx)),4)
        length_cvx = np.round(np.mean((up_uq_cvx - lo_uq_cvx)),4)
        
        return coverage_cmc_cvx, coverage_cmc_als, coverage_cvx, coverage_als, length_cmc_cvx, length_cmc_als, length_cvx, length_als
    
    else:
        return coverage_cmc_als, coverage_als, length_cmc_als, length_als
        
    



def cfmc_simu_hetero(alpha, rk, M_obs, S, M_star, P, het, full_exp=False):
    
    d1, d2 = M_obs.shape[0], M_obs.shape[1]
        
    # M_star: underlying true matrix
    # M_obs: partially observed matrix

    # unobserved indices
    ind_test = np.transpose(np.nonzero(S==0))
    n0 = ind_test.shape[0]

    
    # construct lower & upper bnds
    q = 0.8
    lo_als_hat, up_als_hat, r, qvals, _, _ = cmc_alg(M_obs, S, alpha, q, rk, P, missing_model=het, base="als")
    
    # oracle case: when P is known
    lo_als, up_als, _, _, _, _ = cmc_alg(M_obs, S, alpha, q, rk, P, missing_model="oracle", base="als")
    
    if full_exp:
        lo_cvx_hat, up_cvx_hat, r_, qvals_, M_cf_cvx, s_cf_cvx = cmc_alg(M_obs, S, alpha, q, rk, P, missing_model="oracle", base="cvx")
        
        lo_cvx, up_cvx, _, _, _, _ = cmc_alg(M_obs, S, alpha, q, rk, P, missing_model=het, base="cvx")
        
        # evaluation
        m_star = []
        for i in range(ind_test.shape[0]):
            m_star = np.append(m_star, M_star[ind_test[i,0],ind_test[i,1]])

        # compute coverage rate and average length
        coverage_cmc_cvx = np.mean((lo_cvx <= m_star) & (up_cvx >= m_star))
        coverage_cmc_als = np.mean((lo_als <= m_star) & (up_als >= m_star))
        coverage_cmc_cvx_hat = np.mean((lo_cvx_hat <= m_star) & (up_cvx_hat >= m_star))
        coverage_cmc_als_hat = np.mean((lo_als_hat <= m_star) & (up_als_hat >= m_star))
       
        
        length_cmc_cvx = np.round(np.mean((up_cvx - lo_cvx)),4)
        length_cmc_als = np.round(np.mean((up_als - lo_als)),4)
        length_cmc_cvx_hat = np.round(np.mean((up_cvx_hat - lo_cvx_hat)),4)
        length_cmc_als_hat = np.round(np.mean((up_als_hat - lo_als_hat)),4)
        
        return coverage_cmc_cvx, coverage_cmc_als, coverage_cmc_cvx_hat, coverage_cmc_als_hat, length_cmc_cvx, length_cmc_als, length_cmc_cvx_hat, length_cmc_als_hat
        
    else:
    
        # evaluation
        m_star = []
        for i in range(ind_test.shape[0]):
            m_star = np.append(m_star, M_star[ind_test[i,0],ind_test[i,1]])
        # compute coverage rate and average length
        coverage_cmc_als = np.mean((lo_als <= m_star) & (up_als >= m_star))
        coverage_cmc_als_hat = np.mean((lo_als_hat <= m_star) & (up_als_hat >= m_star))
        
        
        length_cmc_als = np.round(np.mean((up_als - lo_als)),4)
        length_cmc_als_hat = np.round(np.mean((up_als_hat - lo_als_hat)),4)

        return coverage_cmc_als, coverage_cmc_als_hat, length_cmc_als, length_cmc_als_hat
