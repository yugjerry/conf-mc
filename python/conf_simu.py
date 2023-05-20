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



def cfmc_simu(M_star, S, P, base="als", rk=5, alpha=0.1):
    # M_star: underlying true matrix
    # S: locations for observed entries
    # P: true observation probability
    # base is either als or cvx
    # rk is the rank used in als
    # 1 - alpha is the coverage level
    
    M_obs = M_star * S    
    
    # construct lower & upper bnds
    q = 0.8
    lo_als, up_als, r, qvals, M_cf_als, s_cf_als = cmc_alg(M_obs, S, alpha, q, rk, P, "homo", base)

    # model-based: alternating least squares
    lo_uq_als, up_uq_als, Mhat_als, s_als = model_based(rk,M_obs,S,M_star,alpha,base)

    
    # evaluation
    m_star = M_star[~S].reshape(-1)

    # compute coverage rate and average length
    coverage_cmc_als = np.mean((lo_als <= m_star) & (up_als >= m_star))
    coverage_als = np.mean((lo_uq_als <= m_star) & (up_uq_als >= m_star))
    length_cmc_als = np.round(np.mean((up_als - lo_als)),4)
    length_als = np.round(np.mean((up_uq_als - lo_uq_als)),4)

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
