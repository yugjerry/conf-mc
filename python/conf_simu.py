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


def gen_(d1,d2,het,sd,tail,pr,M_mean,mis_set,k_star):
    N = d1*d2
    d = d1+d2

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
#         u = L*np.random.uniform(-1,1,d1*k_p).reshape((d1,k_p))
        v = L*np.random.uniform(-1,1,d2*k_p).reshape((d2,k_p))
        AA = u @ v.T
        P = 1/(1+np.exp(-AA))
    elif het == 'logis':
        L = 1
        u = L*np.random.uniform(-1,1,d1).reshape((d1,1))
        v = L*np.random.uniform(-1,1,d2).reshape((d2,1))
        o1 = np.ones((d1,1))
        o2 = np.ones((d2,1))
        AA = u[:,:1]@o2.T + o1@v[:,:1].T
        P = 1/(1+np.exp(-AA))
    elif het == 'homo':
        # homogeneous missingness
        pr = pr
        P = pr * np.ones(N).reshape((d1,d2))

    if mis_set == 1:
        M_star, A, S = mat_gen_mis(d1,d2,P,k_star,M_mean)
    elif mis_set == 2:
        M_star, A, S = mat_gen_mis_(d1,d2,P,k_star,M_mean)
    else:
        M_star, A, S = mat_gen(d1,d2,P,k_star,M_mean)
    
    
    if tail == 'exp':
        E = sd * np.random.exponential(size=d1*d2).reshape((d1,d2))
    if tail == 'cauchy':
        E = sd * np.random.standard_cauchy(d1*d2).reshape((d1,d2))
    if tail == 'gaussian':
        E = np.random.normal(0,sd,d1*d2).reshape((d1,d2))
    if tail == 'poisson':
        M_p = np.random.poisson(M_star)
        E = M_p - M_star
    if tail == 't':
        E = sd * np.random.standard_t(1.2, size=d1*d2).reshape((d1,d2))
    if tail == 'het':
        E = np.random.normal(0,(0.5/P).ravel(),d1*d2).reshape((d1,d2))
    if tail == 'het1':
        if het=='rank1':
            # rank-1 structure
            lo_p = 0.3
            up_p = 0.9
            u = np.random.uniform(lo_p,up_p,d1).reshape((d1,1))
            v = np.random.uniform(lo_p,up_p,d2).reshape((d2,1))
            Q = u @ v.T
        elif het == 'logis1':
            u = L*np.random.uniform(0,2,d1*k_p).reshape((d1,k_p))
#             u = L*np.random.uniform(-1,1,d1*k_p).reshape((d1,k_p))
            v = L*np.random.uniform(-1,1,d2*k_p).reshape((d2,k_p))
            AA = u @ v.T
            Q = 1/(1+np.exp(-AA))
        elif het == 'logis':
            L = 1
            u = L*np.random.uniform(-1,1,d1).reshape((d1,1))
            v = L*np.random.uniform(-1,1,d2).reshape((d2,1))
            o1 = np.ones((d1,1))
            o2 = np.ones((d2,1))
            AA = u[:,:1]@o2.T + o1@v[:,:1].T
            Q = 1/(1+np.exp(-AA))
        elif het == 'homo':
            # homogeneous missingness
            pr = pr
            Q = pr * np.ones(N).reshape((d1,d2))
        E = np.random.normal(0,(0.5/Q).ravel(),d1*d2).reshape((d1,d2))
        
    A = A + E * S
    M_star += E
    
    return M_star, A, P

def cfmc_simu(alpha,rk,A,M_star,P,het,kap,sigma_true=False,plot=False,full_exp=False):
    
    d1, d2 = A.shape[0], A.shape[1]
    S = (A!=0)
        
    # M_star: underlying true matrix
    # A: partially observed matrix

    # unobserved indices
    ind_test_all = np.transpose(np.nonzero(S==0))
    n0 = ind_test_all.shape[0]
    # randomly choose m of missing entries
#     ind_test = ind_test_all[np.random.choice(n0,m),:]
    ind_test = ind_test_all
    
    # construct lower & upper bnds
    base2 = 'als'    # base algorithm
    lo_als, up_als, r, qvals, M_cf_als, s_cf_als = CP_split_svd(A,ind_test,alpha,P,rk,wtd=True,het=het,w=[],oracle=False,base=base2,kap=kap)
#     # oracle case: when P is known
#     lo, up, _, _ = CP_split_svd(A,ind_test,alpha,P,rk,wtd=True,het=het,w=[],oracle=True,base=base)
    
    # model-based methods
    p_est = np.mean(S)
    u, s, vh = svds_(A/p_est, rk)
    M_spectral = u@np.diag(s)@vh
    sigma_est_spec = np.sqrt(np.sum(((A-M_spectral)*S)**2)/(d1*d2*p_est))
    
    
    # alternative least squares
    penalty = 0.0
    Mhat_als, sigma_est_als, sigmaS_als = ALS_solve(A, S, rk, penalty)
    itn = 0
    while (np.linalg.norm(Mhat_als-M_star) / np.linalg.norm(M_star) > 1) and (itn <= 5):
        Mhat_als, sigma_est_als, sigmaS_als = ALS_solve(A, S, rk, penalty)
        itn += 1
        
    s_als = np.sqrt(sigmaS_als**2 + sigma_est_als**2)

    mul = norm.ppf(1-alpha/2)
    lo_uq_mat = Mhat_als - s_als * mul
    up_uq_mat = Mhat_als + s_als * mul
    lo_uq_als = lo_uq_mat[S==0].reshape(-1)
    up_uq_als = up_uq_mat[S==0].reshape(-1)

    
    # evaluation
    m_star = []
    for i in range(ind_test.shape[0]):
        m_star = np.append(m_star, M_star[ind_test[i,0],ind_test[i,1]])

    label2 = 'cf-'+base2
    label4 = 'als'
    # compute coverage rate and average length
    cov_rt_als = np.mean((lo_als <= m_star) & (up_als >= m_star))
    cov_rt_uq_als = np.mean((lo_uq_als <= m_star) & (up_uq_als >= m_star))
    len_ave_als = np.round(np.mean((up_als - lo_als)),4)
    len_ave_uq_als = np.round(np.mean((up_uq_als - lo_uq_als)),4)
    
    u_cf_als = np.divide((M_cf_als - M_star)[S==0],s_cf_als[S==0]).reshape(-1)
    u = np.random.normal(0,1,10000)
    u_hat_als = np.divide((Mhat_als - M_star)[S==0], s_als[S==0]).reshape(-1)
    
    if plot==True:
        fig, ax = plt.subplots(ncols=2,figsize=(6,4))
        fig.tight_layout(pad=0.6)
        sns.set(font_scale = 1.4)

        sns.distplot(u_cf_als[np.abs(u_cf_als)<5], bins=60,kde=True, hist=True,label='true',ax=ax[0])
        sns.distplot(u, bins=20,kde=True, hist=True,label='theory',ax=ax[0])
        ax[0].legend(loc='best')
        ax[0].set_title('als')

        sns.distplot(u_hat_als[np.abs(u_hat_als)<5], bins=60,kde=True, hist=True,label='true',ax=ax[1])
        sns.distplot(u, bins=20,kde=True, hist=True,label='theory',ax=ax[1])
        ax[1].legend(loc='best')
        ax[1].set_title('als')

    #     # compare results with oracle and estimated P
    #     ind = np.arange(250,300)
    #     ind_seq = np.arange(len(ind))
    #     ax[1].plot(ind_seq,m_star[ind],'b+',label='true entry')
    #     ax[1].plot(ind_seq,lo[ind],label=label1,c='darkgreen')
    #     ax[1].plot(ind_seq,up[ind],c='darkgreen')
    #     ax[1].plot(ind_seq,lo_hat[ind],label=label2,c='red')
    #     ax[1].plot(ind_seq,up_hat[ind],c='red')
    #     ax[1].plot(ind_seq,lo_uq_cvx[ind],label=label3,c='cyan')
    #     ax[1].plot(ind_seq,up_uq_cvx[ind],c='cyan')
    #     ax[1].plot(ind_seq,lo_uq_als[ind],label=label4,c='orange')
    #     ax[1].plot(ind_seq,up_uq_als[ind],c='orange')
    #     ax[1].legend(loc='best', bbox_to_anchor=(1, 1))
    #     ax[1].set_xlabel('index')
    #     ax[1].set_ylabel('unobserved entries')
    #     ax[1].set_title('Lower and upper bounds')
    
    if full_exp:
        base1 = 'cvx'    # base algorithm
        label1 = 'cf-'+base1
        label3 = 'cvx'
    
        lo_cvx, up_cvx, r_, qvals_, M_cf_cvx, s_cf_cvx = CP_split_svd(A,ind_test,alpha,P,rk,wtd=True,het=het,w=[],oracle=False,base=base1,kap=kap)

        # convex
        eta = 1
        Mhat_cvx, X_d_cvx, Y_d_cvx, sigma_est_cvx, sigmaS_cvx = cvx_mc(A, S, p_est, rk, sigma_est_spec, eta=eta)

        s_cvx = np.sqrt(sigmaS_cvx**2 + sigma_est_cvx**2)

        lo_uq_mat = Mhat_cvx - s_cvx * mul
        up_uq_mat = Mhat_cvx + s_cvx * mul
        lo_uq_cvx = lo_uq_mat[S==0].reshape(-1)
        up_uq_cvx = up_uq_mat[S==0].reshape(-1)
        
        cov_rt_cvx = np.mean((lo_cvx <= m_star) & (up_cvx >= m_star))
        cov_rt_uq_cvx = np.mean((lo_uq_cvx <= m_star) & (up_uq_cvx >= m_star))
        len_ave_cvx = np.round(np.mean((up_cvx - lo_cvx)),4)
        len_ave_uq_cvx = np.round(np.mean((up_uq_cvx - lo_uq_cvx)),4)
        
        return cov_rt_cvx, cov_rt_als, cov_rt_uq_cvx, cov_rt_uq_als, len_ave_cvx, len_ave_als, len_ave_uq_cvx, len_ave_uq_als
    
    else:
        return cov_rt_als, cov_rt_uq_als, len_ave_als, len_ave_uq_als
        
    



def cfmc_simu_hetero(alpha,rk,A,M_star,P,het,kap,sigma_true=False,plot=False,full_exp=False):
    
    d1, d2 = A.shape[0], A.shape[1]
    S = (A!=0)
        
    # M_star: underlying true matrix
    # A: partially observed matrix

    # unobserved indices
    ind_test_all = np.transpose(np.nonzero(S==0))
    n0 = ind_test_all.shape[0]
    # randomly choose m of missing entries
#     m = 500
#     ind_test = ind_test_all[np.random.choice(n0,m),:]
    ind_test = ind_test_all
    
    # construct lower & upper bnds
    base2 = 'als'    # base algorithm
    lo_als_hat, up_als_hat, r, qvals, _, _ = CP_split_svd(A,ind_test,alpha,P,rk,wtd=True,het=het,w=[],oracle=False,base=base2,kap=kap)
    
    # oracle case: when P is known
    lo_als, up_als, _, _, _, _ = CP_split_svd(A,ind_test,alpha,P,rk,wtd=True,het=het,w=[],oracle=True,base=base2,kap=kap)
    
    if full_exp:
        base1 = 'cvx'    # base algorithm
        lo_cvx_hat, up_cvx_hat, r_, qvals_, M_cf_cvx, s_cf_cvx = CP_split_svd(A,ind_test,alpha,P,rk,wtd=True,het=het,w=[],oracle=False,base=base1,kap=kap)
        
        lo_cvx, up_cvx, _, _, _, _ = CP_split_svd(A,ind_test,alpha,P,rk,wtd=True,het=het,w=[],oracle=True,base=base1,kap=kap)
        
        # evaluation
        m_star = []
        for i in range(ind_test.shape[0]):
            m_star = np.append(m_star, M_star[ind_test[i,0],ind_test[i,1]])
        label1 = 'cf*-'+base1
        label2 = 'cf*-'+base2
        label3 = 'cf-'+base1
        label4 = 'cf-'+base2
        # compute coverage rate and average length
        cov_rt_cvx = np.mean((lo_cvx <= m_star) & (up_cvx >= m_star))
        cov_rt_als = np.mean((lo_als <= m_star) & (up_als >= m_star))
        cov_rt_cvx_hat = np.mean((lo_cvx_hat <= m_star) & (up_cvx_hat >= m_star))
        cov_rt_als_hat = np.mean((lo_als_hat <= m_star) & (up_als_hat >= m_star))
        len_ave_cvx = np.round(np.mean((up_cvx - lo_cvx)),4)
        len_ave_als = np.round(np.mean((up_als - lo_als)),4)
        len_ave_cvx_hat = np.round(np.mean((up_cvx_hat - lo_cvx_hat)),4)
        len_ave_als_hat = np.round(np.mean((up_als_hat - lo_als_hat)),4)
        
        return cov_rt_cvx, cov_rt_als, cov_rt_cvx_hat, cov_rt_als_hat, len_ave_cvx, len_ave_als, len_ave_cvx_hat, len_ave_als_hat
        
    else:
    
        # evaluation
        m_star = []
        for i in range(ind_test.shape[0]):
            m_star = np.append(m_star, M_star[ind_test[i,0],ind_test[i,1]])
        # compute coverage rate and average length
        cov_rt_als = np.mean((lo_als <= m_star) & (up_als >= m_star))
        cov_rt_als_hat = np.mean((lo_als_hat <= m_star) & (up_als_hat >= m_star))
        len_ave_als = np.round(np.mean((up_als - lo_als)),4)
        len_ave_als_hat = np.round(np.mean((up_als_hat - lo_als_hat)),4)

        return cov_rt_als, cov_rt_als_hat, len_ave_als, len_ave_als_hat
