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
import math
from proj import *
from SPG import *




'''
The code below generates the simulated inputs we used in the paper.
'''

# generate incoherent matrix M^*
def mat_gen(d1,d2,P,k,M_mean=2):
    U = np.random.normal(0,1,d1*k).reshape((d1,k))
    V = np.random.normal(0,1,d2*k).reshape((d2,k))
    U = scilinalg.orth(U)
    V = scilinalg.orth(V)
    
    M_star = U @ V.T
    M_star = M_star / np.mean(np.abs(M_star)) * M_mean
    S = np.less(np.random.rand(d1,d2), P)
    M = M_star * S
    
    return M_star, M, S

# generate non-incoherent matrix M^*
def mat_gen_mis(d1,d2,P,k,M_mean=2):
    U = np.random.standard_t(1.2, size=d1*k).reshape((d1,k))
    V = np.random.standard_t(1.2, size=d2*k).reshape((d2,k))
    U = scilinalg.orth(U)
    V = scilinalg.orth(V)
    
    M_star = U @ V.T
    M_star = M_star/np.mean(np.abs(M_star)) * M_mean
    S = np.less(np.random.rand(d1,d2), P)
    M = M_star * S
    
    return M_star, M, S


def gen_P(d1, d2, het, pr):
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


def gen_data(d1, d2, het, sd, tail, pr, M_mean, mis_set, k_star):
    P = gen_P(d1, d2, het, pr)

    if mis_set == 1:
        M_star, M_obs, S = mat_gen_mis(d1,d2,P,k_star,M_mean)
    else:
        M_star, M_obs, S = mat_gen(d1,d2,P,k_star,M_mean)
    
    # tails

    if tail == 'gaussian':
        E = sd * np.random.normal(0,1,d1*d2).reshape((d1,d2))
    if tail == 't':
        E = sd * np.random.standard_t(1.2, size=d1*d2).reshape((d1,d2))
    if tail == 'het':
        E = np.random.normal(0,(0.5/P).ravel(),d1*d2).reshape((d1,d2))
    if tail == 'het1':
        Q = gen_P(d1, d2, het, pr)
        E = np.random.normal(0,(0.5/Q).ravel(),d1*d2).reshape((d1,d2))
        
    M_star += E
    M_obs = M_obs + E * S
    
    assert (M_star * S == M_obs).all()
    return M_star, M_obs, P, S






'''
The code below implements the estimation of the observation probability matrix using one bit matrix completion.

Reference: 1-bit matrix completion on https://mdav.ece.gatech.edu/software/
'''


def f_(x,q):
    return q/(1+np.exp(-x))

def fprime(x,q):
    return q*np.exp(x)/(1+np.exp(x))**2

def logObjectiveGeneral(x,y,idx,f,fprime):
    F = -np.sum(np.log(y[idx]*f(x[idx]) - (y[idx]-1)/2))
    G = np.zeros(len(x))
    v = (f(x[idx])+(y[idx]-1)/2)
    w = -fprime(x[idx])
    G[idx] = w/v

    return F, G

def projectNuclear(B,d1,d2,radius,alpha):
    U,S,Vh = np.linalg.svd(B.reshape((d1,d2)), full_matrices=False)

    s2 = euclidean_proj_l1ball(S,radius)

    B_proj = U@np.diag(s2)@Vh
    
    return B_proj.reshape((d1*d2,))


def estimate_P(S_train, q, P, missing_model='homo'):
    d1, d2 = S_train.shape
    if missing_model == "oracle":
        P_hat = P
    elif missing_model == "homo":
        P_hat = (1/q)* np.mean(S_train) * np.ones((d1,d2))
    elif missing_model == 'logis1' or missing_model == 'logis2':
        yy = 2*(S_train-0.5).ravel()
        x_init = np.zeros(d1*d2)
        idx = range(d1*d2)
        const   = 1.0
        if missing_model == 'logis1':
            k_l = 5
        else:
            k_l = 1
        radius  = const * np.sqrt(d1*d2*k_l)
        f_loc = lambda x: f_(x,q )
        fprime_loc = lambda x: fprime(x,q )
        funObj  = lambda x_var: logObjectiveGeneral(x_var,yy,idx,f_loc,fprime_loc)
        funProj = lambda x_var: projectNuclear(x_var,d1,d2,radius,const)

        default_options = SPGOptions()
        default_options.maxIter = 10000
        default_options.verbose = 2
        default_options.suffDec = 1e-4
        default_options.progTol = 1e-9
        default_options.optTol = 1e-9
        default_options.curvilinear = 1
        default_options.memory = 10
        default_options.useSpectral = True
        default_options.bbType = 1
        default_options.interp = 2  # cubic
        default_options.numdiff = 0
        default_options.testOpt = True
        spg_options = default_options
        x_,F_ = SPG(funObj, funProj, x_init, spg_options)
        A_hat = x_.reshape((d1,d2))
        U,s_hat,Vh = np.linalg.svd(A_hat)
        M_d = U[:,:k_l]@np.diag(s_hat[:k_l])@Vh[:k_l,:]
        P_hat = (1/q )*f_(M_d,q ).reshape((d1,d2)) 
    elif missing_model=='rank1':
        u_hat, s_hat, vt_hat = svds_(S_train,1)
        P_hat = (1/q)*u_hat @ np.diag(s_hat) @ vt_hat
    return P_hat 



'''
Various matrix completion estimation algorithms
'''


def svds_(M_,k):
    u, s, vt = scilinalg.svd(M_, full_matrices=False)
    u_k = u[:,:k]
    s_k = s[:k]
    vt_k = vt[:k,:]
    return u_k, s_k, vt_k


# singular value thresholding
def SVT(X, tau):
    U,S,Vh = scilinalg.svd(X, full_matrices=False)
    S = np.maximum(S-tau, np.zeros(len(S)))
    Z = U @ np.diag(S) @ Vh
    return Z

def cvx_mc(A, S, p_est, rk, sigma_est, lam=0, eta=0.1, max_iter=1000):
    # eta: learning rate
    # max_iter: max number of iterations
    d1, d2 = A.shape
    u, s, vh = svds_(A / p_est, rk)
    v = vh.T
    M_spectral = u @ np.diag(s) @ vh
    X = u @ np.diag(np.sqrt(s))
    Y = v @ np.diag(np.sqrt(s))
    
    # estimate sigma
    function_value_old = 10000
    while 1:
        # fix sigma_est, estimate M
        Z_prox = M_spectral
        lam = 2*sigma_est*np.sqrt(max(d1,d2) * p_est)
        for t in range(1000):
            Z_new = SVT(Z_prox-eta*(Z_prox-A)*S,lam*eta)
            grad_norm = np.linalg.norm((Z_new-Z_prox)/eta,'fro')/np.linalg.norm(Z_prox,'fro')
            Z_prox = Z_new
            if grad_norm < 1e-8:
                break
        # A square-root lasso type loss
        u_prox, s_prox, vh_prox = np.linalg.svd(Z_prox, full_matrices=False)
        function_value_new = np.linalg.norm((Z_prox-A)*S,'fro')**2/(2*sigma_est)
        +d1*d2*p_est*sigma_est/2+lam/sigma_est*np.linalg.norm(s_prox,1)
        if (function_value_old-function_value_new)/function_value_old <= 1e-3:
            break
        function_value_old = function_value_new
        sigma_est = np.linalg.norm((Z_prox-A)*S,'fro')/np.sqrt(d1*d2*p_est)
    
    # convex relaxation
    lam = 2 * sigma_est * np.sqrt(max(d1,d2) * p_est)
    Z_prox = M_spectral
    for t in range(max_iter):
        Z_new = SVT( Z_prox - eta*(Z_prox-A) * S, lam * eta)
        grad_norm = np.linalg.norm((Z_new-Z_prox)/eta,'fro')/np.linalg.norm(Z_prox,'fro')
        Z_prox = Z_new
        if grad_norm < 1e-8:
            break

    # Debias (only needed if lambda>0)
    Uz, Sz, Vhz = svds_(Z_prox, rk)
    X = Uz@np.diag(np.sqrt(Sz))
    Y = Vhz.T@np.diag(np.sqrt(Sz))
    if lam>0:
        X_d = X.dot(scilinalg.sqrtm(np.eye(rk)+lam/p_est*np.linalg.inv(X.T@X)))
        Y_d = Y.dot(scilinalg.sqrtm(np.eye(rk)+lam/p_est*np.linalg.inv(Y.T@Y)))
        Z_d = X_d@Y_d.T
    else:
        Z_d = X@Y.T
        X_d = X
        Y_d = Y
        
    Z_d = Z_d.real
    X_d = X_d.real
    Y_d = Y_d.real
    
    X_ = X_d@np.linalg.inv(X_d.T@X_d)@X_d.T
    Y_ = Y_d@np.linalg.inv(Y_d.T@Y_d)@Y_d.T
    s_X, s_Y = np.diag(X_).reshape((d1,1)), np.diag(Y_).reshape((d2,1))
    temp = s_X@np.ones((1,d2)) + np.ones((d1,1))@s_Y.T + 2*s_X@s_Y.T
    sigma_est = np.sqrt(np.sum(((A-Z_d)*S)**2)/(d1*d2*p_est))
    var = (sigma_est**2/p_est) * temp
    sigmaS = np.sqrt(var)
    
    return Z_d, X_d, Y_d, sigma_est, sigmaS
    
    
def ALS_solve(M, Ω, r, mu, epsilon=1e-3, max_iterations=100):
    d1, d2 = M.shape
    U = np.random.randn(d1, r)
    V = np.random.randn(d2, r)
    prev_X = U @ V.T
    
    def solve(M, U, Ω):
        V = np.zeros((M.shape[1], r))
        mu_I = mu * np.eye(U.shape[1])
        for j in range(M.shape[1]):
            X1 = Ω[:, j:j+1].copy() * U
            X2 = X1.T @ X1 + mu_I

            V[j] = (np.linalg.pinv(X2, rcond=1e-3) @ X1.T @ (M[:, j:j+1].copy())).T

        return V

    for _ in range(max_iterations):
        U = solve(M.T, V, Ω.T)
        V = solve(M, U, Ω)
        X = U @ V.T
        mean_diff = np.linalg.norm(X - prev_X) / np.linalg.norm(X)

        if mean_diff < epsilon:
            break
        prev_X = X
                    
    sigma_X = np.sqrt(np.sum(((M - X) * Ω)**2) / np.sum(Ω))
    v_X = compute_Sigma_gaussian(X, r, np.mean(Ω), sigma_X)
                    
    return X, sigma_X, v_X


def compute_Sigma_gaussian(Mhat, r, p_observe, sigma_est):
    u,s,vh = scilinalg.svd(Mhat, full_matrices=False)
    U = u[:, :r]
    V = vh[:r, :].T
    d1, d2 = Mhat.shape
    
    U_ = np.diag(U@U.T).reshape((d1,1))
    V_ = np.diag(V@V.T).reshape((d2,1))

    sigmaS = U_@np.ones((1,d2)) + np.ones((d1,1))@V_.T
    sigmaS /= p_observe
    sigmaS = sigma_est * np.sqrt(sigmaS)

    return sigmaS

    



def estimate_M(M_train, S_train, rk, P_hat, q, base):
    if base=='als': 
        M_hat, sigma_est, sigmaS = ALS_solve(M_train, S_train, rk, 0)   
    elif base=='cvx':
        P_inv = 1 / P_hat
        p_est1 = np.mean(S_train)
        # estimated standard deviation
        u_, s_, vh_ = svds_((1/q) * M_train * P_inv, rk)
        M_spec = u_ @ np.diag(s_) @ vh_
        sigma_est_spec = np.sqrt(np.sum(((M_train-M_spec)*S_train)**2)/np.sum(S_train))
        M_hat,X_d_,Y_d_,sigma_est,sigmaS = cvx_mc(M_train, S_train, p_est1, rk, sigma_est_spec, eta=1)

    s_hat = np.sqrt(sigmaS**2 + sigma_est**2)

    return M_hat, s_hat


'''
Conformalized matrix completion

This aims to implement our Algorithm 1
'''

# helper function for conformal prediction
def weighted_quantile(v,prob,w):
    if(len(w)==0):
        w = np.ones(len(v))
    o = np.argsort(v)
    v = v[o]
    w = w[o]
    i = np.where(np.cumsum(w/np.sum(w)) >= prob)
    if(len(i)==0):
        return float('inf') # Can happen with infinite weights
    else:
        return v[np.min(i)]


def cmc_alg(M_obs, S, alpha, q, rk, P, missing_model="homo", base="als"):

    d1, d2 = M_obs.shape

    n_obs = np.sum(S)
    n0 = d1 * d2 - n_obs # number of unobserved entries

    lo, up = np.zeros(n0), np.zeros(n0)
    
    # split the observed matrix into train and calibration sets
    train_selector = np.less(np.random.rand(d1,d2), q)
    S_train = (S * train_selector).astype(dtype=bool)
    S_cal = (S * (1 - train_selector)).astype(dtype=bool)
    n_train = np.sum(S_train)
    n_cal = n_obs - n_train
    assert n_cal == np.sum(S_cal)
    M_train = M_obs * S_train
    M_cal = M_obs * S_cal


    # estimate P_hat from S_train
    P_hat = estimate_P(S_train, q, P, missing_model)

    # estimate M_hat and s_hat from M_train
    M_hat, s_hat = estimate_M(M_train, S_train, rk, P_hat, q, base)

    # compute the score function
    score = np.divide(np.abs(M_cal - M_hat), s_hat)
    
    H_hat = (1 - P_hat) / P_hat
    w_max = np.max( (1 - S) * H_hat )
    
    
    ww = np.zeros(n_cal+1)
    ww[:n_cal] = H_hat[S_cal].ravel()
    ww[n_cal] = w_max

    r_new = 10000
    r = np.append(score[S_cal],r_new)
        
    qvals = weighted_quantile(r, prob = 1-alpha, w = ww)

    lo_mat = M_hat - qvals * s_hat
    up_mat = M_hat + qvals * s_hat
    lo = lo_mat[~S].reshape(-1)
    up = up_mat[~S].reshape(-1)
    
    return lo, up, r, qvals, M_hat, s_hat


'''
This implements the model-based approach to construct prediction intervals for the missing entries. 
'''

def model_based(rk, M_obs, S, M_star, alpha, base):
    if base == "als":
        Mhat, sigma_est, sigmaS = ALS_solve(M_obs, S, rk, 0.0)
        itn = 0
        while (np.linalg.norm((Mhat - M_star) * S) / np.linalg.norm(M_star * S) > 1) and (itn <= 5):
            Mhat, sigma_est, sigmaS = ALS_solve(M_obs, S, rk, 0.0)
            itn += 1
        # s = np.sqrt(sigmaS**2 + sigma_est**2)
    elif base == "cvx":
        p_est = np.mean(S)
        u, s, vh = svds_(M_obs/p_est, rk)
        M_spectral = u @ np.diag(s) @ vh
        sigma_est_spec = np.sqrt(np.sum((( M_obs - M_spectral)*S)**2)/(np.sum(S)))
        Mhat, X_d, Y_d, sigma_est, sigmaS = cvx_mc(M_obs, S, p_est, rk, sigma_est_spec, eta=1)

    s = np.sqrt(sigmaS**2 + sigma_est**2)

    mul = norm.ppf(1-alpha/2)
    lo_uq_mat = Mhat - s * mul
    up_uq_mat = Mhat + s * mul
    lo_uq = lo_uq_mat[~S].reshape(-1)
    up_uq = up_uq_mat[~S].reshape(-1)
    return lo_uq, up_uq, Mhat, s
