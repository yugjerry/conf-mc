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

# one-bit MC
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
    U,S,Vh = np.linalg.svd(B.reshape((d1,d2)))

    s2 = euclidean_proj_l1ball(S,radius)

    B_proj = U@np.diag(s2)@Vh
    
#     B_proj[B_proj < 0] = 0.01
    
    return B_proj.reshape((d1*d2,))


def logis(data, theta, beta, q=1, tau = 0.3, tol = 0.1):
    N=theta.shape[0]
    J=beta.shape[0]
    o1 = np.ones(J).reshape((1,J))
    o2 = np.ones(N).reshape((N,1))
    temp = theta @ o1 + o2 @ beta.T #M matrix at initial values
    JML0 = -float('inf') #Initialize step 0 log likelihood as negative infinity
    temp1 = np.log(1+np.exp(-temp))
    temp2 = np.log(1-q+np.exp(-temp))
    JML = np.sum(-temp1 + (1-data)*temp2) #Initialize step 1 log likelihood at initial values of beta and theta

    while(JML - JML0 > tol):
        JML0 = JML
        prob = q/(1+np.exp(-temp))
        grad = np.exp(-temp)*(1/(1+np.exp(-temp)) - (1-data)/(1-q+np.exp(-temp)))
        theta = theta + tau * np.mean(grad, axis=1).reshape((N,1)) #update for theta estimates
#         theta = theta - np.mean(theta) #identifiability constraint
        temp = theta @ o1 + o2 @ beta.T
        temp1 = np.log(1+np.exp(-temp))
        temp2 = np.log(1-q+np.exp(-temp))
        
        prob = q/(1+np.exp(-temp))
        grad = np.exp(-temp)*(1/(1+np.exp(-temp)) - (1-data)/(1-q+np.exp(-temp)))
        beta = beta + tau * np.mean(grad, axis=0).reshape((J,1)) #update for beta estimates
#         beta = beta-np.mean(theta)
        temp = theta @ o1 + o2 @ beta.T
        temp1 = np.log(1+np.exp(-temp))
        temp2 = np.log(1-q+np.exp(-temp))
        JML = np.sum(-temp1 + (1-data)*temp2)
#         print(JML)
  
    return theta, beta


# missing mechanism
def link_logis(u,v):
    d1 = u.shape[0]
    d2 = v.shape[0]
    P = np.zeros((d1,d2))
    for i in range(d1):
        for j in range(d2):
            P[i,j] = np.exp(u[i]+v[j]) / (1+np.exp(u[i]+v[j]))
    return P

# generate (low-rank) matrix
def mat_gen(d1,d2,P,k,M_mean):

    U = np.random.normal(0,1,d1*k).reshape((d1,k))
    V = np.random.normal(0,1,d2*k).reshape((d2,k))
#     for i in range(5,k):
#         U[(10*i):(10*(i+1)),i] = U[(10*i):(10*(i+1)),i] + 10
    U = scilinalg.orth(U)
    V = scilinalg.orth(V)
    s = np.ones(k)
    
    M_star = U@np.diag(s)@V.T
    M_star = M_star/np.mean(np.abs(M_star)) * M_mean
    
    S = np.less(np.random.rand(d1,d2), P)
    M = M_star * S
    
    return M_star, M, S

def mat_gen_mis(d1,d2,P,k,M_mean):

    U = np.random.standard_cauchy(d1*k).reshape((d1,k))
    V = np.random.standard_cauchy(d2*k).reshape((d2,k))
    
    U = scilinalg.orth(U)
    V = scilinalg.orth(V)
    s = np.ones(k)
    
    M_star = U@np.diag(s)@V.T
    M_star = M_star/np.mean(np.abs(M_star)) * M_mean
    
    S = np.less(np.random.rand(d1,d2), P)
    M = M_star * S
    
    return M_star, M, S

def mat_gen_mis_(d1,d2,P,k,M_mean):

#     U1 = np.random.standard_cauchy(d1*5).reshape((d1,5))
#     V1 = np.random.standard_cauchy(d2*5).reshape((d2,5))
#     U2 = np.random.standard_cauchy(d1*(k-5)).reshape((d1,k-5))
#     V2 = np.random.standard_cauchy(d2*(k-5)).reshape((d2,k-5))
#     U = np.concatenate((U1,U2), axis=1)
#     V = np.concatenate((V1,V2), axis=1)

    U = np.random.normal(0,1,d1*k).reshape((d1,k))
    V = np.random.normal(0,1,d2*k).reshape((d2,k))
#     U = np.random.standard_cauchy(d1*k).reshape((d1,k))
#     V = np.random.standard_cauchy(d2*k).reshape((d2,k))
    
    U = scilinalg.orth(U)
    V = scilinalg.orth(V)
#     s1 = np.ones(5)
#     eps = 0.8
#     s2 = eps * np.ones(k-5)
#     s = np.append(s1, s2)
    s = np.ones(k)
    
    M_star = U@np.diag(s)@V.T
    M_star = M_star/np.mean(np.abs(M_star)) * M_mean
    
    S = np.less(np.random.rand(d1,d2), P)
    M = M_star * S
    
    return M_star, M, S

# recovery via low-rank svd
def low_rk_svd(M_,P_inv,k):
    u_k, s_k, vt_k = scilinalg.svd(M_ * P_inv, full_matrices=False)
    u_k = u_k[:,:k]
    s_k = s_k[:k]
    vt_k = vt_k[:k,:]
    M_hat = u_k @ np.diag(s_k) @ vt_k
    return M_hat

def svds_(M_,k):
    u_k, s_k, vt_k = scilinalg.svd(M_, full_matrices=False)
    u_k = u_k[:,:k]
    s_k = s_k[:k]
    vt_k = vt_k[:k,:]
    return u_k, s_k, vt_k

# logistic regression to estimate P
def logis_reg(Y):
    d1, d2 = Y.shape[0], Y.shape[1]
    N = d1*d2
    X = np.zeros((N, d1+d2))
    y = np.zeros(N)
    
    # create data arrays
    idx = 0
    for i in range(d1):
        for j in range(d2):
            X[idx, i] = 1
            X[idx, d1+j] = 1
            y[idx] = Y[i,j]
            idx+=1
    
    clf = LogisticRegression(random_state=0, penalty='none', fit_intercept=False).fit(X, y)
    coef = clf.coef_.reshape((d1+d2,))
    a = coef[:d1]
    b = coef[d1:]
    
    return a, b

# nonconvex matrix completion
def SVT(X, tau):
    U,S,Vh = np.linalg.svd(X, full_matrices=False)
    S = np.maximum(S-tau, np.zeros(len(S)))
    Z = U@np.diag(S)@Vh
    return Z

def cvx_mc(A, S, p_est, rk, sigma_est, lam=0, eta=0.1, max_iter=1000):
    # eta: learning rate
    # max_iter: max number of iterations
    # lam: regularization; default = 0 for nonconvex approach
    d1, d2 = A.shape
    u, s, vh = svds_(A/p_est, rk)
    v = vh.T
    M_spectral = u@np.diag(s)@vh
    X = u@np.diag(np.sqrt(s))
    Y = v@np.diag(np.sqrt(s))
    
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
    lam = 2*sigma_est*np.sqrt(max(d1,d2)*p_est)
    Z_prox = M_spectral
    for t in range(max_iter):
        Z_new = SVT(Z_prox-eta*(Z_prox-A)*S,lam*eta)
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

# def solve(M_, U_, Ω_, r, mu):
#     V_ = np.zeros((M_.shape[1], r))
#     mu_I = mu * np.eye(U_.shape[1])
#     for j in range(M_.shape[1]):
#         X1 = Ω_[:, j:j+1].copy() * U_
#         X2 = X1.T @ X1 + mu_I
#         V_[j] = (np.linalg.pinv(X2) @ X1.T @ (M_[:, j:j+1].copy())).T
#         # print(M[:, j:j+1].shape)
#         # V_[j] = np.linalg.solve(X2, X1.T @ (M[:, j:j+1].copy())).reshape(-1)
#     return V_
    
    
def ALS_solve(M, Ω, r, mu, epsilon=1e-3, max_iterations=100, debug = False):
    
    #logger = logging.getLogger(__name__)
    n1, n2 = M.shape
    U = np.random.randn(n1, r)
    V = np.random.randn(n2, r)
    prev_X = np.dot(U, V.T)
    
    def solve(M, U, Ω):
        V = np.zeros((M.shape[1], r))
        mu_I = mu * np.eye(U.shape[1])
        for j in range(M.shape[1]):
            X1 = Ω[:, j:j+1].copy() * U
            X2 = X1.T @ X1 + mu_I
#             V[j] = (np.linalg.pinv(X2, rcond=1e-4) @ X1.T @ (M[:, j:j+1].copy())).T
            V[j] = (np.linalg.pinv(X2, rcond=1e-3) @ X1.T @ (M[:, j:j+1].copy())).T
            #print(M[:, j:j+1].shape)
#             V[j] = np.linalg.solve(X2, X1.T @ (M[:, j:j+1].copy())).reshape(-1)
        return V

    for _ in range(max_iterations):
        U = solve(M.T, V, Ω.T)
        V = solve(M, U, Ω)
        X = np.dot(U, V.T)
        mean_diff = np.linalg.norm(X - prev_X) / np.linalg.norm(X)
        #if _ % 1 == 0:
        #    logger.info("Iteration: %i; Mean diff: %.4f" % (_ + 1, mean_diff))
        if (debug):
            print("Iteration: %i; Mean diff: %.4f" % (_ + 1, mean_diff))
        
        if mean_diff < epsilon:
            break
        prev_X = X
                    
    sigma_X = np.sqrt(np.sum(((M-X)*Ω)**2)/np.sum(Ω))
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

def compute_Sigma_adaptive(Mhat, E, r, p_observe):
    u,s,vh = np.linalg.svd(Mhat, full_matrices=False)
    U = u[:, :r]
    V = vh[:r, :].T

    sigmaS = ((U.dot(U.T))**2).dot(E**2) + (E**2).dot((V.dot(V.T))**2)
    sigmaS /= (p_observe**2)
    sigmaS = np.sqrt(sigmaS)

    return sigmaS

# defined functions for conformal prediction
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
    
# conformalized matrix completion
def CP_split_svd(M0,ind,alpha,P,rk,wtd,het,w,oracle,base,kap,verbose=False):
    # weights are used for computing quantiles for the prediction interval
    d1, d2 = M0.shape
    S = np.zeros((d1,d2))
    ind_nonzero = np.transpose(np.nonzero(M0))
    S[ind_nonzero[:,0],ind_nonzero[:,1]] = 1
    
    N = d1*d2
    n = int(np.sum(S))
    n0 = ind.shape[0]

    lo = np.zeros(n0)
    up = np.zeros(n0)
    
    # split
    p_split = 0.8
    a = np.random.rand(ind_nonzero.shape[0])
    mask = a<=p_split
    ind_train = ind_nonzero[mask,:]    # training set
    ind_calib = ind_nonzero[~mask,:]   # calibration set
    n_train = np.sum(mask)
    n_calib = n-n_train
    M_calib = np.copy(M0)
    M_train = np.copy(M0)
    M_train[ind_calib[:,0],ind_calib[:,1]] = 0
    S_train = M_train!=0
    
    # apply mc algorithm 
    # estimate P
    if ((oracle==False) and (wtd==True)):
        if het=='homo':
            P_hat_ = (1/p_split)*np.mean(S_train)*np.ones(d1*d2).reshape((d1,d2))
            P_lo_ = P_hat_
        if het == 'logis1':
            yy = 2*(S_train-0.5).ravel()
            x_init = np.zeros(N)
            idx = range(N)
            const   = 1.0
            radius  = const * np.sqrt(d1*d2*2)
            f_loc = lambda x: f_(x,p_split)
            fprime_loc = lambda x: fprime(x,p_split)
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
            k_p = 2
            M_d = U[:,:k_p]@np.diag(s_hat[:k_p])@Vh[:k_p,:]
            P_hat_ = (1/p_split)*f_(M_d,p_split).reshape((d1,d2))
            P_lo_ = P_hat_
            
        if het=='rank1':
            u_hat, s_hat, vt_hat = svds_(S_train,1)
            P_hat_ = (1/p_split)*u_hat @ np.diag(s_hat) @ vt_hat
            P_lo_ = P_hat_
            
        if het=='logis':
            y_mat = S_train.astype(int)
            theta_int = np.random.uniform(-1,1,d1).reshape((d1,1))
            theta_int = theta_int - np.mean(theta_int)
            beta_int = np.random.uniform(-1,1,d2).reshape((d2,1))
            theta, beta = logis(y_mat, theta_int, beta_int, p_split)
            M_d = theta @ np.ones(d2).reshape(1,d2) + np.ones(d1).reshape(d1,1) @ beta.T
            P_hat_ = 1/(1+np.exp(-M_d))
            P_lo_ = P_hat_
            
    elif oracle==True:
        P_hat_ = P
        P_lo_ = P
    else:
    	P_hat_ = (1/p_split)*np.mean(S_train)*np.ones(d1*d2).reshape((d1,d2))

    # apply mc algorithm
    # spectral initialization
    P_inv = 1/P_hat_
    p_est1 = np.mean(S_train)
    # estimated standard deviation
    u_,s_,vh_ = svds_((1/p_split)*M_train*P_inv, rk)
    M_spec = u_@np.diag(s_)@vh_
    sigma_est_spec = np.sqrt(np.sum(((M_train-M_spec)*S_train)**2)/np.sum(S_train))
    
    if base=='als':
        penalty = 0.0
        M_hat,sigma_est,sigmaS = ALS_solve(M_train, S_train, rk, penalty)

        s_hat = np.sqrt(sigmaS**2+kap*sigma_est**2)
    if base=='cvx':
        M_hat,X_d_,Y_d_,sigma_est,sigmaS = cvx_mc(M_train, S_train, p_est1, rk, sigma_est_spec, eta=1)
        s_hat = np.sqrt(sigmaS**2+kap*sigma_est**2)
    if base=='svd':
        M_hat = low_rk_svd(M_train,P_inv,rk)
        s_hat = np.abs(M_hat)
    
    dist = np.divide(np.abs(M_calib-M_hat), s_hat)
    # Check the weights
    w_max = np.max((1-P_lo_)/P_lo_)
    if((len(w)==0) & (wtd==False)):
        ww = np.ones(n_calib+1)
    elif((len(w)==0) & (wtd==True)):
        ww = np.zeros(n_calib+1)
        for j in range(n_calib):
            pi = P_lo_[ind_calib[j,0],ind_calib[j,1]]
            ww[j] = (1-pi)/pi
        ww[n_calib] = w_max
    else:
        ww = w
        
    r_new = 10000
    r = np.append(dist[ind_calib[:,0],ind_calib[:,1]],r_new)
        
    qvals = weighted_quantile(r,prob=1-alpha,w=ww)

    lo_mat = M_hat - qvals * s_hat
    up_mat = M_hat + qvals * s_hat
    lo = lo_mat[S==0].reshape(-1)
    up = up_mat[S==0].reshape(-1)
    
    return lo, up, r, qvals, M_hat, s_hat
