import warnings
from sklearn.cluster import KMeans
import numpy as np
import tqdm
import os
import glob
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import func
import scipy.stats
from scipy.linalg import qr, solve, lstsq
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy import *
import library as lib
import random as rd

def replace_nan(X, to_print=False):
    X_new= reshape(X,X.shape[0]*X.shape[1])
    is_nan = np.where(isnan(X_new))[0]
    if len(is_nan)!=0:
        for index in is_nan:
            if to_print:
                print("indexNan=",index)
            neighbors = concatenate((X_new[index-2:index-1],X_new[index+1:index+2]))
            neighbors_not_nan = neighbors[[not isnan(neighbors)[k] for k in range(len(neighbors))]]
            X_new[is_nan]=mean(neighbors_not_nan)
    X_filled =X_new.reshape((X.shape[0],X.shape[1]))
    return(X_filled)


def is_square(arr):
    """
    Determines whether an array is square or not.

    :param ndarray arr: numpy array

    :return: condition
    :rtype: bool
    """
    shape = arr.shape
    if len(shape) == 2 and (shape[0] == shape[1]):
        return True
    return False

def least_squares_covariance(A, B, W):
    """
    Least squares solution in the presence of known covariance.
    This function is known as lscov() in MATLAB.

    :param ndarray A: Design matrix
    :param ndarray b: Observations (vector of phase values)
    :param ndarray v: Covariances (vector of weights)

    :return: solution
    :rtype: ndarray
    """

    W = np.sqrt(np.diag(W.squeeze()))
    Aw = np.dot(W,A)
    Bw = np.dot(W,B)
    X = np.linalg.lstsq(Aw, Bw)
    #print("Lscov.shape=",X[0].shape)
    return(array(X[0]))

def init_EM_latent_class_regression(X,Y,K,method,clust1=[0]):
    # size
    n=size(Y,0)
    nb_features = X.shape[1]
    depth = Y.shape[1]
    
    # clustering method
    if method == "kmeans":
        clust = KMeans(n_clusters=K, random_state=0).fit_predict(hstack((X[:,1:X.shape[1]],Y))) # see why its better than hstack((X[:,1:X.shape[1]],Y))
        #hstack((array([X[:,1]]).T,Y))
    elif method == "random":
        if len(clust1)>1:
            clust=clust1
        else:
            clust=np.array([np.random.randint(0,K) for i in range(n)]);
    elif method== "init_weight":
        clust=clust1
    # initialize lambda
    hist_k = plt.hist(clust,K,density=True);
    plt.close()
    width_bins = hist_k[1][1] -hist_k[1][0] 
    nb_k = hist_k[0]*width_bins * len(X)

    lambda_init=zeros(K)
    for k in range(K):
        lambda_init[k]=nb_k[k]/n

    Beta_init  =array(zeros((K,nb_features,depth)))
    Sigma_init =array(zeros((K,depth,depth)))

    # initialize Beta and Sigma
    for k in range(0,K): 
        Beta_init[k,:,:] = np.linalg.lstsq(X[np.where(clust==k)[0],:],Y[np.where(clust==k)[0],:])[0]
        Sigma_init[k,:,:] = cov((Y[np.where(clust==k)[0],:]-X[np.where(clust==k)[0],:]@(Beta_init[k,:,:].reshape((nb_features,depth)))).T)

    return(lambda_init,Beta_init,Sigma_init)

def EM_with_init(inputs):
    X_orth,Y_orth,K,method,iter_EM,clust = inputs
    lambda_init,Beta_init,Sigma_init = init_EM_latent_class_regression(X_orth,Y_orth,K,method,clust)
    log_lik,lambda_hat,Beta_hat,Sigma_hat,Y_hat,pi_hat,Z_hat=EM_latent_class_regression(X_orth,Y_orth,lambda_init,Beta_init,Sigma_init,iter_EM)
    return(pi_hat,log_lik)

def stop_condition_EM(l_new,l_old,epsilon):
    return(abs(l_new-l_old)<epsilon)

def latent_class_regression(X_test,Y_test,lambda_hat,Beta_hat,Sigma_hat,reg_type,K):
    # parameters
    n = X_test.shape[0]
    p = Y_test.shape[1];

    # compute pi_test
    sum_prob_tmp=zeros((n,1));
    pi_hat=zeros((n,K));
    Z_hat=zeros((n,1));
    for k in range(K):
        sum_prob_tmp[:,0] = [sum_prob_tmp[j]+lambda_hat[k]*multivariate_normal.pdf(Y_test[j,:],mean=X_test[j,:]@Beta_hat[k,:,:],cov=Sigma_hat[k,:,:]) for j in range(n)]
        #sum_prob_tmp = sum_prob_tmp + array([lambda_hat[k]*multi.pdf(Y_test)]).T
        
    for k in range(K):
        pi_hat[:,k]=[lambda_hat[k]*multivariate_normal.pdf(Y_test[j,:],mean=X_test[j,:]@Beta_hat[k,:],cov=Sigma_hat[k,:,:]) for j in range(n)]/sum_prob_tmp[:,0]
        
    for i_n in range(n):
        Z_hat[i_n,:]=np.where(pi_hat[i_n,:]==max(pi_hat[i_n,:]))[0]

    # compute Y hat
    Y_hat=zeros((n,p));
    if reg_type == 'fuzzy':
        for i_n in range(n):
            for k in range(K):
                Y_hat[i_n,:]=Y_hat[i_n,:]+pi_hat[i_n,k]*X_test[i_n,:]@Beta_hat[k,:,:]

    elif reg_type == 'natural':
        for i_n in range(n):
            for k in range(K):
                Y_hat[i_n,:]=X_test[i_n,:]@Beta_hat[int(Z_hat[i_n]),:,:]
    return(Y_hat,pi_hat,Z_hat)

def EM_latent_class_regression(X,Y,lambda_init,Beta_init,Sigma_init,iter_EM,print_=False):  
    if print_:
        print("Beta_init.shape=",Beta_init[0].shape)
    # Dimensions
    K = len(Beta_init)
    n,p = Y.shape
    nb_features = X.shape[1]
    depth = Y.shape[1]
    # Initialization
    lambda_hat = lambda_init
    Beta_hat   = Beta_init
    Sigma_hat  = Sigma_init

    # if condition=0, we stop the EM
    stop_cound = 0
    j = 0
    log_lik =[]
    pi_hat = zeros((n,K))
    Z_hat = zeros((1,n))
    if print_:
        print("X.shape=",X.shape)
        print("Y.shape=",Y.shape)
    # loop
    for j in range(iter_EM):
        # display
        if print_:
            display(["***EM ITERATION:",j,"***"]);

        # E-step
        sum_prob=zeros((n,1))
        if print_:
            print("compute multivariate")
            print("lambda_hat",lambda_hat)
        for k in range(K):
            sum_prob[:,0] = [sum_prob[j,0]+ lambda_hat[k]*multivariate_normal.pdf(Y[j,:],mean=X[j,:]@Beta_hat[k,:,:],cov=Sigma_hat[k,:,:]) for j in range(n)]

        for k in range(K):
            pi_hat[:,k]= [lambda_hat[k]*multivariate_normal.pdf(Y[j,:],mean=X[j,:]@Beta_hat[k,:],cov=Sigma_hat[k,:,:]) for j in range(n)]/sum_prob[:,0]

        for i_n in range(n):
            most_likely = np.where(pi_hat[i_n,:]==max(pi_hat[i_n,:]))[0]
            Z_hat[:,i_n]= most_likely
        lambda_hat = sum(pi_hat,0)/n
        
        # M-step
        for k in range(K):
            #print("compute lscov")
            Beta_hat[k,:,:]=least_squares_covariance(X,Y,pi_hat[:,k].T)
            Sigma_tmp=array(zeros((depth,depth)))
            for i_n in range(n):
                Sigma_tmp=Sigma_tmp+pi_hat[i_n,k]*reshape((Y[i_n,:]-X[i_n,:]@Beta_hat[k,:,:]),(depth,1))@reshape(((Y[i_n,:]-X[i_n,:]@Beta_hat[k,:,:]).T),(1,depth))
            Sigma_hat[k,:,:]=Sigma_tmp/sum(pi_hat[:,k],0)
            #print("Sigma_hat=",Sigma_hat)
            

        # log-likelihood
        log_lik_tmp=0
        for i_n in range(n):
            sum_tmp=0
            for k in range(K):
                MU = X[i_n,:]@Beta_hat[k,:,:]
                SIGMA=Sigma_hat[k,:,:];
                #print("SIGMA",SIGMA)
                sum_tmp = sum_tmp+lambda_hat[k]*multivariate_normal.pdf(Y[i_n,:],mean=MU,cov=SIGMA)
                #print("sum_tmp =", sum_tmp )
            log_lik_tmp=log_lik_tmp+log(sum_tmp)
        # stock log_likelihood
        
        log_lik.append(log_lik_tmp)

        # compute stop condition
        if j>0:
            stop_cound=stop_condition_EM(log_lik[j],log_lik[j-1],10^(-1))
        j=j+1

    # generate Y_hat
    Y_hat=zeros((n,p));
    for i_n in range(n):
        for k in range(K):
            Y_hat[i_n,:]=Y_hat[i_n,:]+pi_hat[i_n,k]*array(X[i_n,:]@Beta_hat[k,:,:]);
    return(log_lik,lambda_hat,Beta_hat,Sigma_hat,Y_hat,pi_hat,Z_hat)

def double_acp_target_feature(X,Y,to_plot=False,nb_factor_max =15,nb_factor_max_target = 5):
    #nombre d'observations
    n = X.shape[0]
    #nombre de variables
    p = X.shape[1]
    # Instanciation
    sc_features = StandardScaler()
    sc_target   = StandardScaler()
    # Transformation – centrage-réduction
    Z  = sc_features.fit_transform(X)
    YY = sc_target.fit_transform(Y)

    #instanciation
    acp = PCA(svd_solver='full')
    print(acp)
    acp_target = PCA(svd_solver='full')

    coord = acp.fit_transform(Z)
    target = acp_target.fit_transform(YY)

    #nb of component computed
    print("Number of acp components features= ", acp.n_components_) 
    print("Number of acp components target= ", acp_target.n_components_) 

    #variance explained
    eigval = (n-1)/n*acp.explained_variance_
    eigval_target = (n-1)/n*acp_target.explained_variance_

    #percentage of variance explained
    cumsum_var_explained= cumsum(acp.explained_variance_ratio_)
    cumsum_var_explained_target= cumsum(acp_target.explained_variance_ratio_)
    print("cumsum variance explained= ",cumsum_var_explained[0:nb_factor_max-1])
    print("cumsum variance explained target= ",cumsum_var_explained_target[0:nb_factor_max_target-1])
    if to_plot :
        #scree plot
        plt.figure(figsize=(15,10))

        plt.subplot(221)
        plt.plot(np.arange(1,nb_factor_max),eigval[0:nb_factor_max-1])
        plt.scatter(np.arange(1,nb_factor_max),eigval[0:nb_factor_max-1])
        plt.title("Scree plot")
        plt.ylabel("Eigen values")
        plt.xlabel("Factor number")

        plt.subplot(222)
        plt.plot(np.arange(1,nb_factor_max),cumsum_var_explained[0:nb_factor_max-1])
        plt.scatter(np.arange(1,nb_factor_max),cumsum_var_explained[0:nb_factor_max-1])
        plt.title("Variance plot")
        plt.ylabel("Total Variance explained")
        plt.xlabel("Factor number")

        plt.subplot(223)
        plt.plot(np.arange(1,nb_factor_max_target),eigval_target[0:nb_factor_max_target-1])
        plt.scatter(np.arange(1,nb_factor_max_target),eigval_target[0:nb_factor_max_target-1])
        plt.title("Scree plot")
        plt.ylabel("Eigen values target")
        plt.xlabel("Factor number")

        plt.subplot(224)
        plt.plot(np.arange(1,nb_factor_max_target),cumsum_var_explained_target[0:nb_factor_max_target-1])
        plt.scatter(np.arange(1,nb_factor_max_target),cumsum_var_explained_target[0:nb_factor_max_target-1])
        plt.title("Variance plot")
        plt.ylabel("Total Variance explained")
        plt.xlabel("Factor number")

        plt.show()
    return(cumsum_var_explained,cumsum_var_explained_target)

def BIC_calculation(inputs_):

    X,Y,K,method,iter_EM,cumsum_var_explained_target,cumsum_var_explained,var_feature,var_target,clust = inputs_
    sc_features = StandardScaler()
    sc_target   = StandardScaler()
    # Transformation – centrage-réduction
    X  = sc_features.fit_transform(X)
    Y = sc_target.fit_transform(Y)
    nb_component_target = np.where(cumsum_var_explained_target>var_target)[0][0]
    nb_component_features = np.where(cumsum_var_explained>var_feature)[0][0]
    acp_features = PCA(svd_solver='full',n_components =nb_component_features+1)
    acp_target = PCA(svd_solver='full',n_components =nb_component_target+1)
    X_orth = acp_features.fit_transform(X)
    Y_orth = acp_target.fit_transform(Y)
    BIC_,nb_parameters=BIC_calculation_orth(X_orth,Y_orth,K,method,iter_EM,clust)
    return(BIC_,nb_parameters)

def BIC_calculation_orth(X_orth,Y_orth,K,method,iter_EM,clust):
    lambda_init,Beta_init,Sigma_init = init_EM_latent_class_regression(X_orth,Y_orth,K,method,clust)
    log_lik,lambda_hat,Beta_hat,Sigma_hat,Y_hat,pi_hat,Z_hat=EM_latent_class_regression(X_orth,Y_orth,lambda_init,Beta_init,Sigma_init,iter_EM)
    sample_size = X_orth.shape[0]
    nb_parameters =Sigma_init.shape[0]*Sigma_init.shape[1]*Sigma_init.shape[2] + Beta_hat.shape[0]*Beta_hat.shape[1]*Beta_hat.shape[2]
    BIC_ = -2*log_lik[-1] + nb_parameters*log(sample_size)
    return(BIC_,nb_parameters)

#def EM_with_init(inputs):
#    X_orth,Y_orth,K,method,iter_EM = inputs
#    lambda_init,Beta_init,Sigma_init = init_EM_latent_class_regression(X_orth,Y_orth,K,method)
#    #log_lik,lambda_hat,Beta_hat,Sigma_hat,Y_hat,pi_hat,Z_hat=EM_latent_class_regression(X_orth,Y_orth,lambda_init,Beta_init,Sigma_init,iter_EM#)
    return(pi_hat)
    
    
