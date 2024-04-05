import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import os
import importlib
import download_clean_data as dc
import ipca
import metrics
import kernel_regression as kr
importlib.reload(dc) 
importlib.reload(ipca)
importlib.reload(metrics)
importlib.reload(kr)

############## ---------------------------------------------------- ##############

def split_dataset(x, y, trsh):

    '''
    Split the dataset into training and test sets.

    Parameters:
    - x (list of matrices): List of matrices corresponding to characteristics for each stock.
    - y (list of arrays): List of returns for each time period.
    - trsh (float): Proportion of the dataset to be used for training.

    Returns:
    - list of matrices: Training set of matrices corresponding to characteristics for each stock.
    - list of arrays: Training set of returns for each time period.
    - list of matrices: Test set of matrices corresponding to characteristics for each stock.
    - list of arrays: Test set of returns for each time period.
    '''

    n = int(np.floor(len(y)*trsh))

    x_train=x[:n]
    y_train=y[:n]
    x_test=x[n:]
    y_test=y[n:]
        
    return x_train, y_train, x_test, y_test

############## ---------------------------------------------------- ##############

def val_IPCA(y, x, trsh, gamma_first, max_iter):

    '''
    Compute the out of sample performance of IPCA.

    Parameters:
    - y (list of arrays): List of returns for each time period.
    - x (list of matrices): List of matrices corresponding to characteristics for each stock.
    - trsh (float): Proportion of the dataset to be used for training.
    - gamma_first (matrix): Initial guess for the mapping matrix from characteristics to factors.
    - max_iter (int): Maximum number of iterations.

    Returns:
    - dictionary: Dictionary containing the out of sample R-squared for IPCA.
    '''

    total_R2_dict = {}
    xx_train,yy_train,xx_test,yy_test = split_dataset(x,y, trsh)
    gamma, _ = ipca.ipca(xx_train, yy_train, gamma_first.copy(), max_iter)
    yy_pred = []

    for i in range(len(xx_test)):

        f = ipca.solve_f(yy_test, xx_test, gamma, i)
        yy_pred.append(f)

    total_R2_dict[('IPCA')] = metrics.total_R_squared(yy_test, xx_test, gamma, yy_pred)

    return total_R2_dict

############## ---------------------------------------------------- ##############

def val_IPCA_reg(y, x, trsh, lambda1_v, lambda2_v, gamma_first, max_iter, W_list):

    '''
    Validation of IPCA over a grid of regularization parameters.

    Parameters:
    - y (list of arrays): List of returns for each time period.
    - x (list of matrices): List of matrices corresponding to characteristics for each stock.
    - trsh (float): Proportion of the dataset to be used for training.
    - lambda1_v (list of floats): List of regularization parameters.
    - lambda2_v (list of floats): List of regularization parameters.
    - gamma_first (matrix): Initial guess for the mapping matrix from characteristics to factors.
    - max_iter (int): Maximum number of iterations.
    - W_list (list of matrices): List of weight matrices (here for generality, we used the identity).

    Returns:
    - dictionary: Dictionary containing the out of sample R-squared for IPCA regularized.
    '''

    total_R2_dict = {}
    xx_train,yy_train,xx_test,yy_test = split_dataset(x,y, trsh)

    for lambda1 in lambda1_v:

        for lambda2 in lambda2_v:

            gamma_reg_w, _ = ipca.ipca_reg_w(xx_train, yy_train, gamma_first.copy(), max_iter, lambda1, lambda2, W_list)
            yy_pred = []

            for i in range(len(xx_test)):

                f = ipca.solve_f_reg_w(yy_test, xx_test, gamma_reg_w, i, lambda2, W_list[i])
                yy_pred.append(f)

            total_R2_dict[('IPCA_reg', lambda1, lambda2)] = metrics.total_R_squared(yy_test, xx_test, gamma_reg_w, yy_pred)
            
    return total_R2_dict

############## ---------------------------------------------------- ##############

def val_gaussian(y, x, trsh, lambda1_v, lambda2_v, l_v, N, f_list_input, Omega2, max_iter):

    '''
    Validation of kernel regression with Gaussian kernel over a grid of regularization parameters and kernel parameters.

    Parameters:
    - y (list of arrays): List of returns for each time period.
    - x (list of matrices): List of matrices corresponding to characteristics for each stock.
    - trsh (float): Proportion of the dataset to be used for training.
    - lambda1_v (list of floats): List of regularization parameters.
    - lambda2_v (list of floats): List of regularization parameters.
    - l_v (list of floats): List of kernel parameters.
    - N (int): Number of stocks to keep.
    - f_list_input (list of arrays): Initial value of list of factor (factors after an iteration of IPCA) v
    alues for each time period.
    - Omega2 (matrix): Inverse of weight matrix.
    - max_iter (int): Maximum number of iterations.

    Returns:
    - dictionary: Dictionary containing the out of sample R-squared for kernel regression with Gaussian kernel.
    '''

    total_R2_dict = {}
    xx_train,yy_train,xx_test,yy_test = split_dataset(x,y, trsh)
    Omega_train_inv = np.eye(len(xx_train)*N)

    for l in l_v:

        data2_train = xx_train.copy()
        data2_train = np.array(np.array(data2_train).reshape(len(xx_train)*N,94)) #flatten data, build X
        data2_test = xx_test.copy()
        data2_test = np.array(np.array(data2_test).reshape(len(xx_test)*N,94))
        K_train = kr.K_LR(data2_train, 1, l)
        K_test_train = kr.Kernel(data2_test, data2_train, 1, l)
        
        for lambda1 in lambda1_v:

            for lambda2 in lambda2_v:
                
                f_list, v, _, _, _ = kr.kernel_regression(xx_train, yy_train, f_list_input.copy(), lambda1, lambda2, Omega_train_inv, Omega2, max_iter, N, K_train)
                yy_pred = []
                g_list = []
                c_list = []
    
                for t in range(0,len(xx_test)):

                    g = kr.solve_g_kernel(xx_train, f_list, v, t, K_test_train)
                    G = g@g.T
                    g_list.append(g)
                    c = np.linalg.solve(G+lambda2*Omega2, yy_test[t])
                    c_list.append(c)
                    yy_pred.append(g.T@c)

                total_R2_dict[('Gaussian', lambda1, lambda2, l)] = metrics.total_R_squared_kr_out_os(yy_test, g_list, c_list)
                
    return total_R2_dict

############## ---------------------------------------------------- ##############

def val_rq(y, x, trsh, lambda1_v, lambda2_v, l_v, alphas_v, N, f_list_input, Omega2, max_iter):

    '''
    Validation of kernel regression with Rational Quadratic kernel over a grid of regularization parameters and kernel parameters.

    Parameters:
    - y (list of arrays): List of returns for each time period.
    - x (list of matrices): List of matrices corresponding to characteristics for each stock.
    - trsh (float): Proportion of the dataset to be used for training.
    - lambda1_v (list of floats): List of regularization parameters.
    - lambda2_v (list of floats): List of regularization parameters.
    - l_v (list of floats): List of kernel parameters.
    - alphas_v (list of floats): List of kernel parameters.
    - N (int): Number of stocks to keep.
    - f_list_input (list of arrays): Initial value of list of factor (factors after an iteration of IPCA) v
    alues for each time period.
    - Omega2 (matrix): Inverse of weight matrix.
    - max_iter (int): Maximum number of iterations.

    Returns:
    - dictionary: Dictionary containing the out of sample R-squared for kernel regression with Rational Quadratic kernel.
    '''

    total_R2_dict = {}
    xx_train,yy_train,xx_test,yy_test = split_dataset(x,y, trsh)
    Omega_train_inv = np.eye(len(xx_train)*N)

    for alpha in alphas_v:

        for l in l_v:

            data2_train = xx_train.copy()
            data2_train = np.array(np.array(data2_train).reshape(len(xx_train)*N,94)) #flatten data, build X
            data2_test = xx_test.copy()
            data2_test = np.array(np.array(data2_test).reshape(len(xx_test)*N,94))
            K_train = kr.K_LR(data2_train, 2, l, alpha)
            K_test_train = kr.Kernel(data2_test, data2_train, 2, l, alpha)
            
            for lambda1 in lambda1_v:

                for lambda2 in lambda2_v:

                    f_list, v, _, _, _ = kr.kernel_regression(xx_train, yy_train, f_list_input.copy(), lambda1, lambda2, Omega_train_inv, Omega2, max_iter, N, K_train)
                    yy_pred = []
                    g_list = []
                    c_list = []
        
                    for t in range(0,len(xx_test)):

                        g = kr.solve_g_kernel(xx_train, f_list, v, t, K_test_train)
                        G = g@g.T
                        g_list.append(g)
                        c = np.linalg.solve(G+lambda2*Omega2, yy_test[t])
                        c_list.append(c)
                        yy_pred.append(g.T@c)

                    total_R2_dict[('Rational Quadratic', lambda1, lambda2, l, alpha)] = metrics.total_R_squared_kr_out_os(yy_test, g_list, c_list)
                
    return total_R2_dict

############## ---------------------------------------------------- ##############

def val_linear(y, x, trsh, lambda1_v, lambda2_v, N, f_list_input, Omega2, max_iter):

    '''
    Validation of kernel regression with Linear kernel over a grid of regularization parameters.

    Parameters:
    - y (list of arrays): List of returns for each time period.
    - x (list of matrices): List of matrices corresponding to characteristics for each stock.
    - trsh (float): Proportion of the dataset to be used for training.
    - lambda1_v (list of floats): List of regularization parameters.
    - lambda2_v (list of floats): List of regularization parameters.
    - N (int): Number of stocks.
    - f_list_input (list of arrays): Initial value of list of factor (factors after an iteration of IPCA) 
    values for each time period.
    - Omega2 (matrix): Inverse of weight matrix.
    - max_iter (int): Maximum number of iterations.

    Returns:
    - dictionary: Dictionary containing the out of sample R-squared for kernel regression with Linear kernel.
    '''

    total_R2_dict = {}   
    xx_train,yy_train,xx_test,yy_test = split_dataset(x,y, trsh)
    Omega_train_inv = np.eye(len(xx_train)*N)
    data2_train = xx_train.copy()
    data2_train = np.array(np.array(data2_train).reshape(len(xx_train)*N,94)) 
    data2_test = xx_test.copy()
    data2_test = np.array(np.array(data2_test).reshape(len(xx_test)*N,94))
    K_train = kr.K_LR(data2_train, 0, 1)
    K_test_train = kr.Kernel(data2_test, data2_train, 0, 1)
    g_list = []
    
    for lambda1 in lambda1_v:

        for lambda2 in lambda2_v:

            f_list, v, _, _, _ = kr.kernel_regression(xx_train, yy_train, f_list_input.copy(), lambda1, lambda2, Omega_train_inv, Omega2, max_iter, N, K_train)
            yy_pred = []
            g_list = []
            c_list = []

            for t in range(0,len(xx_test)):

                g = kr.solve_g_kernel(xx_train, f_list, v, t, K_test_train)
                G = g@g.T
                g_list.append(g)
                c = np.linalg.solve(G+lambda2*Omega2, yy_test[t])
                c_list.append(c)
                yy_pred.append(g.T@c)

            total_R2_dict[('Linear', lambda1, lambda2)] = metrics.total_R_squared_kr_out_os(yy_test, g_list, c_list)
                
    return total_R2_dict

############## ---------------------------------------------------- ##############

def surface_IPCA(dictionary, lambdas1, lambdas2):

    '''
    Plot the surface of the out of sample R-squared for IPCA over a grid of regularization parameters.

    Parameters:
    - dictionary (dictionary): Dictionary containing the out of sample R-squared for IPCA.
    - lambdas1 (list of floats): List of regularization parameters.
    - lambdas2 (list of floats): List of regularization parameters.

    Returns:
    - matrix: Matrix of out of sample R-squared for IPCA.
    '''
    
    z = np.zeros((len(lambdas1), len(lambdas2)))

    for l1, lambda1 in enumerate(lambdas1):

        for l2, lambda2 in enumerate(lambdas2):

            for k,v in dictionary.items():

                if k[1] == lambda1 and k[2] == lambda2:

                    z[l1, l2] = v
    
    return z

############## ---------------------------------------------------- ##############

def surface_gaussian(dictionary, lambdas1, lambdas2):

    '''
    Plot the surface of the out of sample R-squared for kernel regression with Gaussian 
    kernel over a grid of regularization parameters.

    Parameters:
    - dictionary (dictionary): Dictionary containing the out of sample R-squared for kernel regression with Gaussian kernel.
    - lambdas1 (list of floats): List of regularization parameters.
    - lambdas2 (list of floats): List of regularization parameters.

    Returns:
    - matrix: Matrix of out of sample R-squared for kernel regression with Gaussian kernel.
    '''
    
    z = np.zeros((len(lambdas1), len(lambdas2)))

    for l1, lambda1 in enumerate(lambdas1):

        for l2, lambda2 in enumerate(lambdas2):

            for k,v in dictionary.items():

                if k[1] == lambda1 and k[2] == lambda2 and k[3]==20:

                    z[l1, l2] = v
    
    return z

############## ---------------------------------------------------- ##############

def surface_rq(dictionary, lambdas1, lambdas2):

    '''
    Plot the surface of the out of sample R-squared for kernel regression with Rational Quadratic.

    Parameters:
    - dictionary (dictionary): Dictionary containing the out of sample R-squared for kernel regression with Rational Quadratic.
    - lambdas1 (list of floats): List of regularization parameters.
    - lambdas2 (list of floats): List of regularization parameters.

    Returns:
    - matrix: Matrix of out of sample R-squared for kernel regression with Rational Quadratic.
    '''
    
    z = np.zeros((len(lambdas1), len(lambdas2)))

    for l1, lambda1 in enumerate(lambdas1):

        for l2, lambda2 in enumerate(lambdas2):

            for k,v in dictionary.items():

                if k[1] == lambda1 and k[2] == lambda2 and k[3]==20 and k[4]==20:

                    z[l1, l2] = v
    
    return z

