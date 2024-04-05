import numpy as np
import pandas as pd

############## ---------------------------------------------------- ##############

def Kernel(x, xt, tk, l=1, alpha=1):

    '''
    Compute the kernel matrix.

    Parameters:
    - x (matrix): Matrix of data points.
    - xt (matrix): Matrix of data points.
    - tk (int): Type of kernel.
    - l (float): parameter for the gaussian and rational quadratic kernels.
    - alpha (float): parameter for the rational quadratic kernel.

    Returns:
    - matrix: Kernel matrix.
    '''

    #Linear
    if tk==0:
        return x@xt.T 

    #Gaussian
    if tk==1:
        K=np.zeros([x.shape[0], xt.shape[0]])
        for i in range(0,x.shape[0]):
            K[i]=np.exp(-np.sum((xt-x[i])**2, axis=1)/(2*l**2))
        return K
    
    #Rational Quadratic
    if tk==2:
        K=np.zeros([x.shape[0], xt.shape[0]])
        for i in range(0,x.shape[0]):
            K[i]=(1+np.sum((xt-x[i])**2, axis=1)/(2*alpha*l**2))**(-alpha)
        return K

############## ---------------------------------------------------- ##############
    
def K_LR(data2, tk, l=1, alpha=1):

    '''
    Compute the kernel matrix between the data points and itself.

    Parameters:
    - data2 (matrix): Matrix of data points.
    - tk (int): Type of kernel.
    - l (float): parameter for the gaussian and rational quadratic kernels.
    - alpha (float): parameter for the rational quadratic kernel.

    Returns:
    - matrix: Kernel matrix.
    '''

    return Kernel(data2, data2, tk, l, alpha)

############## ---------------------------------------------------- ##############
    
def solve_v_kernel (ret, lambda1, data, f_list, N, Omega, K):
    
    '''
    Solve for the vector v in the kernel regression.

    Parameters:
    - ret (list of arrays): List of returns for each time period.
    - lambda1 (float): Regularization parameter.
    - data (list of matrices): List of matrices corresponding to characteristics for each stock.
    - f_list (list of arrays): List of factor values for each time period.
    - N (int): Number of stocks.
    - Omega (matrix): Inverse of the weight matrix.
    - K (matrix): Kernel matrix.

    Returns:
    - array: Vector v.
    - matrix: Matrix Q.
    '''

    T=len(data)
    A=np.array(f_list)@np.array(f_list).T
    Q=np.zeros([(T)*N, (T)*N])
    R=np.array(ret).flatten()

    for t in range(0,T):
        for s in range(0,T):
            Q[t*100:(t+1)*100,s*100:(s+1)*100]=A[t,s]*K[t*100:(t+1)*100,s*100:(s+1)*100]
    v=np.linalg.solve(Q+lambda1*Omega, R)

    return v, Q

############## ---------------------------------------------------- ##############

def solve_g_kernel(data, f_list, v, t, K):

    '''
    Solve for the vector g in the kernel regression (this is equivalent to the factor loadings).

    Parameters:
    - data (list of matrices): List of matrices corresponding to characteristics for each stock.
    - f_list (list of arrays): List of factor values for each time period.
    - v (array): Vector v.
    - t (int): Time period.
    - K (matrix): Kernel matrix.

    Returns:
    - array: Vector g.
    '''

    T=len(data)

    g = np.sum([(K[t*100:(t+1)*100,s*100:(s+1)*100]@v[s*100:(s+1)*100]).reshape(-1,1)@f_list[s].reshape(1,-1) for s in range(0,T)], axis=0)
    
    return g

############## ---------------------------------------------------- ##############

def Gram_matrix(data, v, f_list, t, K):

    '''
    Compute the Gram matrix.

    Parameters:
    - data (list of matrices): List of matrices corresponding to characteristics for each stock.
    - v (array): Vector v.
    - f_list (list of arrays): List of factor values for each time period.
    - t (int): Time period.
    - K (matrix): Kernel matrix.

    Returns:
    - matrix: Gram matrix.
    '''

    g=solve_g_kernel(data, f_list, v, t, K)

    return g@g.T

############## ---------------------------------------------------- ##############

def solve_f(ret, v, lambda2, data, f_list, Omega, K):

    '''
    Solve for the list of the factors in the kernel regression.

    Parameters:
    - ret (list of arrays): List of returns for each time period.
    - v (array): Vector v.
    - lambda2 (float): Regularization parameter.
    - data (list of matrices): List of matrices corresponding to characteristics for each stock.
    - f_list (list of arrays): List of factor values for each time period at the previous iteration.
    - Omega (matrix): Inverse of the weight matrix.
    - K (matrix): Kernel matrix.

    Returns:
    - list of arrays: List of factor values for each time period at the current iteration.
    - array: Vector g.
    - matrix: Gram matrix.
    '''

    T=len(data)
    f_list_new=[]

    for t in range(0,T):
        G = Gram_matrix(data, v, f_list, t, K)
        c = np.linalg.solve(G+lambda2*Omega, ret[t])
        g = solve_g_kernel(data, f_list, v, t, K)
        f_list_new.append(g.T@c)
    f_list=f_list_new.copy()

    return f_list, g, G

############## ---------------------------------------------------- ##############

def kernel_regression(data, ret, f_list, lambda1, lambda2, Omega1, Omega2, max_iter, N, K):

    '''
    Compute the kernel regression through the alternating minimization algorithm.

    Parameters:
    - ret (list of arrays): List of returns for each time period.
    - data (list of matrices): List of matrices corresponding to characteristics for each stock.
    - f_list (list of arrays): List of factor values for each time period at the previous iteration.
    - lambda1 (float): Regularization parameter.
    - lambda2 (float): Regularization parameter.
    - Omega1 (matrix): Inverse of the weight matrix.
    - Omega2 (matrix): Inverse of the weight matrix.
    - max_iter (int): Maximum number of iterations.
    - N (int): Number of stocks.
    - K (matrix): Kernel matrix.

    Returns:
    - list of arrays: List of factor values for each time period at the current iteration.
    - array: Vector v.
    - matrix: Matrix Q.
    - array: Vector g.
    - matrix: Gram matrix.
    '''
    
    for i in range(max_iter):

        v, Q = solve_v_kernel(ret, lambda1, data, f_list, N, Omega1, K)
        f_list, g, G = solve_f(ret, v, lambda2, data, f_list, Omega2, K)

    return f_list, v, Q, g, G

############## ---------------------------------------------------- ##############

def pivoted_chol(K, m_hat):

    '''
    Compute the Low Rank Approximation of the kernel matrix.

    Parameters:
    - K (matrix): Kernel matrix.
    - m_hat (int): Number of iterations (final number of rows of the matrix L).

    Returns:
    - matrix: Matrix L.
    - matrix: Matrix B.
    '''

    d = np.diag(K).reshape(-1,1)
    p_max = np.argmax(d)
    d_max = d[p_max]

    e = np.zeros(len(K)).reshape(-1,1)
    e[p_max] = 1

    L = np.sqrt(1/d_max) * K@e
    B = np.sqrt(1/d_max) * np.eye(len(K))@e

    d = d-L*L

    for m in range(1,m_hat):

        p_max = np.argmax(d)
        d_max = d[p_max]

        e = np.zeros(len(K)).reshape(-1,1)
        e[p_max] = 1

        l = np.sqrt(1/d_max) * (K-L@L.T)@e
        b = np.sqrt(1/d_max) * (np.eye(len(K))-B@L.T)@e

        L = np.concatenate((L, l), axis = 1)
        B = np.concatenate((B, b), axis = 1)

        d = d-l*l

    return L,B

############## ---------------------------------------------------- ##############

def solve_v_LR(data, B, ret, f_list, K, lambda1, Omega, m_hat):

    '''
    Solve for the vector v in the kernel regression using Low Rank approximation.

    Parameters:
    - data (list of matrices): List of matrices corresponding to characteristics for each stock.
    - B (matrix): Matrix used in Low Rank decomposition.
    - ret (list of arrays): List of returns for each time period.
    - f_list (list of arrays): List of factor values for each time period.
    - K (matrix): Kernel matrix.
    - lambda1 (float): Regularization parameter.
    - Omega (matrix): Inverse of the weight matrix.
    - m_hat (int): Number of iterations (final number of rows of the matrix L).

    Returns:
    - array: Vector v.
    '''

    rhs = np.sum([np.kron(B.T@K[i*100:(i+1)*100,:].T@Omega@(B.T@K[i*100:(i+1)*100,:].T).T, 
                        f_list[i].reshape(-1,1)@f_list[i].reshape(1,-1)) for i in range(len(data)-1)], 
                        axis=0) + lambda1*np.eye(m_hat*len(f_list[0]))

    lhs = np.sum([np.kron(B.T@K[i*100:(i+1)*100,:].T@Omega,f_list[i].reshape((-1,1)))@ret[i+1] 
                for i in range(len(data)-1)], axis=0)

    v = np.linalg.solve(rhs, lhs)

    return v

############## ---------------------------------------------------- ##############

def solve_g_kernel_LR(B, v, t, K, N):

    '''
    Solve for the vector g in the kernel regression (this is equivalent to the factor loadings) using Low Rank approximation.

    Parameters:
    - B (matrix): Matrix used in Low Rank decomposition.
    - v (array): Vector v.
    - t (int): Time period.
    - K (matrix): Kernel matrix.
    - N (int): Number of stocks.

    Returns:
    - array: Vector g.
    '''

    v_mat = v.reshape(-1,5)

    g = np.sum([(B.T@K[t*100:(t+1)*100,:].T)[i,:].reshape(-1,1)@v_mat[i,:].reshape(1,-1) for i in range(B.shape[1])], axis=0)

    return g

############## ---------------------------------------------------- ##############

def gram_matrix_LR(B, v, t, K, N):

    '''
    Compute the Gram matrix using Low Rank approximation.

    Parameters:
    - B (matrix): Matrix used in Low Rank decomposition.
    - v (array): Vector v.
    - t (int): Time period.
    - K (matrix): Kernel matrix.
    - N (int): Number of stocks.

    Returns:
    - matrix: Gram matrix.
    '''

    g = solve_g_kernel_LR(B, v, t, K, N)

    return g@g.T

############## ---------------------------------------------------- ##############

def solve_f_LR(ret, v, B, lambda2, data, Omega, K, N):

    '''
    Solve for the list of the factors in the kernel regression using Low Rank approximation.

    Parameters:
    - ret (list of arrays): List of returns for each time period.
    - v (array): Vector v.
    - B (matrix): Matrix used in Low Rank decomposition.
    - lambda2 (float): Regularization parameter.
    - data (list of matrices): List of matrices corresponding to characteristics for each stock.
    - Omega (matrix): Inverse of the weight matrix.
    - K (matrix): Kernel matrix.
    - N (int): Number of stocks.

    Returns:
    - list of arrays: List of factor values for each time period at the current iteration.
    - matrix: Gram matrix.
    - array: Vector g.
    '''
    
    T = len(data)
    f_list_new = []

    for t in range(T-1):
        G = gram_matrix_LR(B, v, t, K, N)
        g = solve_g_kernel_LR(B, v, t, K, N)
        c = np.linalg.solve(G+lambda2*Omega, ret[t+1])
        f_list_new.append(g.T@c)

    return f_list_new, G, g

############## ---------------------------------------------------- ##############

def kernel_regression_LR(data, K, B, ret, f_list, lambda1, lambda2, Omega, max_iter, m_hat, N):

    '''
    Compute the kernel regression through the alternating minimization algorithm using Low Rank approximation.

    Parameters:
    - data (list of matrices): List of matrices corresponding to characteristics for each stock.
    - K (matrix): Kernel matrix.
    - B (matrix): Matrix used in Low Rank decomposition.
    - ret (list of arrays): List of returns for each time period.
    - f_list (list of arrays): List of factor values for each time period at the previous iteration.
    - lambda1 (float): Regularization parameter.
    - lambda2 (float): Regularization parameter.
    - Omega (matrix): Inverse of the weight matrix.
    - max_iter (int): Maximum number of iterations.
    - m_hat (int): Number of iterations for pivoted Cholesky algorithm (final number of rows of the matrix L).
    - N (int): Number of stocks.

    Returns:
    - list of arrays: List of factor values for each time period at the current iteration.
    - array: Vector v.
    - matrix: Gram matrix.
    - array: Vector g.
    '''

    for i in range(max_iter):

        v = solve_v_LR(data, B, ret, f_list, K, lambda1, Omega, m_hat)
        f_list, G, g = solve_f_LR(ret, v, B, lambda2, data, Omega, K, N)

    return f_list, v, G, g

    