import numpy as np
from scipy.stats import norm, multivariate_normal, bernoulli, multivariate_t, uniform
from scipy.optimize import minimize


### GIBBS FUNCTIONS ###########################################################################################
def diffuse_full_conditional_beta(Z, X):
    '''
    full conditional of beta assuming an uniformative prior of beta
    '''
    p = X.shape[1]
    var_cov = np.linalg.inv(np.dot(X.T, X))
    #print(var_cov.shape)
    dot = np.dot(X.T, Z)
    normal_mean = np.dot(var_cov, dot).flatten()
    #print(normal_mean.shape)
    z = np.random.multivariate_normal(mean = normal_mean, cov = var_cov).reshape((p,1))
    return z

def standard_normal_full_conditional_beta(Z, X, beta_prior = np.zeros(p+1) , var_prior = np.eye(p+1), stretch = 10):
    '''
    full conditional of beta assuming a multivariate normal prior of beta with mean beta_prior
    and variance-covariance matrix var_prior
    '''
    var_prior = stretch*var_prior
    var_cov = np.linalg.inv(np.linalg.inv(var_prior) + np.dot(X.T, X))
    normal_mean = np.dot(var_cov, np.dot(np.linalg.inv(var_prior),beta_prior) + np.dot(X.T, Z).flatten())
    #print(normal_mean.shape, '/n', normal_mean, var_cov.shape, '/n', var_cov)
    return np.random.multivariate_normal(mean = normal_mean, cov = var_cov)

def normal_MLE_full_conditional_beta(Z, X, beta_prior = beta_MLE , var_prior = Var_MLE):
    '''
    full conditional of beta assuming a multivariate normal prior of beta with mean beta_prior
    and variance-covariance matrix var_prior
    '''
    var_cov = np.linalg.inv(np.linalg.inv(var_prior) + np.dot(X.T, X))
    normal_mean = np.dot(var_cov, np.dot(np.linalg.inv(var_prior),beta_prior) + np.dot(X.T, Z).flatten())
    #print(normal_mean.shape, '/n', normal_mean, var_cov.shape, '/n', var_cov)
    return np.random.multivariate_normal(mean = normal_mean, cov = var_cov)

def draw_from_truncated(Y, X, theta):
    '''
    Draw from a truncated positive or negative normal distribution with the simplified formula
    mentioned in the notes. It is more concise but not generalized to any truncated normal as 
    it assumes that it is restricted to the positive / negative case with d = 1
    '''
    c = np.dot(X, theta).flatten()
    n = Y.shape[0]
    Z = c # c is common for all z
    U = np.random.uniform(size = n) # we sample n uniform variables
    Phi_of_minus_c = norm.cdf(-c) # this is the value of Phi(-Xbeta)
    # we then index depending on
    Z[(Y == 0).flatten()] += norm.ppf(U[(Y == 0).flatten()] * Phi_of_minus_c[(Y == 0).flatten()]) 
    # when Y is zero, the formula simplifies to this
    Z[(Y == 1).flatten()] += norm.ppf(Phi_of_minus_c[(Y == 1).flatten()] + U[(Y == 1).flatten()] * (1 - Phi_of_minus_c[(Y == 1).flatten()]))
    # when Y is 1, the formula simplifies to this
    #Z = np.minimum(Z, 10e3)
    #Z = np.maximum(Z, -10e10)
    return Z.reshape((n, 1))
  
####################################################################################################


#### GIBBS ALGORITHM ####################################################################################################
def Auxiliary_Gibbs_Sampling(beta_0, Y, X, burnin, iterations_after_burnin,
                             print_warning_every = 500, prior = diffuse_full_conditional_beta,
                             verbose = True):
    '''
    Auxiliary gibbs sampling. Here the proposal is always accepted and in order to sample from full conditional
    an auxiliary variable Z is introduced. Full conditionals are worked out in the report.
    '''
    print('Computation starts')
    theta_t = beta_0
    p = beta_0.shape[0]
    n = X.shape[0]
    #Which prior to use?
    #if prior == "diffuse":
        #proposal = lambda z : diffuse_full_conditional_beta(z, X = X)
    #else:
        #proposal = lambda z : normal_full_conditional_beta(z, X = X)
    proposal = lambda z : prior(z, X = X)
    # starting burnin phase
    for i in range(1, burnin + 1):
        if i % print_warning_every == 0 and verbose:
            print('Reached {} iterations of burn-in'.format(i))
        Z = draw_from_truncated(Y, X, theta_t)
        theta_t = proposal(Z)
    
    #starting with ergodic chain
    thetas = np.zeros((iterations_after_burnin, p))
    if verbose:
        print('Starting to run the ergodic chain...')
    for i in range(iterations_after_burnin): # to print warning for iterations reached
        j = i + 1
        if j % print_warning_every == 0 and verbose:
            print('Reached {} iterations of the ergodic chain'.format(j))
        Z = draw_from_truncated(Y, X, theta_t)
        theta_t = proposal(Z)
        thetas[i, :] = theta_t.flatten()
    mean_thetas = np.mean(thetas, axis = 0)
    if verbose:
        print('Iterations terminated')
    print('Mean at termination:', mean_thetas.flatten())
    #print('Mean Square Error:', np.sum((mean_thetas.flatten() - beta_true.flatten()) ** 2) / beta_true.shape[0])
    return thetas, mean_thetas
####################################################################################################      