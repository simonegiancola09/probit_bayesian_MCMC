#### PLOT DIFFERENTS ##########################################################
# graphical exploration of different settings for the algorithm


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from scipy.stats import norm, multivariate_normal, bernoulli, multivariate_t, uniform
from scipy.optimize import minimize
from MH_functions import MetropolisAlgorithm
#####################################################

###### MATPLOTLIB SETTINGS #########################
# we should avoid this for later purposes %matplotlib inline
plt.rc('legend', fontsize=11)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
colors = ['tab:cyan', 'tab:purple', 'tab:olive', 'tab:red', 'tab:orange',
                  'tab:brown', 'tab:pink', 'tab:gray', 'tab:green', 'tab:blue']




def plot_different_priors(starting_point = np.zeros(p+1), tau = 3, iterations = 5000, priors = None):
    '''
    Plot the cumulative mean evolution of the result using different prior specifications in the algorithm.
    Can specify a priors dictionary with custom names and functions or use the default one. 
    Default is standard normal, uninformative and multivariate t student
    '''
    dim = starting_point.shape[0]
    fig, ax = plt.subplots(dim, figsize = (16,12))
    if priors is None:
      priors = {prior_MLE : "Standard Normal", prior_uninformative : "Uninformative", 
                prior_t_student : "Multivariate_t_student"}
    l = []
    for prior in priors.keys():
        print(f'Prior is: {priors[prior]}')
        betas, mean_betas, _, _ = MetropolisAlgorithm(Y, X, beta_0, tau = tau, burnin = 0, iterations_after_burnin = iterations, log=True, verbose=False)
        l.append(betas)
    keys = list(priors.keys())
    for dimension in range(dim):
        for i in range(len(l)):
            axis_ = ax[dimension]
            axis_.plot(l[i][:,dimension].cumsum()/np.arange(1,iterations+1), label = priors[keys[i]] )
        axis_.hlines(beta_true[dimension], 0, iterations, color = "red", label = f"beta_{dimension}_true")
        axis_.legend()
    plt.close(fig)
    return fig


def plot_normal_priors_different_stretch(starting_point = np.zeros(p+1), tau = 3, stretch = [1,5, 100], iterations = 5000):
    '''
    Plot the cumulative mean evolution of the result using different stretch for the normal prior with cov matrix stretch*I.
    Here the hypermarameter of the model such as tau and iterations are predefined, also the starting point beta_0. However
    it is possible to customize them during the call of the function.
    '''
    keys = []
    priors = dict()
    for i in range(len(stretch)):
        keys.append(lambda x: prior_MLE(x, stretch = stretch[i]))
        priors[keys[i]] = f"Prior with stretch {stretch[i]}"
    l = []
    dim = starting_point.shape[0]
    fig, ax = plt.subplots(dim, figsize = (15,15))
    for prior in priors.keys():
            print(f'{priors[prior]}')
            betas, mean_betas, _, _ = MetropolisAlgorithm(Y, X, beta_0, tau_try, 0, iterations, log=True, verbose=False)
            l.append(betas)
    keys_2 = list(priors.keys())
    for dimension in range(dim):
        for i in range(len(l)):
            axis_ = ax[dimension]
            axis_.plot(l[i][:,dimension].cumsum()/np.arange(1,iterations+1), label = priors[keys_2[i]] )
        axis_.hlines(beta_true[dimension], 0, iterations, color = "red", label = f"beta_{dimension}_true")
        axis_.legend()
        axis_.title.set_text(f'Comparison for beta_{dimension}')
    plt.close(fig)
    return fig

# TODO beta should take a list of starting beta_0s which are reasonable or to explore, not random values
# optimal would be a dictionary with names (MLE, OLS, all Zeros, all ones, random) from which it takes inspiration
# for the legend and shows where each goes precisely
  
def plot_different_starting_betas(dimension, starting_values, tau = 1, iterations=5000):
  '''
  Plot the cumulative mean evolution of the result using different starting points.
  Here the hypermarameter of the model such as tau and iterations are predefined. However
  it is possible to customize them during the call of the function.
  You have also to specify which dimension of the resulting beta you want to plot.
  '''
  fig = plt.figure(figsize=(14,10))
  for i in starting_values:
    beta_0 = starting_values[i]
    print(f'Starting point at iteration {i}: {beta_0.flatten()}')
    betas, mean_betas, _, _ = MetropolisAlgorithm(Y, X, beta_0, tau = tau,
                                                  burnin = 0, iterations_after_burnin = iterations,
                                                  log=True, verbose=False)
    plt.plot(betas[:,dimension].cumsum()/np.arange(1,iterations+1), label=i)
  plt.legend()
  plt.title('Different Beta_0 Plots')
  plt.ylabel('Cumulative Mean')
  plt.xlabel('Iterations')
  plt.hlines(beta_true[dimension], 0, iterations, label='True Beta')
  plt.close(fig)
  return fig


def plot_different_taus(plot_type, tau_low, tau_high, dimension, iterations=5000):
  '''
  Plot the cumulative mean (plot_type != Acceptance) or the Acceptance rate (plot_type = Acceptance) using 
  different values of taus. The function takes as parameter the lowest and highest value of tau to use.
  You have also to specify which dimension of the resulting beta you want to plot.
  '''
  tau = np.arange(tau_low, tau_high)
  beta_0 = np.zeros((p+1,1))
  fig = plt.figure(figsize=(12,10))
  for i in tau:
    betas, _, _, accepted = MetropolisAlgorithm(Y, X, beta_0, i, 0, iterations, log=True, verbose=False)
    if plot_type=='Acceptance':
      plt.plot(1-accepted.cumsum()/np.arange(1, iterations+1), label='tau = {}'.format(i))
      plt.ylabel('Acceptance Proportion')
    else:
      plt.plot(betas[:,dimension].cumsum()/np.arange(1,iterations+1), label='tau = {}'.format(i))
      plt.ylabel('Cumulative Mean')
  if plot_type != 'Acceptance':
    plt.hlines(beta_true[dimension], 0, iterations, label = 'beta {} true'.format(dimension))
    
  plt.title('Different Tau Plots')
  plt.xlabel('Iterations')
  plt.legend()
  plt.close(fig)
  return fig

