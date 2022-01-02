######### IMPORTS ############################

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from scipy.stats import norm, multivariate_normal, bernoulli, multivariate_t, uniform
from scipy.optimize import minimize
#####################################################

###### METROPOLIS HASTINGS FUNCTIONS ###############

def proposal_beta_new(beta_t, X, tau = 1):
  '''
  Calculate the proposed beta sampling from the current one
  '''
  return np.random.multivariate_normal(mean = beta_t, cov = tau * np.linalg.inv(fisher_info(beta_t, X)))


def acceptance(beta_t, beta_new, X, Y, take_log=False, prior = prior_MLE):
  '''
  Calculate acceptance rate of the proposed beta with the MH algorithm formula and an MLE prior
  '''
  p = beta_t.shape[0]
  # check if this works, assign specified prior to acceptance
  # defualt is MLE
  posterior = lambda beta, y, x, log: posterior_beta_Y(beta, y, x, log, prior = prior)
  if take_log: 
    a = posterior(beta_new.reshape((p, 1)), Y, X, True)
    b = posterior(beta_t.reshape((p,1)), Y, X, True)
    return a - b
  return np.min([1, posterior(beta_new.reshape((p,1)), Y, X, False) / posterior(beta_t.reshape((p,1)), Y, X, False)])

####### METROPOLIS ALGORITHM ##############

def MetropolisAlgorithm(Y, X, beta_0 = np.zeros((beta_true.shape[0],1)), tau = 1, burnin = 1000, iterations_after_burnin = 5000, print_warning_every = 500, prior = prior_MLE, log=False, verbose=True):
  '''
  Metropolis Algorithm implementation
  Given a starting value and a proposal instrumental density do the following:
  1. draw a candidate from the proposal
  2. draw u distributed as a uniform
  3. evaluate the acceptance rate
  4. set the new value if the acceptance rate is higher than u
  Iterating, we obtain a stationary chain. Burnin parameter is used to get to stationarity. 
  '''
  print('Computation starts')
  accepted = 0
  theta_t = beta_0
  p = beta_0.shape[0]
  # new version
  acceptance_rate = lambda theta_t, theta_new : acceptance(theta_t, theta_new, X = X, Y = Y, take_log=log, prior = prior)
  # old version
  # if method == 'MLE':
  #  acceptance_rate = lambda theta_t, theta_new : acceptance(theta_t, theta_new, X = X, Y = Y, log=log)
  #else:
  #  acceptance_rate = lambda theta_t, theta_new : acceptance_uninformative(theta_t, theta_new, X = X, Y = Y, take_log=log)
  
  proposal = lambda theta_t : proposal_beta_new(theta_t, X = X, tau = tau)
  for i in range(1, burnin + 1): # done to print average acceptance rate
    if i % print_warning_every == 0 and verbose:
      print('Reached {} iterations of burn-in, so far {} values were accepted. \nAcceptance proportion is {}\n'.format(i, accepted, round(accepted / i, 2)))
    theta_prop = proposal(theta_t.flatten())
    acc_rate = acceptance_rate(theta_t, theta_prop)
    u = np.random.uniform()
    if log:
      u = np.log(u)

    if u <= acc_rate:
      theta_t = theta_prop
      accepted += 1

  # after burnin iterations the chain should be stationary
  # we start storing values to check for stationarity
  rej = np.zeros(iterations_after_burnin, dtype=bool)
  rejected_betas = np.zeros((iterations_after_burnin, p))
  thetas = np.zeros((iterations_after_burnin, p))
  if verbose:
    print('Starting to run the ergodic chain...')
  for i in range(iterations_after_burnin): # to print warning for iterations reached
    j = i + 1
    if j % print_warning_every == 0 and verbose:
      print('Reached {} iterations of the ergodic chain'.format(j))
    theta_prop = proposal(theta_t.flatten())
    acc_rate = acceptance_rate(theta_t, theta_prop)
    u = np.random.uniform()
    if log:
      u = np.log(u)
    if u <= acc_rate:
      theta_t = theta_prop
    else:
      rejected_betas[i,:] = theta_prop
      rej[i] = True
    thetas[i, :] = theta_t.flatten()
  mean_thetas = np.mean(thetas, axis = 0)
  if verbose:
    print('Iterations terminated')
  print('Mean at termination:', mean_thetas.flatten())
  return thetas, mean_thetas, rejected_betas, rej
##################################################