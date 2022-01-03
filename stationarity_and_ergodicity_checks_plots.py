# CHECKING STATIONARITY AND ERGODICITY


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from scipy.stats import norm, multivariate_normal, bernoulli, multivariate_t, uniform
from scipy.optimize import minimize
from scipy.stats import kde

#####################################################

###### MATPLOTLIB SETTINGS #########################
# we should avoid this for later purposes %matplotlib inline
plt.rc('legend', fontsize=11)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
colors = ['tab:cyan', 'tab:purple', 'tab:olive', 'tab:red', 'tab:orange',
                  'tab:brown', 'tab:pink', 'tab:gray', 'tab:green', 'tab:blue']




def plot_trace_dimension(thetas, dimension = 0, ax = None, plot_theta_true = True):
  '''
  Plot for a given dimension of the parameter the distribution of the values stored.
  Should present a somewhat stationary behavior. 
  Ideally, call function.show() to plot it
  '''
  if plot_theta_true:
    theta_true = beta_true.flatten()[dimension]
  theta_i = thetas[:, dimension]
  fig = plt.figure()
  if ax is None:
    ax = plt.gca()
    # single plot, set some decorator values
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Values')
    plt.title('Chain behavior for the parameter at dimension {}'.format(dimension))
  else:
    ax.set_title('Dimension {}'.format(dimension), fontsize = 10)
  ax.plot(theta_i, label = 'beta {} chain'.format(dimension), c = colors[0])
  ax.plot(np.ones(len(theta_i)) * theta_i.mean(), label = 'beta {} mean'.format(dimension), c = colors[1])
  if plot_theta_true:
    ax.plot(np.ones(len(theta_i)) * theta_true, label = 'beta {} True'.format(dimension), c = colors[2])
  ax.legend()
  plt.close(fig) # to avoid showing it
  return fig

def plot_density_dimension(thetas, dimension = 0, ax = None, plot_theta_true = True):
  '''
  Plot for a given dimension of the parameter the density of the values stored.
  Should present a somewhat normal behavior hopefully centered around beta true. 
  Ideally, call function.show() to plot it. 
  We will sample 1000 points
  '''
  if plot_theta_true:
    theta_true = beta_true.flatten()[dimension]
  theta_i = thetas[:, dimension]
  fig = plt.figure()
  density = kde.gaussian_kde(theta_i)
  x_points = np.linspace(np.min(theta_i), np.max(theta_i), 1000)
  if ax is None:
    ax = plt.gca()
    # single plot, set some decorator values
    # ax.legend()
    plt.xlabel('Space')
    plt.ylabel('Density')
    plt.title('density for the parameter at dimension {}'.format(dimension))
  else:
    ax.set_title('Dimension {}'.format(dimension), fontsize = 10)
  ax.plot(x_points, density(x_points), label = 'beta {} density'.format(dimension), c = colors[0])
  ax.axvline(theta_i.mean(), label = 'beta {} mean'.format(dimension), c = colors[1])
  if plot_theta_true:
    ax.axvline(theta_true, label = 'beta {} True'.format(dimension), c = colors[2])
  ax.legend()
  plt.close(fig) # to avoid showing it
  return fig

def multi_plot_chain(thetas, function, ax = None, title = ''):
  '''
  Plot for all dimensions of the parameter some function.
  '''
  if ax == None:
    plt.gca()
  # retrieving dimension count and finding an efficient double column representation
  n = thetas.shape[1]
  odd = (n % 2 == 1)
  if not odd:
    # we can create a multiple subplot with two columns
    fig, ax = plt.subplots(n // 2, 2, figsize = (15, 15))
  else:
    odd = True
    # we are dealing with an odd index so we will atttach it to the last row
    fig, ax = plt.subplots(n // 2 + 1, 2, figsize = (15, 15))
  # filling the first column
  for i in range(n // 2):
    ax[i,0] = function(thetas, dimension = i, ax = ax[i, 0])
  # filling the second column
  for i in range(n // 2):
    function(thetas, dimension = n // 2 + i, ax = ax[i, 1])
  if odd:
    function(thetas, dimension = n - 1, ax = ax[n // 2, 0])
  fig.tight_layout(pad = 4.0)
  fig.suptitle('Multi_{}_plot'.format(title), fontsize = 20)
  plt.close(fig)
  return fig

from statsmodels.graphics.tsaplots import plot_acf


def plot_autocorrelation(betas, dimension = 0, iterations = 1000, ax = None):
    '''
    Plot autocorrelation for a given dimension and number of iterations
    '''
    n_dim = betas.shape[1]
    figure = plt.figure()
    if ax == None:
      #ax = plt.gca
      # plt.legend()
      plt.xlabel('Lags')
      plt.ylabel('Rho')
      plt.title('Autocorrelation beta{}'.format(dimension))
    # else:
      #ax.set_title('Autocorrelation beta{}'.format(dimension))
    betas_i = betas[:, dimension]
    plot_acf(betas_i, ax = ax, lags = iterations - 1, title = 'Autocorrelation beta{}'.format(dimension))
    
    plt.close(figure)
    # return figure

    
def plot_cumulative_mean(thetas, dimension = 0, ax = None, plot_theta_true = True):
  if plot_theta_true:
    theta_true = beta_true.flatten()[dimension]
  theta_i = thetas[:, dimension]
  n = theta_i.shape[0]
  fig = plt.figure()
  cum_avg = theta_i.cumsum() / np.arange(1, n + 1)
  if ax is None:
    ax = plt.gca()
    # single plot, set some decorator values
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Cum Avg')
    plt.title('Cumulative Average of parameter at dimension {}'.format(dimension))
  else:
    ax.set_title('Dimension {}'.format(dimension), fontsize = 10)
  ax.plot(cum_avg, label = 'beta {} chain'.format(dimension), c = colors[0])
  if plot_theta_true:
    ax.plot(np.ones(len(theta_i)) * theta_true, label = 'beta {} True'.format(dimension), c = colors[2])
  ax.legend()
  plt.close(fig)
  return fig

def plot_alpha(rejected, iterations):
  fig = plt.figure(figsize=(10,5))
  alpha =  np.cumsum(1-rejected) / np.arange(1,iterations+1)
  plt.plot(alpha)
  plt.title('Acceptance Ratio')
  plt.xlabel("Iterations")
  plt.ylabel("Acceptance ratio")
  plt.close(fig)
  return fig


def plot_accepted_rejected(thetas, dimension=0, ax = None, plot_theta_true=True, rejected = None, chain_rejected = None):
  '''
  Plot the accepted rejected beta for the selected dimension.
  '''
  if plot_theta_true:
    theta_true = beta_true.flatten()[dimension]
  theta_i = thetas[:, dimension]
  rejected_i = rejected[:,dimension]
  fig = plt.figure(figsize=(12,16))
  if ax is None:
    ax = plt.gca()
    # single plot, set some decorator values
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Values')
    plt.title('Accepted vs Rejected Betas for parameter at dimension {}'.format(dimension))
  else:
    ax.set_title('Dimension {}'.format(dimension), fontsize = 10)

  l = len(theta_i)
  final = np.zeros(l)
  final[np.where(chain_rejected==False)] = theta_i[np.where(chain_rejected==False)]
  final[np.where(chain_rejected==True)] = rejected_i[np.where(chain_rejected==True)]

  ax.scatter(np.where(chain_rejected==True), final[np.where(chain_rejected==True)], marker='x', c='r', label='Rejected',alpha=0.5)
  ax.scatter(np.where(chain_rejected==False), final[np.where(chain_rejected==False)], marker='.', c='b', label='Accepted',alpha=0.8)
  if plot_theta_true:
    ax.hlines(beta_true[dimension], 0, l, label='Beta true', alpha=1, linewidth=3)
  ax.hlines(thetas[:,dimension].mean(), 0, l, label='Estimated Beta', alpha=1, color='g', ls='solid', linewidth=3)
  ax.legend()
  plt.close(fig)
  return fig
