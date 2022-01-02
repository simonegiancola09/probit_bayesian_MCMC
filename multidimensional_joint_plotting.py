## MULTIDIMENSIONAL JOINT PLOTTING #################################################


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from scipy.stats import norm, multivariate_normal, bernoulli, multivariate_t, uniform
from scipy.optimize import minimize
#####################################################

###### MATPLOTLIB SETTINGS #########################
# we should avoid this for later purposes %matplotlib inline
plt.rc('legend', fontsize=11)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
colors = ['tab:cyan', 'tab:purple', 'tab:olive', 'tab:red', 'tab:orange',
                  'tab:brown', 'tab:pink', 'tab:gray', 'tab:green', 'tab:blue']




def plot_double_dim_accepted_rejected(thetas, dimensions = (0,1), plot_theta_true = True,
                                      plot_rejected = True, rejected = None,
                                      chain_rejected = None):
  '''
  Given two dimensions and some options plot the chain in an R^2 plane
  '''
  fig = plt.figure(figsize=(10,10))
  dim1, dim2 = dimensions
  if plot_theta_true:
    plt.plot(beta_true[dim1], beta_true[dim2], '.', alpha=1, markersize=15, label='Beta True', color='darkorange')
  theta_1 = thetas[:1000, dim1]
  theta_2 = thetas[:1000, dim2]
  plt.plot(theta_1, theta_2, label='Path', alpha=.8)
  plt.plot(thetas[np.where(chain_rejected==False)][:1000,dim1], thetas[np.where(chain_rejected==False)][:1000,dim2], 'b.', label='Accepted', alpha=.5)
  if plot_rejected:
    plt.plot(rejected[np.where(chain_rejected==True)][:2000,dim1], rejected[np.where(chain_rejected==True)][:2000,dim2], 'rx', label='Rejected', alpha=.4)
  plt.legend()
  plt.title('Double Dimensional Chain convergence for {}, {}'.format(dim1, dim2))
  plt.xlabel('Beta{}'.format(dim1))
  plt.ylabel('Beta{}'.format(dim2))
  plt.close(fig)
  return fig

def plot_triple_dim_accepted_rejected(thetas, dimensions = (0,1,2), plot_theta_true = True,
                                      plot_rejected = True, rejected = None,
                                      chain_rejected = None):
  '''
  Given three dimensions and some options plot the chain in an R^3 plane
  '''
  fig = plt.figure(figsize=(22,14))
  ax1 = fig.add_subplot(121, projection = '3d')
  ax1.view_init(20, 130)
  dim1, dim2, dim3 = dimensions
  ax1.plot(thetas[:1000,dim1], thetas[:1000,dim2], thetas[:1000,dim3], alpha=.5)
  ax1.plot(thetas[np.where(chain_rejected==False)][:300,dim1],
           thetas[np.where(chain_rejected==False)][:300,dim2],
           thetas[np.where(chain_rejected==False)][:300,dim3],
           'b.', label='Accepted', alpha=.5)
  if plot_rejected:
    ax1.plot(rejected[np.where(chain_rejected==True)][:200,dim1],
            rejected[np.where(chain_rejected==True)][:200,dim2],
             rejected[np.where(acc_rej==True)][:200,dim3],
            'rx', label='Rejected', alpha=.5)
  if plot_theta_true:
    ax1.plot(beta_true[dim1], beta_true[dim2], beta_true[dim3], '.',
             alpha=1, markersize=15, label='Beta True', color='darkorange')
  ax1.set_xlabel('Beta{}'.format(dim1))
  ax1.set_ylabel('Beta{}'.format(dim2))
  ax1.set_zlabel('Beta{}'.format(dim3))
  plt.title('Triple Dimensional Chain Convergence for {}, {}, {}'.format(dim1, dim2, dim3))

  plt.legend()
  # plt.close(fig)
  return fig 