import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import kde



####### CHECKING STATIONARITY AND ERGODICITY #####################

def multi_plot_chain(thetas, function, ax = None, title = '', plot_theta_true = True, fig_size=(12,14)):
    '''
    Plot for all dimensions of the parameter for some function.
    '''
    if ax == None:
        plt.gca()
    # retrieving dimension count and finding an efficient double column representation
    n = thetas.shape[1]
      # we can create a multiple subplot with two columns
    fig, ax = plt.subplots(n, 1, figsize = fig_size)
    # filling the first column
    for i in range(n):
        ax[i] = function(thetas, dimension = i, ax = ax[i], plot_theta_true = plot_theta_true )
    fig.tight_layout(pad = 5.0)
    fig.suptitle('Multi_{}_plot'.format(title), fontsize = 20)
    plt.close(fig)
    return fig


def plot_trace_dimension(thetas, dimension = 0, ax = None, plot_theta_true = True, title=True):
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
        plt.xlabel('Iterations')
        plt.ylabel('Values')
        if title:
            plt.title('Chain behavior for the parameter at dimension {}'.format(dimension))
    elif ax is not None and title:
        ax.set_title('Dimension {}'.format(dimension), fontsize = 10)
    ax.plot(theta_i, label = 'beta {} chain'.format(dimension), c = colors[0], alpha=.7)
    ax.plot(np.ones(len(theta_i)) * theta_i.mean(), label = 'beta {} mean'.format(dimension), c = colors[3], ls='--')
    if plot_theta_true:
        ax.plot(np.ones(len(theta_i)) * theta_true, label = 'beta {} True'.format(dimension), c = 'black', ls='--')
    ax.legend()
    
    plt.close(fig) # to avoid showing it
    return fig


# compatible with multi_plot
def plot_autocorrelation(betas, dimension = 0, ax = None, plot_theta_true=None, title=True):
    '''
    Plot autocorrelation for a given dimension and number of iterations
    '''
    iterations = betas[:,dimension].size
    n_dim = betas.shape[1]
    figure = plt.figure()
    if ax == None:
        #ax = plt.gca
        # plt.legend()
        plt.xlabel('Lags')
        plt.ylabel('Rho')
        plt.title('Autocorrelation beta{}'.format(dimension))
    betas_i = betas[:, dimension]
    if title:
        t = 'Autocorrelation beta{}'.format(dimension)
    else:
        t = None
    plot_acf(betas_i, ax = ax, lags = iterations - 1, title = t)
    
    plt.close(figure)
    # return figure

def plot_cumulative_mean(thetas, dimension = 0, ax = None, plot_theta_true = True, title=True):
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
        if title:
            plt.title('Cumulative Average of parameter at dimension {}'.format(dimension))
    elif ax is not None and title:
        ax.set_title('Dimension {}'.format(dimension), fontsize = 10)
    ax.plot(cum_avg, label = 'beta {} chain'.format(dimension), c = colors[0])
    if plot_theta_true:
        ax.plot(np.ones(len(theta_i)) * theta_true, label = 'beta {} True'.format(dimension), c = 'black', ls='--')
    ax.legend()
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
        ax.hlines(beta_true[dimension], 0, l, label='Beta true', alpha=1, linewidth=3, color='black', ls='--')
    ax.hlines(theta_i.mean(), 0, l, label='Estimated Beta', alpha=1, color='g', ls='--', linewidth=3)
    ax.legend()
    plt.close(fig)
    return fig

def plot_density_dimension(thetas, dimension = 0, ax = None, plot_theta_true = True, title = None):
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
    ax.axvline(theta_i.mean(), label = 'beta {} mean'.format(dimension), c = colors[3], ls='--')
    if plot_theta_true:
        ax.axvline(theta_true, label = 'beta {} True'.format(dimension), c = 'black', ls='--')
    ax.legend()
    plt.close(fig) # to avoid showing it
    return fig
####################################################################################################

def plot_alpha(rejected, iterations):
    fig = plt.figure(figsize=(10,5))
    alpha =  np.cumsum(1-rejected) / np.arange(1,iterations+1)
    plt.plot(alpha)
    plt.title(f'Acceptance Ratio equal to {np.round(alpha[-1],3)}', size=16)
    plt.xlabel("Iterations")
    plt.ylabel("Acceptance ratio")
    plt.close(fig)
    return fig