###### BAYES FUNCTIONS ###########################
### assuming MLE
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from scipy.stats import norm, multivariate_normal, bernoulli, multivariate_t, uniform
from scipy.optimize import minimize
def prior_MLE(beta, stretch = 100):
  '''
  Prior is normally distributed centered around the MLE for the parameter
  beta_MLE and Var_MLE are global variables
  '''
  n = beta.shape[0]
  return multivariate_normal(mean = beta_MLE, cov= stretch * Var_MLE).pdf(beta.flatten())

def prior_uninformative(beta):
  '''
  A not informative prior, where every beta is as likely. 
  '''
  # prior that does not add any information
  return 1

def prior_t_student(beta, stretch = 100):
  '''
  Prior is a t-student multivariate distribution 
  '''
  p = beta.shape[0]
  mean_t_student = np.zeros(p)
  var_t_student = np.eye(p) #actually it isn't the var-cov matrix, which is df*var_t
  return multivariate_t(mean_t_student,stretch* var_t_student, df = 3.0).pdf(beta.flatten())

def truncated_normal(c,d,a,b):
  '''
  Returns a number from a Normal(c,d**2) restricted to (a,b) 
  '''
  
  u = uniform(0,1).rvs()
  k1 = norm.cdf((a-c)/np.sqrt(d))
  k2 = norm.cdf((b-c)/np.sqrt(d))

  p = k1 + u * (k2 - k1)

  Y = c + np.sqrt(d) * norm.ppf(p)
  
  return Y
  
def likelihood(Y, X, beta, take_log = False):
  '''
  Calculate Likelihood of beta with the given observations, 
  if take_log is True we take the log likelihoods
  '''
  prob = norm.cdf(np.dot(X, beta))
  #p = np.minimum(p, .9999999)
  #p = np.maximum(p, .0000001)
  if take_log:
    return np.sum(np.log(prob ** Y * (1 - prob) ** (1 - Y)))
  return np.prod(prob ** Y * (1 - prob) ** (1 - Y))
    

def posterior_beta_Y(beta, Y, X, take_log=False, prior = prior_MLE):
  '''
  Given beta and Y evaluate the posterior distribution of beta 
  without the normalization constant. 
  The variable prior is used to specify a prior distribution. Likelihood is always
  the same as we assume the model is always binary probit
  '''
  if take_log:
    return np.log(prior(beta)) + likelihood(Y, X, beta, True)
  post = prior(beta) * likelihood(Y, X,  beta, False)
  return post

def plot_likelihood(likelihood, Y, X, beta_true, take_log, colors = colors):
  '''
  Function plots likelihood estimation for a set of toy values

  '''
  if take_log:
    x = np.linspace(beta_true - 3, beta_true + 3, 1000)
  else:
    x = np.linspace(beta_true - .5, beta_true + .6, 1000)
  y = []
  fig = plt.figure(figsize = (12, 8))
  for i in range(1000):
    if take_log:
      l = likelihood(Y, X, x[i,:], True)
      #l = posterior_beta_Y(x[i,:], Y, X, log=True)
    else:
      l = likelihood(Y, X, x[i,:], False)
      # l = posterior_beta_Y(x[i,:], Y, X, take_log=False)
    y.append(l)
  for j in range(beta_true.shape[0]):
    plt.plot(x[:,j], y, c=colors[j])
    plt.axvline(beta_true[j], c = colors[j], label = 'beta {}'.format(j), linestyle = 'dashed')
  # main options to set for a single plot
  plt.legend()
  plt.xlabel('Beta range')
  plt.ylabel('Likelihood')
  plt.title('Likelihood function for all dimensions of beta')
  plt.close(fig) # as to not show it when called, only if the variable is called
  return fig



##################################################