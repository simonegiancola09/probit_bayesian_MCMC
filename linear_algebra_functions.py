######### LINEAR ALGEBRA FUNCTIONS ###############
import numpy as np
def eta(X, beta):
  '''
  calculate eta which is the dot product of observations and the vector
  '''
  if X.shape[0] == 1:
    return np.dot(X.T, beta)
  else:
    return np.dot(X, beta)

def phi_normal(eta):
  '''
  Calculate the normal cdf of a given value eta
  '''
  return norm.cdf(eta)

def fisher_info(beta_t, X):
  '''
  Calculate the fisher information matrix, method used is X^TWX that we saw in class
  '''
  n = X.shape[0]
  p = X.shape[1]
  eta_val = eta(X, beta_t)
  e = phi_normal(eta_val)
  # to avoid 'strange' behaviors of numbers with approximations
  # we limit the result
  e = np.minimum(e, .999999)
  e = np.maximum(e, .000001)
  # this ensures that the matrix is not singular
  num = norm.pdf(eta_val)
  #num = np.minimum(num, .999999)
  #num = np.maximum(num, .000001)
  v = num ** 2 / (e * (1 - e))
  W = np.diag(v.reshape((n,)))
  I = np.linalg.multi_dot([X.T, W, X])
  assert I.shape == (p, p)
  return I
##################################################