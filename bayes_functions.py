import numpy as np
from scipy.stats import norm, multivariate_normal, bernoulli, multivariate_t, uniform
from scipy.optimize import minimize

###### BAYES FUNCTIONS ###########################
### assuming MLE

def prior_Standard_Normal(beta, stretch = 10):
    '''
    Prior is standard normal multivariate distribution. The variable stretch allows
    to control the variance of the distribution.
    '''
    p = beta.shape[0]
    return multivariate_normal(mean = beta_0, cov= stretch * np.eye(p)).pdf(beta.flatten())

def prior_MLE(beta, stretch = 10):
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

def prior_t_student(beta, stretch = 10):
    '''
    Prior is a t-student multivariate distribution 
    '''
    p = beta.shape[0]
    mean_t_student = np.zeros(p)
    var_t_student = np.eye(p) #actually it isn't the var-cov matrix, which is df*var_t
    return multivariate_t(mean_t_student,stretch* var_t_student, df = 3.0).pdf(beta.flatten())
  
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
        l = likelihood(Y, X, x[i,:], take_log)
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