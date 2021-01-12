"""Class for Bernoulli mixture model"""

# Author: Shuo Wang <shuow@princeton.edu>
# License: BSD 2 clause

import warnings

import numpy as np
from scipy.special import logsumexp


class BernoulliMixture:
    """Bernoulli Mixture Model
    Maximum likelihood estimates calculated with
    expectation maximization.
    Parameters
    ----------
    n_components : int
        The number of mixture components.
    tol : float, defaults to 1e-3
        The convergence threshold for log likelihood in EM.
    max_iter : int, defaults to 100
        Maximum number of iterations in EM.
    smoothing : array_like, defaults to (0.1, 0.1)
        Add psuedo-counts in estimation of Bernoulli parameters p and 
        mixing coefficients pi. This is equivalent to apply Beta prior 
        over p and symmetric Dirichlet prior over pi and solve for
        maximum a posteriori estimates. No smoothing if
        set to (0, 0).
    init_params : {"random", "kmeans"}, default to "random"
        The method used to initialize mixing coefficients and
        Bernoulli parameters.
    n_init : int, defaults to 1
        Number of initializations for Bernoulli mixture
        model parameters. Only best model is kept.
    random_state : int, defaults to None
        Set seed if specified.

    Attributes
    ----------
    pi_ : numpy array, shape (n_components,)
        Mixing coefficients of Bernoulli distributions.
    p_ : numpy array, shape (n_components, n_features)
        Probability of 1 for each variable for each Bernoulli component.
    converged_ : bool
        Whether EM has converged within maximum iterations.
    n_iter_ : int
        Number of iterations to reach convergence.
    n_params_ : int
        Number of free parameters in the Bernoulli mixture
        distributions.
    aic_ : float
        Akaike information criterion for fitted Bernoulli mixture model.
    bic_ : float
        Bayesian information criterion for fitted Bernoulli mixture
        model.
    X : numpy array, shape (n_samples, n_features)
        Data used to fit Bernoulli mixture model.
    """
    def __init__(self, n_components, tol=1e-3, max_iter=100,
                 smoothing=(0.1, 0.1), init_params="random",
                 n_init=1, random_state=None):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        assert np.all(np.array(smoothing) >= 0), \
            "Smoothing must be non-negative."
        self.smoothing = smoothing
        self.init_params = init_params
        self.n_init = n_init
        self.random_state = random_state

        self.pi_ = None
        self.p_ = None
        self.converged_ = False
        self.n_iter_ = None
        self.n_params_ = None
        self.aic_ = None
        self.bic_ = None
        self.X = None

        self._log_X_cond_Z = None
        self._resp = None
        self._log_l_per_sample = None
        self._log_l = -np.Inf

    def _initialize(self):
        """Initialize parameters with random initialization"""
        # log P(x|z) per example
        self._log_X_cond_Z = np.empty((self.X.shape[0], self.n_components))

        self.pi_ = np.ones(self.n_components) / self.n_components
        self.p_ = np.random.rand(self.n_components, self.X.shape[1])
        if self.init_params == "kmeans":
            kmeans = KMeans(n_clusters=self.n_components)
            kmeans.fit(self.X)
            self._resp = np.zeros((self.X.shape[0], self.n_components))
            self._resp[np.arange(self.X.shape[0]), kmeans.labels_] = 1
            self._m_step()

    def _e_step(self):
        """E step"""
        log_l = self._log_likelihood()

        log_resp = self._log_X_cond_Z + np.log(self.pi_) - \
            self._log_l_per_sample.reshape(-1, 1)
        self._resp = np.exp(log_resp)

        return log_l

    def _m_step(self):
        """M step"""
        eff_k = self._resp.sum(axis=0)

        # for each component
        for j in range(self.n_components):
            self.p_[j, :] = np.sum(self.X * self._resp[:, j]\
                .reshape(-1, 1), axis=0)

        self.p_ = (self.p_ + self.smoothing[0]) / \
                  (eff_k.reshape(-1, 1) + 2 * self.smoothing[0])

        self.pi_ = (eff_k + self.smoothing[1]) / \
                   (self.X.shape[0] + self.smoothing[1] * \
                    self.n_components)

    def _log_likelihood(self):
        """Calculate observed data log likelihood given parameters"""
        # for each example
        for i in range(self.X.shape[0]):
            self._log_X_cond_Z[i, :] = np.sum(self.X[i, :] * np.log(self.p_) \
                + (1 - self.X[i, :]) * np.log(1 - self.p_), axis=1)

        # observed data log likelihood
        # each example
        self._log_l_per_sample = logsumexp(self._log_X_cond_Z + \
            np.log(self.pi_), axis=1)
        # all examples
        return np.sum(self._log_l_per_sample)

    def _n_parameters(self):
        """Calculate number of free parameters in the model"""
        # number of Bernoulli parameters
        n_p = self.p_.shape[0] * self.p_.shape[1]
        # number of free mixing coefficients
        n_pi = len(self.pi_) - 1
        self.n_params_ = n_p + n_pi

    def fit(self, X):
        """Fit a Bernoulli mixture with EM
        Parameters
        ----------
        X : numpy array, shape (n_samples, n_features)
            Data used fit Bernoulli mixture model. Each row
            represents an observation.

        Returns
        -------
        self
        """
        assert X.shape[0] > self.n_components, \
            "Number of components should be smaller than size of data."

        self.X = X

        if self.random_state:
            np.random.seed(self.random_state)

        # multiple initializations
        best_log_l = -np.Inf
        for _ in range(self.n_init):
            self._initialize()
            self._n_parameters()

            for n_iter in range(self.max_iter):
                log_l = self._e_step()

                if log_l - self._log_l > self.tol:
                    self._log_l = log_l
                    self._m_step()
                else:
                    self._log_l = log_l
                    self.converged_ = True
                    break
            else:
                self.converged_ = False
                warnings.warn("Solution not converged within tolerance. " \
                              "Try to use larger max_iter.")
            self.n_iter_ = n_iter + 1

            if self._log_l > best_log_l:
                best_log_l = self._log_l
                best_params = (self.pi_, self.p_)
                best_intermediate = (self._log_X_cond_Z, self._resp)
                best_state = (self.converged_, self.n_iter_)

        # keep the best model
        self._log_l = best_log_l
        self._log_l_per_sample = best_log_l / self.X.shape[0]
        self.pi_, self.p_ = best_params
        self._log_X_cond_Z, self._resp = best_intermediate
        self.converged_, self.n_iter_ = best_state

        # calculate AIC, BIC
        self.aic_ = 2 * self.n_params_ - 2 * self._log_l
        self.bic_ = np.log(self.X.shape[0]) * self.n_params_ - \
            2 * self._log_l

        return self

    def predict(self, X):
        """Predict latent class for each observation
        Parameters
        ----------
        X : numpy array, shape (n_samples, n_features)
            Data to predict latent class labels. Each row
            represents an observation.

        Returns
        -------
        y : numpy array, shape (n_samples,)
            Predicted latent class labels.
        """
        _, resp = self.predict_proba(X)

        return resp.argmax(axis=1)

    def predict_proba(self, X):
        """Calculate log likelihood and responsibilities
        Parameters
        ----------
        X : numpy array, shape (n_samples, n_features)
            Data to predict latent class labels. Each row
            represents an observation.

        Returns
        -------
        log_l : numpy array, shape (n_samples,)
            Log likelihood of observing X given fitted parameters.
        resp : numpy array, shape (n_samples, n_components)
            Responsibilities of each component of Bernoulli
            distribution.
        """
        log_X_cond_Z = np.empty((X.shape[0], self.n_components))
        # for each example
        for i in range(X.shape[0]):
            log_X_cond_Z[i, :] = np.sum(X[i, :] * np.log(self.p_) + \
                (1 - X[i, :]) * np.log(1 - self.p_), axis=1)

        # per sample observed data log likelihood given fitted model
        log_l = logsumexp(log_X_cond_Z + np.log(self.pi_), axis=1)
        log_resp = log_X_cond_Z + np.log(self.pi_) - \
            log_l.reshape(-1, 1)
        resp = np.exp(log_resp)

        return log_l, resp

    def generate_sample(self, n=1, random_state=None, component=None):
        """Generate random samples from fitted Bernoulli
           mixture distribution
        Parameters
        ----------
        n : int, defaults to 1
            Number of samples to generate.
        random_state : int, defaults to None
            Set seed when generating samples.
        component : int, defaults to None
            Component used to generate samples.

        Returns
        -------
        X : numpy array, shape (n, n_features)
            Generated samples.
        y : numpy array, shape (n,)
            Components selected.
        """
        X = np.empty((n, self.X.shape[1]))
        y = np.empty(n, dtype = int)

        if random_state:
            np.random.seed(random_state)

        for j in range(n):
            if component:
                y[j] = component
            else:
                # select component from multinoulli distribution
                y[j] = np.random.choice(range(self.n_components),
                                        p=self.pi_)

            # generate random sample from Bernoulli for that component
            X[j, :] = np.random.binomial(np.ones(self.X.shape[1],
                                         dtype=int),
                                         self.p_[y[j], :])

        return X, y