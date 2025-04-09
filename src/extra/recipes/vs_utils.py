from typing import Optional, Union, Tuple
import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin

import time
from functools import partial
import numpy as np
from scipy import linalg, sparse

from sklearn.decomposition import IncrementalPCA

import sklearn.decomposition
from sklearn.utils.extmath import randomized_svd
from numbers import Integral
from sklearn.utils import metadata_routing
from sklearn.base import _fit_context
from sklearn.utils import gen_batches
from sklearn.utils._param_validation import Interval
from sklearn.utils.extmath import _incremental_mean_and_var, svd_flip
from sklearn.utils.validation import validate_data

class TruncatedIncrementalPCA(IncrementalPCA):
    """Incremental principal components analysis (IPCA).

    Linear dimensionality reduction using Singular Value Decomposition of
    the data, keeping only the most significant singular vectors to
    project the data to a lower dimensional space. The input data is centered
    but not scaled for each feature before applying the SVD.

    Exactly the same as IncrementalPCA from sklearn, but with truncated SVD for speed.
    """

    def __init__(self, n_components=None, *, whiten=False, copy=True, batch_size=None, svd_solver='random'):
        super().__init__(n_components=n_components,whiten = False, copy=True, batch_size=None)
        self.svd_solver = svd_solver
        if n_components is not None:
            self.svd = self.create_svd(n_components)

    def create_svd(self,n_components):
        if self.svd_solver == 'blas':
            svd = partial(np.linalg.svd, full_matrices=False)
        elif self.svd_solver == 'random':
            svd = partial(randomized_svd,n_components=n_components)
        else:
            raise Exception(f'svd_solver has to be "blas" or "random" but is {self.solver}')
        return svd

    def partial_fit(self, X, y=None, check_input=True, X_var = None, mask=None):
        """Incremental fit with X. All of X is processed as a single batch.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        check_input : bool, default=True
            Run check_array on X.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        start = time.time()
        first_pass = not hasattr(self, "components_")
        
        if check_input:
            if sparse.issparse(X):
                raise TypeError(
                    "IncrementalPCA.partial_fit does not support "
                    "sparse input. Either convert data to dense "
                    "or use IncrementalPCA.fit to do so in batches."
                )
            X = validate_data(
                self,
                X,
                copy=self.copy,
                dtype=[np.float64, np.float32],
                force_writeable=True,
                reset=first_pass,
            )
        #print(f'check input = {time.time()-start}s')
        n_samples, n_features = X.shape
        if first_pass:
            self.components_ = None
            
        if self.n_components is None:
            if self.components_ is None:
                self.n_components_ = min(n_samples, n_features)
            else:
                self.n_components_ = self.components_.shape[0]
            self.svd = self.create_svd(self.n_components_)
        elif not self.n_components <= n_features:
            raise ValueError(
                "n_components=%r invalid for n_features=%d, need "
                "more rows than columns for IncrementalPCA "
                "processing" % (self.n_components, n_features)
            )
        elif self.n_components > n_samples and first_pass:
            raise ValueError(
                f"n_components={self.n_components} must be less or equal to "
                f"the batch number of samples {n_samples} for the first "
                "partial_fit call."
            )
        else:
            self.n_components_ = self.n_components
        
        if (self.components_ is not None) and (
            self.components_.shape[0] != self.n_components_
        ):
            raise ValueError(
                "Number of input features has changed from %i "
                "to %i between calls to partial_fit! Try "
                "setting n_components to a fixed value."
                % (self.components_.shape[0], self.n_components_)
            )
        #print(f'set svd and n_components = {time.time()-start}s')

        # This is the first partial_fit
        if not hasattr(self, "n_samples_seen_"):
            self.n_samples_seen_ = 0
            self.mean_ = 0.0
            self.var_ = 0.0

        # Update stats - they are 0 if this is the first step
        col_mean, col_var, n_total_samples = _incremental_mean_and_var(
            X,
            last_mean=self.mean_,
            last_variance=self.var_,
            last_sample_count=np.repeat(self.n_samples_seen_, X.shape[1]),
        )
        n_total_samples = n_total_samples[0]
        
        #print(f'mean & var = {time.time()-start}s')
        # Whitening
        if self.n_samples_seen_ == 0:
            # If it is the first step, simply whiten X
            X -= col_mean
        else:
            col_batch_mean = np.mean(X, axis=0)
            X -= col_batch_mean
            # Build matrix of combined previous basis and new data
            mean_correction = np.sqrt(
                (self.n_samples_seen_ / n_total_samples) * n_samples
            ) * (self.mean_ - col_batch_mean)
            X = np.vstack(
                (
                    self.singular_values_.reshape((-1, 1)) * self.components_,
                    X,
                    mean_correction,
                )
            )
        self.n_samples_seen_ +=len(X)

        #print(f'prep svd = {time.time()-start}s')
        U, S, Vt = self.svd(X)
        #print(f'calc svd = {time.time()-start}s')
        U, Vt = svd_flip(U, Vt, u_based_decision=False)
        explained_variance = S**2 / (self.n_samples_seen_ - 1)
        explained_variance_ratio = S**2 / np.sum(col_var * n_total_samples)

        self.components_ = Vt[: self.n_components_]
        self.singular_values_ = S[: self.n_components_]
        self.mean_ = col_mean
        self.var_ = col_var
        self.explained_variance_ = explained_variance[: self.n_components_]
        self.explained_variance_ratio_ = explained_variance_ratio[: self.n_components_]
        # we already checked `self.n_components <= n_samples` above
        if self.n_components_ not in (n_samples, n_features):
            self.noise_variance_ = explained_variance[self.n_components_ :].mean()
        else:
            self.noise_variance_ = 0.0
        #print(f'rest = {time.time()-start}s')
        return self

def _fit_estimator(y: np.ndarray, X: np.ndarray, estimator, w: Optional[np.ndarray]=None):
    from sklearn.base import clone
    estimator = clone(estimator)
    if w is None:
        estimator.fit(X, y)
    else:
        estimator.fit(X, y, w)
    return estimator

def smap(e, X, return_std):
    return e.predict(X, return_std)

def get_properties(model):
    return {k: v for k, v in model.__dict__.items() if k.endswith('_')}

class MultiOutputGenericWithStd(MetaEstimatorMixin, BaseEstimator):

    def __init__(self, estimator, *, n_jobs=-1):
        self.estimator = estimator
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray]=None):
        """Fit the model to data, separately for each output variable.

        Args:
          X: {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
          y: {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets. An indicator matrix turns on multilabel
            estimation.

        Returns: self.
        """
        from multiprocessing import Pool
        from functools import partial
        if y.ndim == 1:
            raise ValueError(
                "y must have at least two dimensions for "
                "multi-output regression but has only one."
            )
        with Pool(self.n_jobs) as p:
            this_fit_estimator = partial(_fit_estimator, X=X, estimator=self.estimator, w=weights)
            y_split = np.split(y, y.shape[1], axis=1)
            self.estimators_ = p.map(this_fit_estimator, y_split)

        return self

    def predict(self, X: np.ndarray, return_std: bool=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict multi-output variable using model for each target variable.

        Args:
          X: {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns: {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets predicted across multiple predictors.
            Note: Separate models are generated for each predictor.
        """
        from multiprocessing import Pool
        from functools import partial
        with Pool(self.n_jobs) as p:
            y = p.map(partial(smap, X=X, return_std=return_std), self.estimators_)
        if return_std:
            y, unc = zip(*y)
            return np.asarray(y).T, np.asarray(unc).T

        return np.asarray(y).T

