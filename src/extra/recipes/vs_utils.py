from typing import Optional, Union, Tuple
import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin

def _fit_estimator(estimator, X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray]=None):
    from sklearn.base import clone
    estimator = clone(estimator)
    if w is None:
        estimator.fit(X, y)
    else:
        estimator.fit(X, y, w)
    return estimator

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
        from joblib import Parallel, delayed
        if y.ndim == 1:
            raise ValueError(
                "y must have at least two dimensions for "
                "multi-output regression but has only one."
            )
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(
                self.estimator, X, y[:, i], weights
            )
            for i in range(y.shape[1])
        )

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
        from joblib import Parallel, delayed
        y = Parallel(n_jobs=self.n_jobs)(
            delayed(e.predict)(X, return_std) for e in self.estimators_
        )
        if return_std:
            y, unc = zip(*y)
            return np.asarray(y).T, np.asarray(unc).T

        return np.asarray(y).T

