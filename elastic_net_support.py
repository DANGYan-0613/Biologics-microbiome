"""Exact Elastic Net selector fitting from the completed LOGO benchmark."""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

COEFFICIENT_TOLERANCE = 1e-8
EN_SEED = 20261125


def fit_one(x,y,candidate):
    if x.columns.duplicated().any() or not all(n.startswith(("species__","pwy__")) for n in x):
        raise ValueError("Only unique microbial columns are allowed")
    if set(y)!={0,1} or not np.isfinite(x.to_numpy()).all(): raise ValueError("Invalid training matrix or target")
    messages=[]
    for maximum in (30000,100000):
        model=LogisticRegression(solver="saga",C=float(candidate["C"]),l1_ratio=float(candidate["l1_ratio"]),
            class_weight=None,random_state=EN_SEED,max_iter=maximum,tol=1e-4)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always",ConvergenceWarning)
            model.fit(x.to_numpy(),y)
        messages.extend(str(w.message) for w in seen)
        converged=not any(issubclass(w.category,ConvergenceWarning) for w in seen) and int(model.n_iter_.max())<maximum
        if converged:
            names=x.columns[np.abs(model.coef_[0])>COEFFICIENT_TOLERANCE].tolist()
            return model,names,dict(status="valid",n_iter=int(model.n_iter_.max()),iteration_limit=maximum,
                                   diagnostic_warnings="; ".join(messages))
    return None,[],dict(status="failed",n_iter=int(model.n_iter_.max()),iteration_limit=maximum,
                       diagnostic_warnings="; ".join(messages),error="Elastic Net failed to converge at the predeclared numerical retry limit")
