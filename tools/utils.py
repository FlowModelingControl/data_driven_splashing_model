import numpy as np

def validate_input(X):
    """
    Helper to validate that input X is either
    - numpy array of shape (8, ) -> vector
    - numpy array of shape (n, 8) -> matrix, i.e. list of vectors

    :X:         input to be validated

    :returns:   (`valid`, `is_matrix`), i.e.
                - (True, False) for vector
                - (True, True) for matrix
    """
    # Make sure, input is numpy ndarray
    if isinstance(X, np.ndarray):
        if len(X.shape) == 1:
            # Vector
            if X.shape[0] == 8:
                return (True, False)
        elif len(X.shape) == 2:
            if X.shape[1] == 8:
                return (True, True)

    raise ValueError(
        "Input has incorrect shape. Please provide numpy ndarray of shape "
        "(8, ) or (n,  8)."
    )
