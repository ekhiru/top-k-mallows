import mallows_kendall as mk
import numpy as np
import itertools as it

def compose(s, p):
    """This function composes two given permutations
    Parameters
    ----------
    s: ndarray
        First permutation array
    p: ndarray
        Second permutation array
    Returns
    -------
    ndarray
        The composition of the permutations
    """
    return np.array(s[p])

def compose_partial(partial, full):
    """ This function composes a partial permutation with an other (full)
        Parameters
        ----------
        partial: ndarray
            Partial permutation (should be filled with float)
        full:
            Full permutation (should be filled with integers)
        Returns
        -------
        ndarray
            The composition of the permutations
    """
    return [partial[i] if not np.isnan(i) else np.nan for i in full]

def inverse(s):
    """ This function computes the inverse of a given permutation
        Parameters
        ----------
        s: ndarray
            A permutation array
        Returns
        -------
        ndarray
            The inverse of given permutation
    """
    return np.argsort(s)

def inverse_partial(sigma):
    """ This function computes the inverse of a given partial permutation
        Parameters
        ----------
        sigma: ndarray
            A partial permutation array (filled with float)
        Returns
        -------
        ndarray
            The inverse of given partial permutation
    """
    inv = np.full(len(sigma), np.nan)
    for i, j in enumerate(sigma):
        if not np.isnan(j):
            inv[int(j)] = i
    return inv





#end