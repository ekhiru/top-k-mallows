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

def select_model(mid, n):
  N = int(n*(n-1)/2) # max dist ken
  if mid == 0:
    phi = mk.find_phi(n, N/10, N/10+1)
    mname, params, mtext, mtextlong = 'mm_ken', phi , 'MM_peaked', 'Mallows model, peaked'
  elif mid == 1:
    phi = mk.find_phi(n, N/4, N/4+1)
    mname, params, mtext, mtextlong = 'mm_ken', phi , 'MM_unif', 'Mallows model, disperse'
  elif mid == 2:
    phi = mk.find_phi(n, N/10, N/10+1)
    theta = mk.phi_to_theta(phi)
    theta = [np.exp(theta/(i+1)) for i in range(n-1)] #+ [0]
    mname, params, mtext, mtextlong = 'gmm_ken', theta , 'GMM_peaked', 'Generalized Mallows model, peaked'
  elif mid == 3:
    phi = mk.find_phi(n, N/4, N/4+1)
    theta = mk.phi_to_theta(phi)
    theta = [theta/(i+1) for i in range(n-1)] #+ [0]
    mname, params, mtext, mtextlong = 'gmm_ken', theta , 'GMM_unif', 'Generalized Mallows model, disperse'
  elif mid == 4:
    w = np.array([np.exp(n-i) for i in range(n)])
    mname, params, mtext, mtextlong = 'pl', w , 'PL_peaked', 'Plackett-Luce, peaked'
  elif mid == 5:
    w = np.array([(n-i) for i in range(n)])
    mname, params, mtext, mtextlong = 'pl', w , 'PL_unif', 'Plackett-Luce, disperse'
  return mname, params, mtext, mtextlong


def sample(n,m,model='mm_ken', params=None):
    #m: mum perms, n: perm size; model='mm_ken'
    # sample = np.zeros((m,n))
    if model == 'mm_ken':
        return mk.sample(m=m,n=n,phi=params)
    elif model == 'pl':
        return plackett_luce_sample(m,n,w=params)
    elif model == 'gmm_ken':
        return mk.sample(m=m,n=n,theta=params)
    return

def plackett_luce_sample(m,n,w=None):
  if w is None: w = np.array([np.exp(i) for i in reversed(range(n))])
  sample = np.zeros((m,n))
  for m_ in range(m):
      ordering = []
      bucket = np.arange(n, dtype=int) #list of items to insert
      for i in range(n):
        j = np.random.choice(bucket,p=w[bucket]/w[bucket].sum())
        ordering.append(j)
        bucket = bucket[bucket!=j]
      sample[m_] = np.argsort(ordering).copy()
  return sample

def pl_proba(perm, w):
    n = len(perm)
    ordering = np.argsort(perm)
    return np.prod([  w[ordering[i]]/w[ordering[i:]].sum()  for i in range(n)])


def sample_to_marg_rel(sample):
  m,n = sample.shape
  P = np.zeros((n,n))
  for i in range(n):
    for j in range(i+1,n):
      P[i,j] = (sample[:,i]<sample[:,j]).mean()
      P[j,i] = 1 - P[i,j]
#   print("triangles",np.tril(P).sum(),np.tril(P).sum())
  return P

def dist_to_sample(perm,P,dist='k'):
  if dist=='k':
    return np.tril(P[np.ix_(np.argsort(perm),np.argsort(perm))],k=-1).sum()




def full_perm_path(n):
#     perm = np.random.permutation(n)
# [mk.kendall_tau(perm,p[perm]) for p in pu.full_perm_path(n)]
# ?this is alway
  perm = list(range(n))
  drifts = [perm[:]]
  while perm != list(range(n))[::-1]:
    i = np.random.choice(n-1)
    while perm[i]>perm[i+1]:
      i = np.random.choice(n-1)
    perm[i], perm[i+1] = perm[i+1],perm[i]
    drifts.append(perm[:])
  return   [np.argsort(perm) for perm in drifts]
#   [(np.argsort(perm), mk.kendall_tau(np.argsort(perm))) for perm in drifts]


def get_P(n,model='mm_ken',params=None):
  def h(k,phi):return k/(1-phi**k)
  pairw = np.empty((n,n))
  pairw[:] = np.nan
  if model=='mm_ken':
    phi = params
    # theta, phi = mk.check_theta_phi(theta, phi)
    for i in range(n):
      for j in range(i+1,n):
        pairw[i,j] = h(j-i+1,phi) - h(j-i,phi)
        pairw[j,i] = 1-pairw[i,j]
  elif model == 'pl': # generate a pairwise
    for i in range(n):
      for j in range(i+1,n):
        pairw[i,j] = params[i] / (params[i]+params[j])
        pairw[j,i] = 1-pairw[i,j]
  return pairw



#end
