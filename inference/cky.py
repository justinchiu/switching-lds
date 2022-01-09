import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as lse


def cky_level(T, W, alphasT):
    """
    First get left indices and right indices, then combine
    ```
    left_idxs = idxsT[np.arange(0,W-1),np.arange(T+1-W)[:,None]]
    right_idxs = idxsT[
        np.arange(W-2,-1,-1),
        np.vstack((np.arange(x, x+W-1) for x in range(1,T+1-W+1))),
    ]
    ```
    """
    left_scores1 = alphasT[np.arange(0,W-1),np.arange(T+1-W)[:,None]]
    right_scores1 = alphasT[
        np.arange(W-2,-1,-1),
        np.vstack([np.arange(x, x+W-1) for x in range(1, T+1-W+1)]),
    ]
    # number of start points x number of split points x num_states^2
    outer_product1 = (
        left_scores1[:,:,:,None] + right_scores1[:,:,None,:]
    ).reshape(T+1-W, W-1, -1)
    return lse((
        transition[None,:,None,:] + outer_product1[:,None,:,:]
    ).reshape(T+1-W, num_states, -1), axis=-1)
