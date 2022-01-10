import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as lse


def cky_level(alphas:jnp.ndarray, W:int, T:int) -> jnp.ndarray:
    """
    First get left indices and right indices, then combine.
    See scratch/cky.py to play with indexing logic.
    """
    left_scores = alphas[
        jnp.arange(0,W-1),
        jnp.arange(T+1-W)[:,None],
    ]
    right_scores = alphas[
        jnp.arange(W-2,-1,-1),
        jnp.vstack([jnp.arange(x, x+W-1) for x in range(1, T+1-W+1)]),
    ]
    # number of start points x number of split points x num_states^2
    outer_product = (
        left_scores[:,:,:,None] + right_scores[:,:,None,:]
    ).reshape(T+1-W, W-1, -1)
    return lse((
        transition[None,:,None,:] + outer_product[:,None,:,:]
    ).reshape(T+1-W, num_states, -1), axis=-1)

def cky(alpha0):
    T, Z = alpha0.shape
    # alphas: span_width [T] x starting_index [T] x num_states [Z]
    alphas = jnp.full((T, T, Z), float("-inf"))
    # scan : (c -> a -> (c, b)) -> c -> [a] -> (c, [b])
    def f(carry, w):
        alphas, T = carry
        next_level = cky_level(alphas, w, T)
        return alphas.at[w-1, :T+1-w].set(next_level), next_level
    init = (alphas.at[0].set(alpha0), T)
    return jax.lax.scan(f, init, jnp.arange(1, T+1))
