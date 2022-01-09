# scratch dev for batched inside algorithm

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as lse

from inference.cky import cky_level

key = jax.random.PRNGKey(1234)

num_states = 7

# all scores are unnormalized
transition = jax.random.uniform(
    key,
    shape=(num_states, num_states ** 2),
    minval=-2,
    maxval=-1,
)

T = 5

# First attempt: alphas: span_length x start_position x state

# alphas[w] contains span scores for spans of size w+1
alphas = [
    jnp.full((T+1-span_width, num_states), float("-inf"))
    for span_width in range(1, T+1)
]

idxs = [ [
    (start_idx, start_idx + span_width)
    for start_idx in range(0, T+1-span_width)
] for span_width in range(1, T+1) ]

# focus on
W = 3

# randomly initialize scores for spans of length 1 and 2
for w in range(W):
    alphas[w] = jax.random.uniform(
        key,
        shape=(alphas[w].shape),
        minval = -2,
        maxval = -1,
    )

# test single case of W = 3, looping solution
num_terms = T+1-W
alpha3 = jnp.full((num_terms, num_states), float("-inf"))
for start_idx in range(num_terms):
    for left_len in range(1, W):
        right_len = W - left_len
        split_point = start_idx + left_len

        # make sure the spans are correct
        left_span = (start_idx, split_point)
        right_span = (split_point, start_idx + W)
        assert(left_span == idxs[left_len-1][start_idx])
        assert(right_span == idxs[right_len-1][split_point])
        print(left_span, right_span)

        # compute p(BC | A)
        left_scores = alphas[left_len-1][start_idx]
        right_scores = alphas[right_len-1][split_point]
        outer_product = left_scores[:,None] + right_scores[None]
        new_scores = lse(transition + outer_product.reshape((-1,)), axis=-1)

        # update without mutation
        alpha3 = alpha3.at[start_idx].set(jnp.logaddexp(
            alpha3[start_idx],
            new_scores
        ))

# copy alphas to a tensor
# wastes memory but easier for indexing
alphasT = jnp.full((T, T, num_states), float("-inf"))
for w in range(T):
    alphasT = alphasT.at[w, :T-w].set(alphas[w])
idxsT = jnp.full((T, T, 2), float("-inf"))
for w in range(T):
    idxsT = idxsT.at[w, :T-w].set(idxs[w])

# batch over split points
# span lengths, starting point
left_idx = idxsT[np.arange(W-1),0]
# span lengths, starting points
right_idx = idxsT[np.arange(1,-1,-1), np.arange(1,3)]
print(left_idx)
print(right_idx)
left_scores = alphasT[np.arange(W-1), 0].reshape(2, num_states, 1)
right_scores = alphasT[np.arange(1,-1,-1), np.arange(1,3)].reshape(2, 1, num_states)
outer_product = (left_scores + right_scores).reshape(2, -1)
new_scores = lse(transition[:,None,:] + outer_product[None,:,:], axis=-1)
alpha30 = lse(new_scores, axis=-1)

print(jnp.allclose(alpha3[0], alpha30))

# batch over a whole level
left_idxs = idxsT[np.arange(0,W-1),np.arange(T+1-W)[:,None]]
right_idxs = idxsT[
    np.arange(W-2,-1,-1),
    np.vstack((np.arange(x, x+W-1) for x in range(1,T+1-W+1))),
]
print(left_idxs)
print(right_idxs)
left_scores1 = alphasT[np.arange(0,W-1),np.arange(T+1-W)[:,None]]
right_scores1 = alphasT[
    np.arange(W-2,-1,-1),
    np.vstack([np.arange(x, x+W-1) for x in range(1, T+1-W+1)]),
]
# number of start points x number of split points x num_states^2
outer_product1 = (
    left_scores1[:,:,:,None] + right_scores1[:,:,None,:]
).reshape(T+1-W, W-1, -1)
new_scores1 = lse(transition[None,:,None,:] + outer_product1[:,None,:,:], axis=-1)
alpha31 = lse(new_scores1, axis=-1)
alpha32 = lse((
    transition[None,:,None,:] + outer_product1[:,None,:,:]
).reshape(T+1-W, num_states, -1), axis=-1)
print(jnp.allclose(alpha3, alpha31))
print(jnp.allclose(alpha3, alpha32))
print(jnp.max(jnp.abs(alpha32 - alpha31)))

# test cky level
alphasT1 = alphasT.at[2, :T+1-3].set(cky_level(T, 3, alphasT))
alphasT1 = alphasT1.at[3, :T+1-4].set(cky_level(T, 4, alphasT1))
alphasT1 = alphasT1.at[4, :T+1-5].set(cky_level(T, 5, alphasT1))

print(jnp.allclose(alphasT1[2,:3], alpha32))
