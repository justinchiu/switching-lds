
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as lse
import equinox as eqx # type: ignore

from models.language_model import LanguageModel

class Hmm(LanguageModel):
    start: jnp.ndarray
    transition: jnp.ndarray
    emission: jnp.ndarray

    def __init__(self, num_states, num_outputs, key):
        start_key, transition_key, emission_key = jax.random.split(key, 3)
        epsilon = 1e-3
        self.start = jax.random.uniform(
            start_key,
            shape = (num_states,),
            minval = -epsilon,
            maxval = epsilon,
        )
        self.transition = jax.random.uniform(
            transition_key,
            shape = (num_states, num_states),
            minval = -epsilon,
            maxval = epsilon,
        )
        self.emission = jax.random.uniform(
            emission_key,
            shape = (num_states, num_outputs),
            minval = -epsilon,
            maxval = epsilon,
        )

    def f(log_state: jnp.ndarray, x: int) -> Tuple[jnp.ndarray, float]:
        log_emit = log_state[:,None] + self.emission[:,x]
        log_Z = lse(log_emit)
        un_log_next_state = lse(log_emit[:,None] + self.transition, axis=0)
        log_next_state = un_log_next_state - log_Z
        return (log_next_state, log_Z)

    def score(
        self,
        sentence: jnp.ndarray,
        state: Optional[jnp.ndarray],
    ) -> float:
        """ Score a sentence by computing
            log p(sentence) = \sum_t log p(word_t | words_<t)
        """
        # include BOS and EOS for now
         
        # c = p(z), b = p(x)
        # scan : (c -> a -> (c, b)) -> c -> [a] -> (c, [b])
        # f : c -> a -> (c, b)

        next_state, log_p_words = jax.lax.scan(
            f = self.f,
            init = state if state else self.start,
            xs = sentence,
        )
        return log_p_words, log_next_state


    def log_p_next(
        self,
        prefix: jnp.ndarray,
        pad_mask: jnp.ndarray = None,
        length: Optional[int] = None,
    ) -> jnp.ndarray:
        """ Predict the next word given a prefix
            log p(next | prefix)
        """
        raise NotImplementedError
