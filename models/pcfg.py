
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as lse
import equinox as eqx # type: ignore

from models.language_model import LanguageModel
from inference.cky import cky

class Pcfg(LanguageModel):
    start: jnp.ndarray
    transition: jnp.ndarray
    emission: jnp.ndarray

    def __init__(self, num_states, num_outputs, key):
        super().__init__()

        start_key, transition_key, emission_key = jax.random.split(key, 3)
        epsilon = 1e-3

        start = jax.random.uniform(
            start_key,
            shape = (num_states,),
            minval = -epsilon,
            maxval = epsilon,
        )
        transition = jax.random.uniform(
            transition_key,
            shape = (num_states, num_states * num_states),
            minval = -epsilon,
            maxval = epsilon,
        )
        emission = jax.random.uniform(
            emission_key,
            shape = (num_states, num_outputs),
            minval = -epsilon,
            maxval = epsilon,
        )

        self.start = start - lse(start)
        self.transition = transition - lse(transition, axis=-1, keepdims=True)
        self.emission = emission - lse(emission, axis=-1, keepdims=True)

    def score(
        self,
        sentence: jnp.ndarray,
        state: Optional[jnp.ndarray] = None,
    ) -> float:
        """ Score a sentence by computing
            log p(sentence) = \sum_t log p(word_t | words_<t)
        """
        # include BOS and EOS for now

        emissions = self.emission[:, sentence]
        (chart, T), levels = cky(emissions)
        import pdb; pdb.set_trace()

        return log_p_words, None


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

