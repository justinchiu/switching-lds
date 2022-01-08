
from typing import Optional

import jax.numpy as jnp
import equinox as eqx # type: ignore

class LanguageModel(eqx.Module):
    """ Interface class for language models
        Must implement the score and log_p_next functions.
    """

    def score(
        self,
        sentence: jnp.ndarray,
        pad_mask: jnp.ndarray = None,
        length: Optional[int] = None,
    ) -> float:
        """ Score a sentence by computing
            log p(sentence) = \sum_t log p(word_t | words_<t)
        """
        raise NotImplementedError

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
