
from typing import Optional

import jax.numpy as jnp
import equinox as eqx # type: ignore

from models.language_model import LanguageModel

class HMM(LanguageModel):
    start: jnp.ndarray
    transition: jnp.ndarray
    emission: jnp.ndarray

    def score(
        self,
        sentence: jnp.ndarray,
    ) -> float:
        """ Score a sentence by computing
            log p(sentence) = \sum_t log p(word_t | words_<t)
        """
        # include BOS and EOS for now
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
