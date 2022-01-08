import jax
import jax.numpy as jnp
import optax
import equinox as eqn

import pytorch_lightning as pl

from models.language_model import LanguageModel


class Trainer:
    def __init__(
        self,
        model: LanguageModel,
        optimizer: optax.GradientTransformation,
        data_module: pl.LightningDataModule,
    ):
        self.model = model
        self.optimizer = optimizer
        self.data_module = data_module

        self._steps = 0 # number of gradient steps

    # overload this in subclasses
    def loss_fn(model, sentences, mask):
        log_probs = jax.vmap(model.score)(sentences, lengths)
        return jnp.sum(log_probs[mask])

    @jax.jit
    @jax.value_and_grad
    def loss_and_grad(model, sentences, mask):
        return self.loss_fn(model, sentences, mask)

    @jax.jit
    def loss(model, sentences, mask):
        return self.loss_fn(model, sentences, mask)

    def loop(self, data, update=False):
        total_loss = 0
        total_n_words = 0
        for iter, batch in enumerate(data):
            mask = batch["attention_mask"]
            sentences = batch["input_ids"]
            nwords = jnp.sum(mask)
            if update:
                loss, grads = self.loss_and_grad(self.model, sentences, mask)
                # divide grad by nwords
                # clip grad (can we merge these with optimizer?)
                # apply optimizer
                eqn.apply_updates(self.model, self.optimizer(grads))

                # number of gradient steps
                self._steps += 1
            else:
                loss = self.loss(self.model, sentences, mask)
            loss += total_loss
            total_n_words += nwords
        return total_loss, total_n_words

    def fit(self, model, epochs=15):
        for epoch in range(epochs):
            loss, n_words = self.loop(
                self.data_module.train_dataloader(),
                update=True,
            )
            # perform logging


