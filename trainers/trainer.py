from tqdm import tqdm

import jax
import jax.numpy as jnp
import optax
import equinox as eqn

import pytorch_lightning as pl

from models.language_model import LanguageModel

def loss_fn(model, sentences, mask):
    log_probs, state = jax.vmap(model.score)(sentences)
    return jnp.sum(log_probs[mask])

@jax.jit
@jax.value_and_grad
def loss_and_grad(model, sentences, mask):
    return loss_fn(model, sentences, mask) / jnp.sum(mask)

@jax.jit
def loss(model, sentences, mask):
    return loss_fn(model, sentences, mask)

class Trainer:
    def __init__(
        self,
        model: LanguageModel,
        optimizer: optax.GradientTransformation,
        data_module: pl.LightningDataModule,
        use_wandb: bool = False,
    ):
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project = "switching-lms",
                config = {
                    "dataset": "wikitext-2-toy",
                },
            )

        self.model = model
        self.data_module = data_module

        self._steps = 0 # number of gradient steps

        self.optimizer = optimizer
        self.opt_state= optimizer.init(model)


    def loop(self, data, update=False):
        total_loss = 0
        total_n_words = 0
        grad_norm = None
        for iter, batch in enumerate(tqdm(data)):
            mask = batch["attention_mask"]
            sentences = batch["input_ids"]
            nwords = jnp.sum(mask)
            # DBG
            if update:
                loss, grads = loss_and_grad(self.model, sentences, mask)
                updates, self.opt_state = self.optimizer.update(
                    grads, self.opt_state, self.model)
                self.model = eqn.apply_updates(self.model, updates)

                #grad_norm = jnp.linalg.norm(grads)

                # number of gradient steps
                self._steps += 1

                # scale back up, since for the gradient we need to average
                loss = loss * nwords
            else:
                loss = loss(self.model, sentences, mask)
            total_loss += loss
            total_n_words += nwords
            if update and self.use_wandb:
                wandb.log({
                    "total_loss": total_loss / total_n_words,
                    #"grad_norm": grad_norm,
                }, step = self._step)
        return total_loss, total_n_words

    def fit(self, epochs=15):
        for epoch in range(epochs):
            loss, n_words = self.loop(
                self.data_module.train_dataloader(),
                update=True,
            )
            # perform end of epoch logging
            import pdb; pdb.set_trace()


