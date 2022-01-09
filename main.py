# Adapted from https://github.com/yang-zhang/lightning-language-modeling/blob/main/language_model.py

from argparse import ArgumentParser
import jax
import optax

import pytorch_lightning as pl
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    AutoConfig,
)

from data import LMDataModule
from trainers.trainer import Trainer


def main():

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str,
                        default="distilbert-base-cased")
    parser.add_argument('--train_file', type=str,
                        default="data/wikitext-2/wiki.train.small.raw")
    parser.add_argument('--validation_file', type=str,
                        default="data/wikitext-2/wiki.valid.small.raw")
    parser.add_argument('--no_line_by_line', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--pad_to_max_length', action='store_true', default=False)
    parser.add_argument('--preprocessing_num_workers', type=int, default=2)
    parser.add_argument('--overwrite_cache', action='store_true', default=False)
    parser.add_argument('--max_seq_length', type=int, default=32)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--dataloader_num_workers', type=int, default=4)

    parser.add_argument('--num_states', type=int, default=32)
    parser.add_argument('--key', type=int, default=1234)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=15)
    args = parser.parse_args()

    pl.seed_everything(args.key)

    # ------------
    # data
    # ------------
    data_module = LMDataModule(
        model_name_or_path=args.model_name_or_path,
        train_file=args.train_file,
        validation_file=args.validation_file,
        line_by_line=not args.no_line_by_line,
        pad_to_max_length=args.pad_to_max_length,
        preprocessing_num_workers=args.preprocessing_num_workers,
        overwrite_cache=args.overwrite_cache,
        max_seq_length=args.max_seq_length,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
    )
    data_module.setup()
    train_data = data_module.train_dataloader()
    valid_data = data_module.val_dataloader()
    #test_data = data_module.test_dataloader()

    """
    notes
    iterate over *_data as follows:
    ```
    for batch in iter(train_data):
        pad_mask = batch["attention_mask"]
        text = batch["input_ids"]
    ```
    special tokens:
    * text[:,0] = all <BOS> / 101
    * <EOS> is 102
    * text == 0 is padding
    """

    # ------------
    # model
    # ------------
    V = data_module.tokenizer.vocab_size
    Z = args.num_states

    key = jax.random.PRNGKey(args.key)

    from models.hmm import Hmm
    model_init = Hmm(Z, V, key)
    from models.pcfg import Pcfg
    model_init = Pcfg(Z, V, key)
    # this will not be mutated, need to get model from trainer.model

    # ------------
    # training
    # ------------
    trainer = Trainer(model_init, optax.adamw(args.lr), data_module)
    trainer.fit(args.epochs)

    # ------------
    # cleanup
    # ------------
    data_module.teardown()


if __name__ == '__main__':
    main()

