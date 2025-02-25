import argparse
import logging
import os

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

import wandb
from utils.training_utils import (
    get_batches_per_lang,
    load_config,
    set_random_seed,
    tokenize_and_encode,
)

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to a '.json' config for fine-tuning",
    )

    parser.add_argument(
        "--train_data_path",
        required=False,
        default="data/predictions_for_paper/train_no_leak.tsv",
    )

    parser.add_argument(
        "--eval_data_path",
        required=False,
        default="data/multiparadetox/dev_with_answers.tsv",
    )

    parser.add_argument(
        "--test_data_path",
        required=False,
        default="data/multiparadetox/test_without_answers.tsv",
    )

    parser.add_argument(
        "--skip_first",
        required=False,
        type=int,
        default=0,
    )

    parser.add_argument(
        "--train_on_multiparadetox",
        required=False,
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--full_training",
        required=False,
        action=argparse.BooleanOptionalAction,
    )

    args = parser.parse_args()

    config = load_config(args.config_path)

    logging.info(f"Setting random seed to {config['train_args']['seed']}")
    set_random_seed(config["train_args"]["seed"])

    logging.info("Loading model.")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_args"]["model_name"], legacy=False
    )
    logging.info(f"Loading training data: '{args.train_data_path}'")
    
    try:
        data = pd.read_csv(args.train_data_path)
    except pd.errors.ParserError:
        data = pd.read_csv(args.train_data_path, sep="\t")

    if not args.full_training:
        batches = get_batches_per_lang(data, batch_size=400, num_batches=10, multiparadetox=args.train_on_multiparadetox)
    else:
        batches = get_batches_per_lang(data, batch_size=400 * 10, num_batches=1)

    basedir = config["train_args"]["output_dir"]
    for batch_num, batch_data in enumerate(batches):
        # Skipping first batches in case the arg is passed
        if batch_num < int(args.skip_first):
            continue

        if args.full_training:
            batch_num = 'full_set'

        if args.train_on_multiparadetox:
            batch_num = 'golden'

        model = AutoModelForSeq2SeqLM.from_pretrained(
            config["model_args"]["model_name"]
        ).cuda()
        
        if isinstance(batch_num, int):
            logging.info(f"Training on batch {batch_num + 1}...")

        train_dataset = Dataset.from_pandas(batch_data)
        logging.info("Preprocessing train data")

        tokenize_and_encode_wrapper = lambda x: tokenize_and_encode(
            x, 
            tokenizer=tokenizer, 
            multiparadetox=args.train_on_multiparadetox,
        )
        encoded_train_dataset = train_dataset.map(
            tokenize_and_encode_wrapper, batched=False
        )
        encoded_train_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )

        config["train_args"]["output_dir"] = basedir + f"mt0_large_batch_{batch_num}"

        train_args = Seq2SeqTrainingArguments(
            **config["train_args"], remove_unused_columns=True
        )

        label_pad_token_id = tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if train_args.fp16 else None,
        )

        trainer = Seq2SeqTrainer(
            args=train_args,
            model=model,
            tokenizer=tokenizer,
            train_dataset=encoded_train_dataset,
            data_collator=data_collator,
        )

        run = wandb.init(
            name=f"mt0_large_batch_{batch_num}",
            config=config,
            project="llms_multiparadetox",
            tags=["mt0", f"batch_{batch_num}"],
        )
        wandb.watch(model, log=None, log_freq=100)

        logging.info(f"Training on batch {batch_num}")
        trainer.train()

        run.finish()

        if args.train_on_multiparadetox:
            break

    logging.info("Training complete.")
