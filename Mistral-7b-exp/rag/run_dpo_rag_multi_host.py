#!/usr/bin/env python
# 
# Adapted from https://github.com/huggingface/alignment-handbook 
import logging
import os
import re
import sys

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed
import torch.distributed
from accelerate import Accelerator
from alignment import (
    DataArguments,
    DPORegConfig,
    ModelArguments,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
)

from alignment import DPORegTrainer


def apply_origin_text(
        example, tokenizer, task, assistant_prefix="<|assistant|>\n"
):
    example["text_real"] = example["chosen"][1]['content']
    example["text_prompt"] = example["chosen"][0]['content']
    example["text_real_neg"] = example["reject"][1]['content']
    return example


def apply_chat_template(
        example, tokenizer, task, assistant_prefix="<|assistant|>\n"
):
    def _strip_prefix(s, pattern):
        # Use re.escape to escape any special characters in the pattern
        return re.sub(f"^{re.escape(pattern)}", "", s)

    if all(k in example.keys() for k in ("chosen", "reject")):
        # Compared to reward modeling, we filter out the prompt, so the text is everything after the last assistant token
        prompt_messages = [[msg for msg in example["chosen"] if msg["role"] == "user"][0]]
        # Insert system message
        if example["chosen"][0]["role"] != "system":
            prompt_messages.insert(0, {"role": "system", "content": ""})
        else:
            prompt_messages.insert(0, example["chosen"][0])

        real_messages = example["chosen"][1:]
        neg_messages = example["reject"][1:]
        example["text_real"] = tokenizer.apply_chat_template(real_messages, tokenize=False)
        example["text_real_neg"] = tokenizer.apply_chat_template(neg_messages, tokenize=False)
        example["text_prompt"] = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        example["text_real"] = _strip_prefix(example["text_real"], assistant_prefix)
        example["text_real_neg"] = _strip_prefix(example["text_real_neg"], assistant_prefix)
    else:
        raise ValueError(
            f"Require `[chosen, reject]` keys but found {list(example.keys())}"
        )
    return example


logger = logging.getLogger(__name__)


def main():
    # model_args, data_args, training_args = parser.parse()
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, DPORegConfig))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.dataset_mixer is None and data_args.dataset_path:
        data_args.dataset_mixer = {
            data_args.dataset_path: 1.0
        }
    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Increase distributed timeout to 3h to enable push to Hub to complete
    # accelerator = Accelerator()

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(data_args, splits=data_args.dataset_splits)
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "task": "spin"},
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    # Replace column names with what TRL needs, text_real -> real and text_generated -> generated
    # for split in ["train", "test"]:
    print(raw_datasets)
    for split in ["train"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_real": "real", "text_real_neg": "real_neg", }
        )

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation="flash_attention_2" if model_args.use_flash_attention_2 else None,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = model_args.model_name_or_path

    ref_model = model
    ref_model_kwargs = model_kwargs

    if model_args.use_peft is True:
        ref_model = None
        ref_model_kwargs = None

    #########################
    # Instantiate spin trainer
    #########################
    dpo_rag_trainer = DPORegTrainer(
        model,
        ref_model,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta1=training_args.beta1,
        beta2=training_args.beta2,
        train_dataset=raw_datasets["train"],
        loss_type=training_args.loss_type,
        # eval_dataset=raw_datasets["test"],
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
    )

    ###############
    # Training loop
    ###############
    train_result = dpo_rag_trainer.train()
    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(raw_datasets["train"])
    )
    metrics["train_samples"] = min(max_train_samples, len(raw_datasets["train"]))
    dpo_rag_trainer.log_metrics("train", metrics)
    dpo_rag_trainer.save_metrics("train", metrics)
    dpo_rag_trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    dpo_rag_trainer.save_model(training_args.output_dir)
    # Save everything else on main process
    if training_args.local_rank == 0:
        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "dataset": list(data_args.dataset_mixer.keys()),
            "dataset_tags": list(data_args.dataset_mixer.keys()),
            "tags": ["alignment-handbook"],
        }
        dpo_rag_trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        dpo_rag_trainer.model.config.use_cache = True
        dpo_rag_trainer.model.config.save_pretrained(training_args.output_dir)

    # Ensure we don't timeout on model save / push to Hub
    logger.info("*** Waiting for all processes to finish ***")
    # accelerator.wait_for_everyone()
    if training_args.local_rank > 0:
        torch.distributed.barrier()
    logger.info("*** Run complete! ***")


if __name__ == "__main__":
    main()
