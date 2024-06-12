import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
from utils import get_local_dir, disable_dropout, init_distributed, get_open_port, get_local_run_dir_gen
from utils import get_local_run_dir_gen as get_local_run_dir

import os
import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig
import trainers
import wandb
import json
import socket
from typing import Optional, Set
import resource

from preference_datasets import get_batch_iterator

OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))

n_examples=None
save_name = 'eval'

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""

    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")


    print(OmegaConf.to_yaml(config))

    
    config.local_run_dir = '/'.join([*config.model.archive.split("/")[:-3],'eval_all', config.datasets[0], config.model.archive.split("/")[-3]])
    print('config.local_run_dir',config.local_run_dir)
    os.makedirs(config.local_run_dir, exist_ok=True)

    # config_path = os.path.join(config.local_run_dir, 'config.yaml')
    # with open(config_path, 'w') as f:
    #     OmegaConf.save(config, f)

    print('=' * 80)
    print(f'Writing to {socket.gethostname()}:{config.local_run_dir}')
    print('=' * 80)
 
    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.local_dirs)
    print('building policy')
    model_kwargs = {'device_map': 'balanced'} if config.trainer == 'BasicTrainer' else {}
    policy_dtype = getattr(torch, config.model.policy_dtype)
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=policy_dtype, **model_kwargs)
    disable_dropout(policy)

    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'left'
        tokenizer.truncation_side = 'left'


    print('Loading from {}'.format(config.model.archive))
    state_dict = torch.load(config.model.archive, map_location='cpu')
    step, metrics = state_dict['step_idx'], state_dict['metrics']
    print(f'loading pre-trained weights at step {step} from {config.model.archive} with metrics {json.dumps(metrics, indent=2)}')
    policy.load_state_dict(state_dict['state'])
    print('loaded pre-trained weights')

    print(policy)
    

    data_iterator_kwargs = dict(
        names=config.datasets,
        tokenizer=tokenizer,
        shuffle=False,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        sft_mode=True
    )

    rank = torch.cuda.device_count()
    eval_iterator = get_batch_iterator(**data_iterator_kwargs, split='test', n_examples=n_examples, n_epochs=1,  batch_size=config.eval_batch_size, silent=rank != 0, cache_dir=get_local_dir(config.local_dirs)) #config.eval_batch_size
    eval_batches = list(eval_iterator)

    response_list=[]
    for data in eval_batches:

        question_encoding = tokenizer(data["prompt"], padding=True, return_tensors='pt') #tokenizer(["hello"], return_tensors="pt", add_special_tokens=False).input_ids.to(policy.device)
        with torch.no_grad():
            generated_ids = policy.generate(input_ids=question_encoding["input_ids"].to(policy.device), 
                                            attention_mask=question_encoding["attention_mask"].to(policy.device),
                                            max_new_tokens=100)[:,question_encoding["input_ids"].shape[1]:]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        question_response = []
        for q,a,c in zip(data["prompt"], response, data["chosen"]):
            question_response.append({"prompt": q, "response":a.strip(), "chosen":c.replace(q,'').strip()})
        response_list.extend(question_response)

    print(response_list)
    print(len(response_list))

    config_path = os.path.join(config.local_run_dir, '{}.json'.format(save_name))
    with open(config_path, 'w') as f:
        json.dump(response_list, f, indent=4)


if __name__ == '__main__':
    main()