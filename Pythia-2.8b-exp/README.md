# Cal-DPO: Calibrated Direct Preference Optimization for  Language Model Alignment

## Train

#### Run Experiment on HH

- Run SFT 

```bash
python -u train.py model=pythia28 datasets=[hh] loss=sft exp_name=hh_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16
```

- Run Cal-DPO

```bash
python -u train.py model=pythia28 datasets=[hh] loss=dpo loss.beta=0.1 exp_name=hh_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=/path/to/archive/from/sft/LATEST/policy.pt
```


#### Run Experiment on TL;DR

- Download & Processing Dataset

Follow instruction provided in [Towards_Efficient_and_Exact_Optimization_of_Language_Model_Alignment](https://github.com/haozheji/exact-optimization/blob/main/exp/tldr_exp/README.md)

```python
from datasets import load_dataset

tldr_dataset = load_dataset("openai/summarize_from_feedback", 'comparisons')
tldr_dataset['train'].to_json('summarize_from_feedback/train.json')
tldr_dataset['validation'].to_json('summarize_from_feedback/test.json')

tldr_filtered_dataset = load_dataset("UCL-DARK/openai-tldr-filtered")
tldr_filtered_dataset['train'].to_json('openai-tldr-filtered/train.jsonl')
tldr_filtered_dataset['test'].to_json('openai-tldr-filtered/test.jsonl')
```

Run the followiung command to preprocess the data, and put the processed data folder *tldr* and *tldr_filtered* under *datasets*

```bash
cd datasets
python preproc_pref.py /path/to/summarize_from_feedback
python preproc_filter.py /path/to/openai-tldr-filtered
```


- Run SFT 

```bash
python -u train.py model=pythia28 datasets=[tldr] loss=sft exp_name=sft_tldr_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16
```

- Run Cal-DPO

```bash
python -u train.py model=pythia28 datasets=[tldr] loss=dpo loss.beta=0.001 exp_name=align_beta0.001_tldr_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=/path/to/archive/from/sft/LATEST/policy.pt
```

---

## Evaluate

```bash
python -u generate.py model=pythia28 datasets=[hh] eval_batch_size=32 sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=/path/to/archive/from/align/LATEST/policy.pt
```