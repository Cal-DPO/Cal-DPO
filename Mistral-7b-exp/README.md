## Cal-DPO: Calibrated Direct Preference Optimization for  Language Model Alignment


## Installation
To install the project, you need to clone the repository and install the dependencies. 
```bash
pip install -e .
```

## Usage

### 1. Download the data
To download the data, you need to run the following command:
```bash
python utils/hub_download.py --token_id <token_id> --repo_id HuggingFaceH4/ultrafeedback_binarized --repo_type dataset --custom_path /data
```
`token_id` is the token id of the huggingface, you can get it from the huggingface profile. `repo_id` is the id of the dataset you want to download. `repo_type` is the type of the repository, it can be either `dataset` or `model`. `custom_path` is the path where you want to download the data.


### 2. Training the model

To train the model, you need to run the following command:
```bash
sh scripts/finetune_dpo_reg.sh
```


### 3. Evaluation

For our evaluation on the Open LLM Leaderboard, please use the lm-evaluation-harness repository at v0.4.0. Also, note that we set the number of few shot examples to be the same as instructed on the Leaderboard. 

to evaluate the model, you need to run the following command:
```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
cp /path/dpo_rag/eval/eval_task.sh .
sh eval_task.sh /path/dpo_rag/finetuned_model task_id finetuned_model_id $num_gpu
```

all the arguments are required. `task_id` is the id of the task you want to evaluate. `finetuned_model_id` is the id of the finetuned model you want to evaluate. `num_gpu` is the number of gpus you want to use for evaluation.

And the few shot config, you need to modify the corresponding yaml file in `lm-evaluation-harness`.