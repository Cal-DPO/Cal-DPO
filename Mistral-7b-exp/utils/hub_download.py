import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import fire
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

# repo_url = "https://huggingface.co/datasets/bigcode/starcoderdata/tree/main"
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"





def download_file(token_id, repo_id, repo_type, file, cachedir, custom_path, revision):
    print(f"Downloading file: {file}")
    if revision:
        print(f"Downloading file: {file} with revision: {revision}")
        file_path = os.path.join(custom_path, repo_id.split("/")[-1] + revision[:5])
    else:
        print(f"Downloading file: {file} without revision")
        file_path = os.path.join(custom_path, repo_id.split("/")[-1])

    if not os.path.exists(file_path):
        os.mkdir(file_path)

    try:
        hf_hub_download(repo_id,
                        file,
                        revision=revision,
                        cache_dir=cachedir,
                        local_dir=file_path,
                        local_dir_use_symlinks=False,
                        repo_type=repo_type,
                        token=token_id
                        )
    except Exception as e:
        print(f"Error downloading file: {file}")
        print(e)


def get_huggingface(token_id, repo_id, repo_type, cachedir, custom_path, subdir=None, num_threads=10, revision=None):
    print(f"Downloading repo: {repo_id}")
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type, token=token_id, revision=revision)
    if subdir:
        all_files = [file for file in files if file.startswith(subdir)]
    else:
        all_files = files
    if revision:
        file_path = os.path.join(custom_path, repo_id.split("/")[-1] + revision[:5])
    else:
        file_path = os.path.join(custom_path, repo_id.split("/")[-1])

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for file in all_files:
            if os.path.exists(os.path.join(file_path, file)):
                print(f"File already exists: {file}")
                continue
            else:
                executor.submit(download_file, token_id, repo_id, repo_type, file, cachedir, custom_path, revision)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_id', type=str, required=True, default=None,
                        help='help text for arg1')
    parser.add_argument('--token_id', type=str, required=True, default=None,
                        help='help text for arg1')
    parser.add_argument('--revision', type=str, required=False, default=None,
                        help='help text for arg1')
    parser.add_argument('--repo_type', type=str, required=False, default="dataset",
                        help='[model, dataset] help text for arg2')
    parser.add_argument('--cachedir', type=str, required=False, default="/data/cache", help='help text for arg2')
    parser.add_argument('--custom_path', type=str, required=False,
                        default="/tmp_data/huggingface/cache", help='help text for arg2')
    parser.add_argument('--num_threads', type=int, required=False, default=10, help='number of threads')
    args = parser.parse_args()
    get_huggingface(args.token_id, args.repo_id, args.repo_type, args.cachedir, args.custom_path, num_threads=args.num_threads,
                    revision=args.revision)
