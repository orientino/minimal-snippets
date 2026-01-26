### Download the dataset
We use [timm/imagenet-1k-wds](https://huggingface.co/datasets/timm/imagenet-1k-wds) . First, use your HF account to accept the terms and conditions. Run
```bash
# download the dataset
huggingface-cli download timm/imagenet-1k-wds --repo-type dataset --local-dir /path/to/imagenet-wds
```
Adjust the run.sh for your own cluster setup and you are good to go.