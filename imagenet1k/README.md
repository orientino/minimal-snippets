### Download the dataset
The simplest way is to use [timm/imagenet-1k-wds](https://huggingface.co/datasets/timm/imagenet-1k-wds) . First, create and setup a HF account. Run 
```bash
huggingface-cli download timm/imagenet-1k-wds --repo-type dataset --local-dir /path/to/imagenet-wds
```
Adjust the run.sh for your own cluster setup and you should be good to run.