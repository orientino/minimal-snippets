# Minimal Snippets
Collection of snippets for ML research in PyTorch. Each file is self-contained and minimal. 

## CIFAR
100 lines of code to achieve ~94% test accuracy with ResNet18 using training, validation, and test splits. There's also a separate code to evaluate CIFAR-Corrupted datasets. Find more information in the `cifar/` folder.

## ImageNet1K
Distributed training to achieve ~78% test accuracy with ViT-S/16 for 90epochs in ~6hours, replicating and slightly improving the results of [Beyer et al. 2022](https://arxiv.org/pdf/2205.01580). The dataset is donwloaded from huggingface.

## Waterbirds
Image binary classification with the background feature as the spurious correlation. Our code automatically downloads the dataset, trains a ResNet50, and evaluates the test accuracy and worst-group accuracy.

## Yearbook
Image binary classification task with distribution shift across time from 1930 to 2013. The dataset contains ~33k samples of size (32x32x1). Our code automatically downloads the dataset, trains a ResNet18, and evaluates the test accuracy (~83%) and worst-group accuracy (~66%), where each group is a specific year.

## TinyStories
300 lines of code for language modeling using GPT architecture on [TinyStories](https://arxiv.org/abs/2305.07759) dataset with ~500M tokens. Our code automatically downloads the dataset and trains a model using the `gpt2` tokenizer.

## Landscape
Minimal PyTorch implementation of the paper [Visualizing the loss landscape of neural nets](https://arxiv.org/abs/1712.09913) by following the blog post [Math for machines](https://mathformachines.com/posts/visualizing-the-loss-landscape/). The code trains a simple MLP and projects the SGD training path into a 2D loss landscape.
