# Minimal Snippets
Collection of snippets for ML research in Pytorch. Each file is a single self-contained runnable code. 

## CIFAR
100 self-contained lines to achieve ~94% test accuracy with ResNet18 using training, validation, and test split. There's also a separate code to evaluate CIFAR-Corrupted datasets. Find more information in the `cifar/` folder.

## Waterbirds
Image binary classification with the background feature as the spurious correlation. Our code automatically downloads the dataset, trains a ResNet50, and evaluates the test accuracy and worst-group accuracy.

## Yearbook
Image binary classification task with distribution shift across time from 1930 to 2013. The dataset contains ~33k samples of size (32x32x1). Our code automatically downloads the dataset, trains a ResNet18, and evaluates the test accuracy (~83%) and worst-group accuracy (~66%), where each group is a specific year.

## TinyStories
Language modeling task with GPT architecture on [TinyStories](https://arxiv.org/abs/2305.07759) dataset with ~500M tokens. Our code automatically downloads the dataset, trains a GPT-like architecture using the `gpt2` tokenizer.

## Landscape
Minimal PyTorch implementation of the paper [Visualizing the loss landscape of neural nets](https://arxiv.org/abs/1712.09913) by following the blog post [Math for machines](https://mathformachines.com/posts/visualizing-the-loss-landscape/). The code trains a simple MLP and projects the SGD training path into 2D dimension loss landscape.
