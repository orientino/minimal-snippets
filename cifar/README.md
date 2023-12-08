### Training
Run `python train_c10.py`

### Evaluation
Download the corrupted dataset [CIFAR10C](https://zenodo.org/records/2535967) and [CIFAR100C](https://zenodo.org/records/3555552)

Run `python eval_c10_corrupted.py`

### Results
CIFAR `test_set` and the variations of `corrupted_set` with severity = 1

|                   | CIFAR10  | CIFAR100 |
|-------------------|----------|----------|
| test_set          | 0.9458   | 0.7618   |
| gaussian_noise    | 0.8578   | 0.6035   |
| shot_noise        | 0.8807   | 0.6377   |
| impulse_noise     | 0.8667   | 0.6349   |
| defocus_blur      | 0.9263   | 0.7226   |
| glass_blur        | 0.7496   | 0.5199   |
| motion_blur       | 0.9133   | 0.6959   |
| zoom_blur         | 0.9105   | 0.7044   |
| snow              | 0.8807   | 0.6619   |
| frost             | 0.8992   | 0.6619   |
| fog               | 0.9254   | 0.7241   |
| brightness        | 0.9264   | 0.7197   |
| contrast          | 0.9196   | 0.7178   |
| elastic_transform | 0.8899   | 0.6670   |
| pixelate          | 0.9089   | 0.6945   |
| jpeg_compression  | 0.8538   | 0.6061   |
