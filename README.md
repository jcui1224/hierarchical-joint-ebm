# hierarchical-joint-ebm
## Train NVAE



## Train EBM

### 1. LSUN Church 64

```python
CUDA_VISIBLE_DEVICES=gpu0, gpu1 python ebm_train.py --backbone=Church64_Gaussian_Decoder

```

### 2. CIFAR 10

```python
CUDA_VISIBLE_DEVICES=gpu0, gpu1 python ebm_train.py --backbone=CIFAR10_Gaussian_Decoder
```

