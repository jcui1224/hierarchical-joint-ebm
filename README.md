# hierarchical-joint-ebm
## NVAE Checkpoint

NVAE with discrete mixture logistic decoder can be downloaded [here](https://github.com/NVlabs/NVAE); NVAE with Gaussian decoder can be trained by using the [original code](https://github.com/NVlabs/NVAE) with hyper-parameters provided by [VAEBM](https://github.com/NVlabs/VAEBM)

 





## Train EBM

To train EBM on pretrained NVAE, the checkpoint path needs to be specified in **ebms_config.py**.



### 1. LSUN Church 64

```python
# Require two 32G-V100
CUDA_VISIBLE_DEVICES=gpu0, gpu1 python ebm_train.py --backbone=Church64_Gaussian_Decoder
```

### 2. CIFAR 10

```python
# Require two 32G-V100
# For Gaussian decoder
CUDA_VISIBLE_DEVICES=gpu0, gpu1 python ebm_train.py --backbone=CIFAR10_Gaussian_Decoder 
# For Logistic decoder
CUDA_VISIBLE_DEVICES=gpu0, gpu1 python ebm_train.py --backbone=CIFAR10_Logistic_Decoder
```

### 3. CelebA HQ 256

```python
# Require two 32G-V100
# For Gaussian decoder
CUDA_VISIBLE_DEVICES=gpu0, gpu1 python ebm_train.py --backbone=CelebA256_Gaussian_Decoder 
# For Logistic decoder
CUDA_VISIBLE_DEVICES=gpu0, gpu1 python ebm_train.py --backbone=CelebA256_Logistic_Decoder
```



### Compute FID Score

```python
# By default, it uses 4 32G-V100. For using other #GPUs, set --nprocs=#GPUs 
CUDA_VISIBLE_DEVICES=gpu0, gpu1, gpu2, gpu3 python ebm_fid.py --backbone=NVAE_ckpt --ebm_exp=yyyy-mm-dd-xx-xx-xx --ebm_ckpt=xxx.pth
```

