# Learning Joint Latent Space EBM Prior for Multi-layer Generator

## Joint Training
```python
CUDA_VISIBLE_DEVICES=gpu0 python train_cifar10.py
```
### Compute FID
```python
CUDA_VISIBLE_DEVICES=gpu0 python task_fid.py --model-dir={your_path}/yyyy-mm-dd-hh-mm-ss/ --model_ckpt=xxx.pth
```
Checkpoint is relaseed. 

## Two-stage Training
```python
# Require two 32G-V100
CUDA_VISIBLE_DEVICES=gpu0, gpu1 python ebm_train.py
```

### NVAE Checkpoint
To train EBM on pretrained NVAE, the checkpoint path needs to be specified in **ebms_config.py**.

NVAE with discrete mixture logistic decoder can be downloaded [here](https://github.com/NVlabs/NVAE); NVAE with Gaussian decoder can be trained by using the [original code](https://github.com/NVlabs/NVAE) with hyper-parameters provided by [VAEBM](https://github.com/NVlabs/VAEBM)
