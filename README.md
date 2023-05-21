# Learning Joint Latent Space EBM Prior for Multi-layer Generator
## NVAE Checkpoint

NVAE with discrete mixture logistic decoder can be downloaded [here](https://github.com/NVlabs/NVAE); NVAE with Gaussian decoder can be trained by using the [original code](https://github.com/NVlabs/NVAE) with hyper-parameters provided by [VAEBM](https://github.com/NVlabs/VAEBM)

## Train EBM

To train EBM on pretrained NVAE, the checkpoint path needs to be specified in **ebms_config.py**.

### CelebA HQ 256

```python
# Require two 32G-V100
CUDA_VISIBLE_DEVICES=gpu0, gpu1 python ebm_train.py
```
