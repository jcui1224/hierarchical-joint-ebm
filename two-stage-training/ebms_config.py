nvae_base_dir = '../nvae/ckpt'
temperature = 1.0

CelebA256_LOGISTIC_Decoder = {
    'dataset': 'celeba_256',
    'data': '../data/celeba256_lmdb',
    'nvae_dir': nvae_base_dir,
    'nvae_ckpt': 'CELEBA256_NVAE_QUALI_DOWNLOADED',
    'batch_size': 2,
    'temperature': temperature,
    'bn': 0,
    'nef': 64,
    'ndf': 200,
    'lr': 2e-5,
    'ld_steps': 10,
    'ld_size': 1e-5,
    'gpu_num': 2,
    'max_iter': 6500,
}