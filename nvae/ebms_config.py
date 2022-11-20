
nvae_base_dir = '/data6/jcui7/nvae/ckpt/eval-.'
temperature = 1.0
CIFAR10_Gaussian_Decoder = {
    'dataset': 'cifar10',
    'data': '/data4/jcui7/images/data/cifar10',
    'nvae_dir': nvae_base_dir,
    'nvae_ckpt': 'CIFAR10_NVAE',
    'batch_size': 32,
    'temperature': temperature,
    'bn': 0,
    'nef': 64,
    'ndf': 200,
    'lr': 1e-4,
    'ld_steps': 20,
    'ld_size': 4e-5,
    'gpu_num': 2,
    'max_iter': 5000,
}

CIFAR10_Gaussian_Decoder_TEST = {
    'dataset': 'cifar10',
    'data': '/data4/jcui7/images/data/cifar10',
    'nvae_dir': nvae_base_dir,
    'nvae_ckpt': 'CIFAR10_NVAE',
    'batch_size': 32,
    'temperature': temperature,
    'bn': 0,
    'nef': 64,
    'ndf': 200,
    'ld_steps': 40,
    'ld_size': 4e-5,
}

Church64_Gaussian_Decoder = {
    'dataset': 'lsun_church_64',
    'data': '/data4/jcui7/HugeData/lsun/',
    'nvae_dir': nvae_base_dir,
    'nvae_ckpt': 'CHURCH_VAEBM_RUN1',
    'batch_size': 32,
    'temperature': temperature,
    'bn': 0,
    'nef': 64,
    'ndf': 200,
    'lr': 1e-4,
    'ld_steps': 30,
    'ld_size': 4e-5,
    'gpu_num': 2,
    'max_iter': 16000,
}

Church64_Gaussian_Decoder_TEST = {
    'dataset': 'lsun_church_64',
    'data': '/data4/jcui7/HugeData/lsun/',
    'nvae_dir': nvae_base_dir,
    'nvae_ckpt': 'CHURCH_VAEBM_RUN1',
    'batch_size': 32,
    'temperature': temperature,
    'bn': 0,
    'nef': 64,
    'ndf': 200,
    'ld_steps': 40,
    'ld_size': 4e-5,
}

CelebA256_LOGISTIC_Decoder = {
    'dataset': 'celeba_256',
    'data': '/data4/jcui7/HugeData/celeba256_org/celeba256_lmdb',
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