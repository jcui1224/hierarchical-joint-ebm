import torch
from torch.autograd import Variable
import torch.distributed as dist
import os
import datetime
import shutil
import pickle
import numpy as np
import utils
import torch.nn as nn
from collections import OrderedDict

class _netEzi(nn.Module):

    def __init__(self, z_shape, args):
        super().__init__()
        self.z_shape = z_shape
        self.nef = args.nef
        self.ndf = args.ndf
        layers = {8: 1, 16: 2, 32: 3, 64: 4, 128: 5}

        b, c, h, w = z_shape[0], z_shape[1], z_shape[2], z_shape[3]
        num_layers = layers[h]

        convlayers = OrderedDict()
        current_dims = c
        for i in range(num_layers):
            convlayers['conv{}'.format(i+1)] = nn.Conv2d(current_dims, self.nef, 4, 2, 1)
            convlayers['lrelu{}'.format(i+1)] = nn.LeakyReLU(0.2)
            current_dims = self.nef
        self.conv2d = nn.Sequential(convlayers)

        linearlayers = OrderedDict()
        current_dims = self.nef*4*4
        num_layers = num_layers
        for i in range(num_layers): # just try num layers, could be other.
            linearlayers['fc{}'.format(i+1)] = nn.Linear(current_dims, self.ndf)
            linearlayers['lrelu{}'.format(i+1)] = nn.LeakyReLU(0.2)
            current_dims = self.ndf

        linearlayers['out'] = nn.Linear(current_dims, 1)
        self.linear = nn.Sequential(linearlayers)

    def forward(self, z):
        assert z.shape[1:] == self.z_shape[1:]
        f = self.conv2d(z)
        f = f.view(z.shape[0], self.nef*4*4)
        en = self.linear(f)
        return en.squeeze(1)


# ========================= DDP utils =====================
def gather(tensor, nprocs):
    tensor_gather = [torch.zeros_like(tensor) for _ in range(nprocs)]
    dist.all_gather(tensor_gather, tensor)
    return tensor_gather

def stop_condition(tensor):
    return torch.isnan(tensor) or torch.isnan(tensor) or tensor.item() > 1e9 or tensor.item() < -1e9


# ========================= Torch utils ====================
def requires_grad(model_list, flag=True):
    for i, m in enumerate(model_list):
        parameters = m.parameters()
        for p in parameters:
            p.requires_grad = flag

def load_VAE(args, local_rank, logging):
    from model import AutoEncoder
    nvae_checkpoint_dir = f'{args.nvae_dir}/{args.nvae_ckpt}/checkpoint.pt'
    logging.info('loading the model at:')
    logging.info(nvae_checkpoint_dir)

    checkpoint = torch.load(nvae_checkpoint_dir, map_location='cpu')
    VAE_cfg = checkpoint['args']
    if not hasattr(VAE_cfg, 'num_mixture_dec'):
        VAE_cfg.num_mixture_dec = 10

    logging.info('loaded model at epoch %d', checkpoint['epoch'])

    arch_instance = utils.get_arch_cells(VAE_cfg.arch_instance)
    VAE = AutoEncoder(VAE_cfg, None, arch_instance)
    VAE = VAE.cuda(local_rank)

    VAE.load_state_dict(checkpoint['state_dict'], strict=False)
    VAE = torch.nn.parallel.DistributedDataParallel(VAE, device_ids=[local_rank])

    requires_grad([VAE], False)  # wow it can save memory

    if args.bn:
        num_samples = args.batch_size
        t = args.temperature
        iter = 500
        from torch.cuda.amp import autocast
        VAE.train()
        with autocast():
            for i in range(iter):
                if i % 10 == 0 and local_rank == 0:
                    print('Re-set BN statistics iter %d out of %d' % (i + 1, iter))
                VAE.module.sample(num_samples, t)
    VAE.eval()
    logging.info('VAE param size = %fM ', utils.count_parameters_in_M(VAE.module))
    logging.info('groups per scale: %s, total_groups: %d', VAE.module.groups_per_scale, sum(VAE.module.groups_per_scale))
    return VAE

def build_EBM(VAE, args, local_rank, logging):
    with torch.no_grad():  # get a bunch of samples to know how many groups of latent variables are there
        _, z_list, _ = VAE.module.sample(args.batch_size, args.temperature)

    if local_rank == 0:
        for i, z in enumerate(z_list):
            shape = z.shape
            logging.info(f"z {i}, shape: {shape}")

    ebm_list = []
    opt_list = []
    param_sum = 0.
    for i, z in enumerate(z_list):
        shape = z.shape
        netE = _netEzi(shape, args).cuda(local_rank)
        param_sum += utils.count_parameters_in_M(netE)
        netE = torch.nn.parallel.DistributedDataParallel(netE, device_ids=[local_rank])
        utils.average_params(netE.parameters(), is_distributed=True)  # syn params
        opt = torch.optim.Adam(netE.parameters(), lr=args.lr, weight_decay=3e-5, betas=(0.99, 0.999))
        opt_list.append(opt)
        ebm_list.append(netE)
    logging.info('EBM param size = %fM ', param_sum)
    return ebm_list, opt_list

def load_EBM(VAE, args, local_rank, logging):
    with torch.no_grad():  # get a bunch of samples to know how many groups of latent variables are there
        _, z_list, _ = VAE.module.sample(args.batch_size, args.temperature)

    if local_rank == 0:
        for i, z in enumerate(z_list):
            shape = z.shape
            logging.info(f"z {i}, shape: {shape}")

    ebm_list = []
    param_sum = 0.

    ckpt_dir = args.ebm_checkpoint
    ckpt = torch.load(ckpt_dir, map_location='cpu')

    for i, z in enumerate(z_list):
        shape = z.shape
        netE = _netEzi(shape, args).cuda(local_rank)
        netE.load_state_dict(ckpt[f'netE{i}'], strict=True)
        param_sum += utils.count_parameters_in_M(netE)
        netE = torch.nn.parallel.DistributedDataParallel(netE, device_ids=[local_rank])
        ebm_list.append(netE)
    logging.info('EBM param size = %fM ', param_sum)
    requires_grad(ebm_list, False)
    eval_flag(ebm_list)
    return ebm_list

def train_flag(model_list):
    for i, m in enumerate(model_list):
        m.train()

def eval_flag(model_list):
    for i, m in enumerate(model_list):
        m.eval()

def init_z_requires_grad(z_list, local_rank):
    epz_list = [Variable(torch.randn_like(z).cuda(local_rank), requires_grad=True) for z in z_list]
    return epz_list

def save_ckpt(ebm_list, opt_list, save_path, DDP=False):
    state_dict = {}
    for i, (netE, opt) in enumerate(zip(ebm_list, opt_list)):
        state_dict[f'netE{i}'] = netE.module.state_dict() if DDP else netE.state_dict()
        state_dict[f'opt{i}'] = opt.state_dict()
    torch.save(state_dict, save_path)


# ========================= OS utils =======================
def overwrite_opt(opt, opt_override):
    for (k, v) in opt_override.items():
        setattr(opt, k, v)
    return opt

def load_config(args):
    if args.backbone == 'CelebA256_LOGISTIC_Decoder':
        from ebms_config import CelebA256_LOGISTIC_Decoder
        args = overwrite_opt(args, CelebA256_LOGISTIC_Decoder)
    return args

def get_output_dir(args, file, add_datetime=True):
    output_dir = f'./{os.path.splitext(os.path.basename(file))[0]}/{args.nvae_ckpt}'

    if add_datetime:
        t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        output_dir += '/' + t

    return output_dir

def copy_source(args):
    base_dir = './'
    file_list = ['datasets.py', 'distributions.py', 'ebms_config.py', 'ebms_train.py', 'ebms_utils.py',
                 'lmdb_datasets.py', 'model.py', 'neural_ar_operations.py', 'neural_operations.py',
                 'utils.py']
    target_dir = args.save + '/codes'
    os.makedirs(target_dir, exist_ok=True)
    for file in file_list:
        file_dir = base_dir + file
        shutil.copyfile(file_dir, os.path.join(target_dir, os.path.basename(file)))

def save_args(args):
    path = args.save
    args = vars(args)

    with open(path + '/config.txt', 'w') as fp:
        for key in args:
            fp.write(
                ('%s : %s\n' % (key, args[key]))
            )
    with open(path + '/config.pkl', 'wb') as fp:
        pickle.dump(args, fp)

def keep_last_ckpt(args, num=5):
    ckpts = os.listdir(args.save + '/ckpt/')
    ckpts = [int(c.replace(".pth", "")) for c in ckpts]
    if len(ckpts) > num:
        ckpts.sort()
        os.remove(args.save + '/ckpt/' + str(ckpts[0]) + ".pth")

