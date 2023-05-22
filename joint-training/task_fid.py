import torch as t
import torch.nn as nn
import torch.backends.cudnn as cudnn
from nets import _Cifar10_netG as _netG_pxgz1, _netEzi, _netGzi_mlp
from utils import *
import numpy as np
import random
import datetime
import argparse
from train_cifar10 import sample_x, Normal

def letgo(args_job, output_dir):
    set_seeds(1224)
    args = parse_args()
    args = overwrite_opt(args, args_job)
    args = vars(args)
    output_dir += '/'
    args['dir'] = output_dir
    EBM_args = load_args(args['model_dir'])
    ckpt_dir = args['model_dir'] + f"ckpt/{args['model_ckpt']}"
    ckpt = t.load(ckpt_dir, map_location='cpu')

    [os.makedirs(args['dir'] + f'{f}/', exist_ok=True) for f in ['imgs']]

    pxgz1 = _netG_pxgz1(z1_dim=EBM_args['z1_dim'], sigma=EBM_args['sigma']).to(args['device'])
    pxgz1.load_state_dict(ckpt['netGz1'], strict=True)
    pz1gz2 = _netGzi_mlp(input_z_dim=EBM_args['z2_dim'], to_z_dim=EBM_args['z1_dim'], ndf=EBM_args['pz1gz2_ndf'], num_layers=EBM_args['pz1gz2_layers'], N=Normal).to(args['device'])
    pz1gz2.load_state_dict(ckpt['netGz2'], strict=True)
    netG = [pxgz1, pz1gz2]

    print(f"netG has {count_parameters_in_M(pxgz1)+count_parameters_in_M(pz1gz2)}M")

    enz1 = _netEzi(z_dim=EBM_args['z1_dim'], nef=EBM_args['enz1_nef'], num_layers=EBM_args['enz1_layers']).to(args['device'])
    enz1.load_state_dict(ckpt['netEz1'], strict=True)
    enz2 = _netEzi(z_dim=EBM_args['z2_dim'], nef=EBM_args['enz2_nef'], num_layers=EBM_args['enz2_layers']).to(args['device'])
    enz2.load_state_dict(ckpt['netEz2'], strict=True)
    netE = [enz1, enz2]

    print(f"netE has {count_parameters_in_M(enz1)+count_parameters_in_M(enz2)}M")

    netG[0].eval()
    netG[1].eval()
    netE[0].eval()
    netE[1].eval()
    requires_grad(netG, False)
    requires_grad(netE, False)

    to_range_0_1 = lambda x: (x + 1.) / 2. if EBM_args['normalize_data'] else x
    from pytorch_fid_jcui7.fid_score import compute_fid
    from tqdm import tqdm
    try:
        s = []
        for _ in tqdm(range(int(50000 / EBM_args['batch_size']))):
            syn = sample_x(EBM_args['batch_size'], netG, netE, EBM_args, logger=None, display=False)
            syn = to_range_0_1(syn).clamp(min=0., max=1.)
            s.append(syn)
        s = t.cat(s)
        fid = compute_fid(x_train=None, x_samples=s, path=args['fid_stat_dir'])
        print('fid: {:.5f}'.format(fid))

    except Exception as e:
        print(e)
    return

def parse_args():
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument('--model_dir', type=str, default="")
    parser.add_argument('--model_ckpt', type=str, default="")
    parser.add_argument('--fid_stat_dir', type=str, default='fid_stats_cifar10_train.npz')
    parser.add_argument('--job_id', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    # Parser
    args, unknown = parser.parse_known_args()
    if len(unknown) > 0:
        print("Invalid arguments %s" % unknown)
        parser.print_help()
        sys.exit()
    return args

def set_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def main():
    output_dir = './{}/'.format(os.path.splitext(os.path.basename(__file__))[0])
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    output_dir += t + '/'
    os.makedirs(output_dir, exist_ok=True)

    opt = dict()
    letgo(opt, output_dir)

if __name__ == '__main__':
    main()
