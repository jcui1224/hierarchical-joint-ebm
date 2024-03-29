import torch as t
import torch.nn as nn
import torch.backends.cudnn as cudnn
from nets import _Cifar10_netG as _netG_pxgz1, _netEzi, _netGzi_mlp
from utils import *
from torch.autograd import Variable
import numpy as np
import random
import datetime
import argparse
import torchvision
import torchvision.transforms as transforms

cifar10_config = {
    'dataset': 'cifar10',
    'img_size': 32,
    'normalize_data': True,
}

def get_dataset(args):
    data_dir = args['data_dir']
    img_size = args['img_size']
    if args['dataset'] == 'cifar10':
        if args['normalize_data']:
            transform = transforms.Compose(
                [transforms.Resize(img_size), transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])

        ds_train = torchvision.datasets.CIFAR10(data_dir + 'cifar10', download=True, train=True, transform=transform)
        ds_val = torchvision.datasets.CIFAR10(data_dir + 'cifar10', download=True, train=False, transform=transform)
        input_shape = [3, img_size, img_size]
        return ds_train, ds_val, input_shape

class Normal:
    def __init__(self, mu, log_sigma, device):
        self.device = device
        self.mu = mu
        self.sigma = t.exp(log_sigma) + 1e-2      # we don't need this after soft clamp

    def rsample(self):
        eps = t.Tensor(self.mu.size()).cuda(self.device).normal_()
        return eps * self.sigma + self.mu

    def sample_given_eps(self, eps):
        return eps * self.sigma + self.mu

    def log_prob(self, samples):
        normalized_samples = (samples - self.mu) / self.sigma
        log_p = - 0.5 * normalized_samples * normalized_samples - 0.5 * np.log(2 * np.pi) - t.log(self.sigma)
        return log_p

mse = nn.MSELoss(reduction='none')

def compute_log_px(netG, x, zs, args):
    z1 = zs[0]
    z2 = zs[1]

    x_rec = netG[0](z1)
    recon_loss = t.sum(mse(x_rec, x), dim=[1, 2, 3]).sum()
    log_pxgz = (-1.0 / (2.0 * args['sigma'] ** 2)) * recon_loss

    z1_dist, _ = netG[1](z2)
    log_pz1gz2 = t.sum(z1_dist.log_prob(z1), dim=1).sum()

    log_pz2 = t.sum(Normal(t.zeros_like(z2), t.zeros_like(z2), device=args['device']).log_prob(z2), dim=1).sum()
    log_pz = log_pz1gz2 + log_pz2
    log_px = log_pxgz + log_pz
    return recon_loss, log_pz, log_px

def compute_log_pz(netG, zs, args):
    z1 = zs[0]
    z2 = zs[1]

    z1_dist, _ = netG[1](z2)
    log_pz1gz2 = t.sum(z1_dist.log_prob(z1), dim=1).sum()

    log_pz2 = t.sum(Normal(t.zeros_like(z2), t.zeros_like(z2), device=args['device']).log_prob(z2), dim=1).sum()

    log_pz = log_pz1gz2 + log_pz2
    return log_pz

def compute_energy(netE, zs):
    z1 = zs[0]
    z2 = zs[1]

    enz1 = netE[0](z1).sum()
    enz2 = netE[1](z2).sum()
    en = enz1 + enz2

    return en, enz1, enz2

def langevin_posterior_sampling(zs, x, netG, netE, args, logger, display=False):
    g_l_steps = args['g_l_steps']
    g_l_step_size = args['g_l_step_size']
    # langevin_coff = 2 / (args['e_l_step_size'] ** 2)
    batch_size = zs[0].shape[0]

    for j in range(g_l_steps):
        recon_loss, log_pz, g_loss = compute_log_px(netG, x, zs, args)
        en, enz1, enz2 = compute_energy(netE, zs)
        # en = langevin_coff * en
        pz_grad = t.autograd.grad(g_loss, zs)
        en_grad = t.autograd.grad(en, zs)

        zs[0].data += 0.5 * g_l_step_size * g_l_step_size * (en_grad[0] + pz_grad[0])
        zs[1].data += 0.5 * g_l_step_size * g_l_step_size * (en_grad[1] + pz_grad[1])

        if args['use_noise']:
            zs[0].data += g_l_step_size * t.randn_like(zs[0])
            zs[1].data += g_l_step_size * t.randn_like(zs[1])

        if display and ((j + 1) % (g_l_steps / 5) == 0 or j == 0):
            log_info = f'Posterior Langevin {j}/{g_l_steps} recon_loss: {recon_loss:.3f} log_pz: {log_pz:.3f} enz1 {enz1:.3f} enz2 {enz2:.3f}'
            logger.info(log_info)

    return [z.detach() for z in zs]

def langevin_prior_sampling(zs, netG, netE, args, logger, display=False):
    e_l_steps = args['e_l_steps']
    e_l_step_size = args['e_l_step_size']

    batch_size = zs[0].shape[0]

    for j in range(e_l_steps):
        g_loss = compute_log_pz(netG, zs, args)
        en, enz1, enz2 = compute_energy(netE, zs)

        pz_grad = t.autograd.grad(g_loss, zs)
        en_grad = t.autograd.grad(en, zs)

        zs[0].data += 0.5 * e_l_step_size * e_l_step_size * (en_grad[0] + pz_grad[0])
        zs[1].data += 0.5 * e_l_step_size * e_l_step_size * (en_grad[1] + pz_grad[1])

        if args['use_noise']:
            zs[0].data += e_l_step_size * t.randn_like(zs[0])
            zs[1].data += e_l_step_size * t.randn_like(zs[1])

        if display and ((j + 1) % (e_l_steps / 5) == 0 or j == 0):
            log_info = f'Prior Langevin {j}/{e_l_steps} log_pz: {g_loss:.3f} enz1 {enz1:.3f} enz2 {enz2:.3f}'
            logger.info(log_info)

    return [z.detach() for z in zs]

def sample_p_0(batch_size, args, requires_grad=True):
    z1 = Variable(t.randn((batch_size, args['z1_dim'])).to(args['device']), requires_grad=requires_grad)
    z2 = Variable(t.randn((batch_size, args['z2_dim'])).to(args['device']), requires_grad=requires_grad)
    zs = [z1, z2]
    return zs

def sample_x(batch_size, netG, netE, args, logger=None, display=False):
    z_e_0 = sample_p_0(batch_size, args)
    z_e_k = langevin_prior_sampling(z_e_0, netG, netE, args, logger, display=display)
    x_mu = netG[0](z_e_k[0])
    return x_mu

def recon_x(x, netG, netE, args, logger=None, display=False):
    z_g_0 = sample_p_0(x.shape[0], args)
    z_g_k = langevin_posterior_sampling(z_g_0, x, netG, netE, args, logger, display=display)
    x_mu = netG[0](z_g_k[0])
    return x_mu

def train_step(netG, netE, optG, optE, dl_train, logger, args, **kwargs):

    def eval_flag():
        netG[0].eval()
        netG[1].eval()
        netE[0].eval()
        netE[1].eval()
        requires_grad(netG, False)
        requires_grad(netE, False)

    def train_flag():
        netG[0].train()
        netG[1].train()
        netE[0].train()
        netE[1].train()
        requires_grad(netG, True)
        requires_grad(netE, True)

    Broken = False
    # log_iter = int(len(dl_train) // 4) if len(dl_train) > 1000 else int(len(dl_train) // 2)
    log_iter = 1

    for i, x in enumerate(dl_train, 0):
        if i % log_iter == 0:
            logger.info("=="*10 + f"ep: {kwargs['ep']} batch: [{i}/{len(dl_train)}]" + "=="*10)

        train_flag()

        x = x[0].to(args['device']) if type(x) is list else x.to(args['device'])
        batch_size = x.shape[0]

        z_g_0 = sample_p_0(batch_size, args)
        z_g_k = langevin_posterior_sampling(z_g_0, x, netG, netE, args, logger, display=(i % log_iter == 0))
        z_e_0 = sample_p_0(batch_size, args)
        z_e_k = langevin_prior_sampling(z_e_0, netG, netE, args, logger, display=(i % log_iter == 0))

        optG[0].zero_grad()
        pxgz1_loss = netG[0].loss(x, z_g_k[0])
        loss = (1.0 / (2.0 * args['sigma'] ** 2)) * pxgz1_loss
        loss.backward()
        optG[0].step()

        optG[1].zero_grad()
        pz1gz2_loss = netG[1].loss(z_g_k[0:2], z_e_k[0:2])
        pz1gz2_loss.backward()
        optG[1].step()

        """
        EBM Update
        """
        optE[0].zero_grad()
        e1_t = netE[0](z_g_k[0]).mean()
        e1_f = netE[0](z_e_k[0]).mean()
        # e1_loss = (e1_f - e1_t)*langevin_coff
        e1_loss = (e1_f - e1_t)
        e1_loss.backward()
        optE[0].step()

        optE[1].zero_grad()
        e2_t = netE[1](z_g_k[1]).mean()
        e2_f = netE[1](z_e_k[1]).mean()
        # e2_loss = (e2_f - e2_t) * langevin_coff
        e2_loss = (e2_f - e2_t)
        e2_loss.backward()
        optE[1].step()

        if i % log_iter == 0:
            log_info = f'|| e1_t: {e1_t:.3f} e1_f: {e1_f:.3f} || e2_t: {e2_t:.3f} e2_f: {e2_f:.3f} ||'
            logger.info(log_info)
            log_info = f'|| pxgz1: {pxgz1_loss:.3f} || pz1gz2: {pz1gz2_loss:.3f} ||'
            logger.info(log_info)

        if t.isnan(e1_t.mean()) or t.isnan(e1_f.mean()):
            logger.info("Got NaN at ep {} iter {}".format(kwargs['ep'], i))
            log_info = f'|| e1_t: {e1_t:.3f} e1_f: {e1_f:.3f} || e2_t: {e2_t:.3f} e2_f: {e2_f:.3f} ||'
            logger.info(log_info)
            log_info = f'|| pxgz1: {pxgz1_loss:.3f} || pz1gz2: {pz1gz2_loss:.3f} ||'
            logger.info(log_info)
            Broken = True
            return Broken

    eval_flag()
    return Broken


def fit(netG, netE, dl_train, test_batch, args, logger):

    optE = [t.optim.Adam(netE[0].parameters(), lr=args['e1_lr']),
            t.optim.Adam(netE[1].parameters(), lr=args['e2_lr'])]

    optG = [t.optim.Adam(netG[0].parameters(), lr=args['g1_lr']),
            t.optim.Adam(netG[1].parameters(), lr=args['g2_lr'])]

    to_range_0_1 = lambda x: (x + 1.) / 2. if args['normalize_data'] else x

    for ep in range(args['epochs']):

        Broken = train_step(netG, netE, optG, optE, dl_train, logger, args, ep=ep)

        if Broken:
            return

        if ep % args['vis_iter'] == 0:
            imgs_dir = args['dir'] + 'imgs/'

            rec_mu = recon_x(test_batch, netG, netE, args, logger=logger, display=False)
            show_single_batch(rec_mu, imgs_dir + f'{ep:>07d}_rec.png', nrow=10)

            syn_mu = sample_x(args['batch_size'], netG, netE, args, logger=logger, display=False)
            show_single_batch(syn_mu, imgs_dir + f'{ep:>07d}_syn.png', nrow=10)

        if args['fid']:
            if ep >= args['n_metrics_start'] and ep % args['n_metrics'] == 0:
                from pytorch_fid_jcui7.fid_score import compute_fid
                from tqdm import tqdm
                try:
                    s = []
                    for _ in tqdm(range(int(50000 / args['batch_size']))):
                        syn = sample_x(args['batch_size'], netG, netE, args, logger=logger, display=False)
                        syn = to_range_0_1(syn).clamp(min=0., max=1.)
                        s.append(syn)
                    s = t.cat(s)
                    fid = compute_fid(x_train=None, x_samples=s, path=args['fid_stat_dir'])
                    logger.info('fid: {:.5f}'.format(fid))

                except Exception as e:
                    print(e)
                    logger.critical(e, exc_info=True)
                    logger.info('FID failed')
    return

def letgo(args_job, output_dir):
    set_seeds(1224)
    args = parse_args()
    args = overwrite_opt(args, args_job)
    args = vars(args)
    args = overwrite_dict(args, cifar10_config)
    output_dir += '/'
    args['dir'] = output_dir

    [os.makedirs(args['dir'] + f'{f}/', exist_ok=True) for f in ['ckpt', 'imgs']]

    logger = Logger(args['dir'], f"job{args['job_id']}")
    logger.info('Config')
    logger.info(args)

    save_args(args, output_dir)

    ds_train, _, input_shape = get_dataset(args)
    dl_train = t.utils.data.DataLoader(ds_train, batch_size=args['batch_size'], shuffle=True, num_workers=0, drop_last=True)
    args['input_shape'] = input_shape
    logger.info("Training samples %d" % len(ds_train))

    fix_x = next(iter(dl_train))
    test_batch = fix_x[0].to(args['device']) if type(fix_x) is list else fix_x.to(args['device'])
    show_single_batch(test_batch, args['dir'] + 'imgs/test_batch.png', nrow=int(args['batch_size'] ** 0.5))

    pxgz1 = _netG_pxgz1(z1_dim=args['z1_dim'], sigma=args['sigma']).to(args['device'])
    pz1gz2 = _netGzi_mlp(input_z_dim=args['z2_dim'], to_z_dim=args['z1_dim'], ndf=args['pz1gz2_ndf'], num_layers=args['pz1gz2_layers'], N=Normal).to(args['device'])
    netG = [pxgz1, pz1gz2]

    print(f"netG has {count_parameters_in_M(pxgz1)+count_parameters_in_M(pz1gz2)}M")

    enz1 = _netEzi(z_dim=args['z1_dim'], nef=args['enz1_nef'], num_layers=args['enz1_layers']).to(args['device'])
    enz2 = _netEzi(z_dim=args['z2_dim'], nef=args['enz2_nef'], num_layers=args['enz2_layers']).to(args['device'])
    netE = [enz1, enz2]

    print(f"netE has {count_parameters_in_M(enz1)+count_parameters_in_M(enz2)}M")

    fit(netG, netE, dl_train, test_batch, args, logger)
    return

def parse_args():
    parser = argparse.ArgumentParser()
    # General arguments

    parser.add_argument('--epochs', type=int, default=101)
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--batch_size', type=int, default=100)

    parser.add_argument('--e1_lr', type=float, default=2e-5)
    parser.add_argument('--e2_lr', type=float, default=2e-5)
    parser.add_argument('--g1_lr', type=float, default=1e-4)
    parser.add_argument('--g2_lr', type=float, default=1e-4)

    parser.add_argument('--z1_dim', type=int, default=128)
    parser.add_argument('--z2_dim', type=int, default=100)

    parser.add_argument('--sigma', type=float, default=0.3)

    parser.add_argument('--use_noise', type=bool, default=True)
    parser.add_argument('--e_l_steps', type=int, default=60)
    parser.add_argument('--e_l_step_size', type=float, default=0.4)
    parser.add_argument('--g_l_steps', type=int, default=40)
    parser.add_argument('--g_l_step_size', type=float, default=0.1)

    parser.add_argument('--enz1_nef', type=int, default=200)
    parser.add_argument('--enz1_layers', type=int, default=3)
    parser.add_argument('--enz2_nef', type=int, default=100)
    parser.add_argument('--enz2_layers', type=int, default=2)

    parser.add_argument('--pxgz1_ngf', type=int, default=64)
    parser.add_argument('--pz1gz2_ndf', type=int, default=200)
    parser.add_argument('--pz1gz2_layers', type=int, default=4)

    parser.add_argument('--fid', type=bool, default=False)
    parser.add_argument('--fid_stat_dir', type=str, default='../fid_stat/fid_stats_cifar10_train.npz')
    parser.add_argument('--n_metrics', type=int, default=5) #10
    parser.add_argument('--n_metrics_start', type=int, default=10)

    parser.add_argument('--vis_iter', type=int, default=1)

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
    os.makedirs(output_dir + 'code/', exist_ok=True)

    [save_file(output_dir, f) for f in ['nets.py', 'utils.py', os.path.basename(__file__)]]

    opt = dict()
    letgo(opt, output_dir)

if __name__ == '__main__':
    main()
