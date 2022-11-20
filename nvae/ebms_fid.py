import sys
sys.path.append("..")
from pytorch_fid_jcui7.fid_score_imgs import compute_fid as get_fid
import math
import random
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from torchvision import utils as vutils
import datasets
from ebms_utils import *
from tqdm import tqdm
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,4,5"
fid_size = {'celeba_256': 30000, 'cifar10': 50000, 'lsun_church_64': 50000,}
fid_stats_dir = '/data4/jcui7/fid_stats/'
fid_dir = {
    'celeba_256': fid_stats_dir + 'celeba256/fid_stats_celeba256_train.npz',
    'cifar10': fid_stats_dir + 'cifar10/fid_stats_cifar10_train.npz',
    'lsun_church_64': fid_stats_dir + 'lsun/fid_stats_church64_train.npz',
}

def langevin_prior_samples(VAE, ebm_list, epz_list, args, noise_list, logging, display):
    sample_steps = args.ld_steps
    step_size = args.ld_size
    batch_size = args.batch_size
    t = args.temperature

    for k in range(sample_steps):
        langevin_log = f'======================== Langevin {k}/{sample_steps}:'
        _, z_list, log_p_total = VAE.module.sample(batch_size, t, epz_list)
        pzgz_loss = log_p_total.mean()

        en_loss = 0.
        en_log = '||'
        for i, (z, netE) in enumerate(zip(z_list, ebm_list)):
            en = netE(z).mean()  # avg on batch
            en_loss += en
            en_log += f' en_z{len(ebm_list) - i}: {en.item():15.2f} ||'

        loss = en_loss + pzgz_loss
        langevin_log += f' en_loss: {en_loss.item():15.2f}  pzgz: {pzgz_loss.item():15.2f} total: {loss.item():15.2f} || ===================='
        loss.backward()

        for i in range(len(epz_list)):
            epz_list[i].data.add_(0.5 * step_size, epz_list[i].grad.data * batch_size)
            noise_list[i].normal_(0, 1)
            epz_list[i].data.add_(np.sqrt(step_size), noise_list[i].data)
            epz_list[i].grad.detach_()
            epz_list[i].grad.zero_()

        if k % (sample_steps // 10) == 0 and display:
            logging.info(langevin_log + '\n' + en_log)

    eps_z = [epz.detach() for epz in epz_list]
    logits, neg_datas, _ = VAE.module.sample(batch_size, t, eps_z)
    return logits, neg_datas

def compute_fid(VAE, ebm_list, local_rank, nprocs, logging, args):

    with torch.no_grad():  # get a bunch of samples to know how many groups of latent variables are there
        _, z_list, _ = VAE.module.sample(args.batch_size, args.temperature)

    noise_list = [torch.randn(zi.size()).cuda(local_rank) for zi in z_list]

    size = fid_size[args.dataset]
    fid_stats = fid_dir[args.dataset]
    n_batch = int((size // args.batch_size) // nprocs)  # make sure it can divide

    if args.local_rank == 0:  # print rank 0 progress
        count = tqdm(range(n_batch))
    else:
        count = range(n_batch)

    global_idx = 0
    img_dir = args.save + f'/fid_imgs/'
    os.makedirs(img_dir, exist_ok=True)

    for _ in count:
        epz_list = init_z_requires_grad(z_list, args.local_rank)
        logits, neg_datas = langevin_prior_samples(VAE, ebm_list, epz_list, args, noise_list, logging, display=False)
        sample = VAE.module.decoder_output(logits).sample(args.temperature)

        for img in sample:
            vutils.save_image(img, img_dir + f'rank_{args.local_rank}_{global_idx}.png', nrow=1, normalize=True)
            global_idx += 1

    dist.barrier()
    if args.local_rank == 0:
        fid = get_fid(x_train=None, x_samples=img_dir, path=fid_stats, device='cuda:0')
        return fid
    else:
        return math.inf

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def main(local_rank, nprocs, args):
    # ==============================Prepare DDP ================================
    args.local_rank = local_rank
    init_seeds(local_rank)
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=nprocs, rank=local_rank)

    logging = utils.Logger(local_rank, args.save)
    logging.info(args)

    VAE = load_VAE(args, local_rank, logging)
    ebm_list = load_EBM(VAE, args, local_rank, logging)

    fid = compute_fid(VAE, ebm_list, local_rank, nprocs, logging, args)

    logging.info(f"FID : {fid}")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint EBM Prior Model on NVAE Backbone PyTorch Training')
    parser.add_argument("--backbone", type=str, default='Church64_Gaussian_Decoder')
    parser.add_argument("--ebm_exp", type=str, default='2022-11-01-00-21-05')
    parser.add_argument("--ebm_ckpt", type=str, default='15424.pth')
    # ================= DDP ===================
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--master_address', type=str, default='127.0.0.1', help='address for master')
    parser.add_argument('--master_port', type=str, default='6020', help='port for master')
    parser.add_argument('--nprocs', type=int, default=4, help='port for master')

    args = parser.parse_args()
    args.distributed = True

    args = load_test_config(args)

    args.save = os.path.join(get_output_dir(args, __file__, add_datetime=False), f'{args.ebm_exp}')

    os.makedirs(args.save, exist_ok=True)

    copy_source(args)

    save_args(args)

    mp.spawn(main, nprocs=args.nprocs, args=(args.nprocs, args))