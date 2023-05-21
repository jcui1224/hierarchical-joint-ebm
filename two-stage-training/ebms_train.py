import math
import random
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from torchvision import utils as vutils
import datasets
from ebms_utils import *

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

def train(VAE, ebm_list, opt_list, train_queue, local_rank, nprocs, logging, args):

    with torch.no_grad():  # get a bunch of samples to know how many groups of latent variables are there
        _, z_list, _ = VAE.module.sample(args.batch_size, args.temperature)

    noise_list = [torch.randn(zi.size()).cuda(local_rank) for zi in z_list]

    global_step = 0

    while True:

        train_queue.sampler.set_epoch(global_step)

        for b, x in enumerate(train_queue):

            if global_step > args.max_iter:
                logging.info("+++++ MAX ITER ++++++")
                return

            global_step += 1
            x = x[0] if len(x) > 1 else x
            x = x.cuda(local_rank)
            train_flag(ebm_list)

            if b % 100 == 0:  # could be useless
                for ebm in ebm_list: utils.average_params(ebm.parameters(), True)

            if b % args.log_iter == 0:
                logging.info("===" * 15 + f'Iteration {global_step:>06d} Batch {b}/{len(train_queue)}' + "===" * 15)

            posterior = VAE.module.get_posterior_samples(x)

            epz_list = init_z_requires_grad(z_list, local_rank)
            requires_grad(ebm_list, False)
            logits, neg_datas = langevin_prior_samples(VAE, ebm_list, epz_list, args, noise_list, logging, display=(b % args.log_iter == 0))
            requires_grad(ebm_list, True)

            train_log = '||'
            for i, (z_t, z_f, netE, opt) in enumerate(zip(posterior, neg_datas, ebm_list, opt_list)):
                layer_idx = len(posterior) - i
                netE.zero_grad()
                ef = netE(z_f.detach()).mean()
                et = netE(z_t.detach()).mean()
                loss = (ef - et)

                loss.backward()

                dist.barrier()  # wait for syn gradient

                ef_gather = [stop_condition(e) for e in gather(ef, nprocs=nprocs)]
                et_gather = [stop_condition(e) for e in gather(et, nprocs=nprocs)]

                if True in ef_gather or True in et_gather:
                    print(f"local_rank: {local_rank} || z{layer_idx} ef: {ef}, et: {et}")
                    return

                utils.average_gradients(netE.parameters(), args.distributed)

                opt.step()

                train_log += f' et_z{layer_idx}: {et.item():15.2f} ef_z{layer_idx}: {ef.item():15.2f} ||'

            if local_rank == 0 and b % args.log_iter == 0:
                logging.info(train_log)

                os.makedirs(args.save + f'/images/', exist_ok=True)
                syn = VAE.module.decoder_output(logits).sample()
                vutils.save_image(syn, args.save + f'/images/{global_step:>07d}.png', nrow=int(math.floor(math.sqrt(args.batch_size))), normalize=True)

            if local_rank == 0 and b % args.log_iter == 0:
                os.makedirs(args.save + f'/ckpt/', exist_ok=True)
                save_ckpt(ebm_list, opt_list, args.save + f'/ckpt/{global_step}.pth', DDP=args.distributed)
                keep_last_ckpt(args, num=20)

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
    init_seeds(1)
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=nprocs, rank=local_rank)

    logging = utils.Logger(local_rank, args.save)
    logging.info(args)

    VAE = load_VAE(args, local_rank, logging)
    ebm_list, opt_list = build_EBM(VAE, args, local_rank, logging)
    train_queue, _, num_classes = datasets.get_loaders(args)

    train(VAE, ebm_list, opt_list, train_queue, local_rank, nprocs, logging, args)
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint EBM Prior Model on NVAE Backbone PyTorch Training')
    parser.add_argument("--backbone", type=str, default='CelebA256_LOGISTIC_Decoder')
    parser.add_argument("--log_iter", type=int, default=10)

    # ================= DDP ===================
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--master_address', type=str, default='127.0.0.1', help='address for master')
    parser.add_argument('--master_port', type=str, default='6020', help='port for master')

    args = parser.parse_args()
    args.distributed = True

    args = load_config(args)
    args.nprocs = args.gpu_num

    args.save = get_output_dir(args, __file__, add_datetime=True)
    os.makedirs(args.save, exist_ok=True)

    copy_source(args)

    save_args(args)

    mp.spawn(main, nprocs=args.nprocs, args=(args.nprocs, args))