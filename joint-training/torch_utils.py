from torchvision import utils as vutils
import numpy as np
def requires_grad(net_list, flag=True):
    for net in net_list:
        params = net.parameters()
        for p in params:
            p.requires_grad = flag

def show_single_batch(x, path, nrow):
    vutils.save_image(x, path, normalize=True, nrow=nrow)

def count_parameters_in_M(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters()) / 1e6