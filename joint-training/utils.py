import os
import logging
import sys
import pickle
import time
import datetime
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

def overwrite_opt(opt, opt_override):
    for (k, v) in opt_override.items():
        setattr(opt, k, v)
    return opt

def overwrite_dict(opt, opt_override):
    for (k, v) in opt_override.items():
        opt[k] = v
    return opt

class Logger(object):
    def __init__(self, save, rank):
        # other libraries may set logging before arriving at this line.
        # by reloading logging, we can get rid of previous configs set by other libraries.
        from importlib import reload
        reload(logging)
        self.rank = rank

        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(save, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
        self.start_time = time.time()

    def info(self, string, *args):

        elapsed_time = time.time() - self.start_time
        elapsed_time = time.strftime(
            '(Elapsed: %H:%M:%S) ', time.gmtime(elapsed_time))
        if isinstance(string, str):
            string = elapsed_time + f" job_id: {self.rank} " +string

        else:
            logging.info(elapsed_time)
        #
        logging.info(string, *args)


def save_file(output_dir, file_name):
    file_in = open('./' + file_name, 'r')
    file_out = open(output_dir + 'code/' + os.path.basename(file_name), 'w')
    for line in file_in:
        file_out.write(line)

def save_args(args, output_dir):
    with open(output_dir + 'config.txt', 'w') as fp:
        for key in args:
            fp.write(
                    ('%s : %s\n' % (key, args[key]))
            )

    with open(output_dir + 'config.pkl', 'wb') as fp:
        pickle.dump(args, fp)

def load_args(path):
    with open(path + 'config.pkl', 'rb') as fp:
        config = pickle.load(fp)
    return config

def get_output_dir(file, add_datetime=True, added_directory=None):
    output_dir = './{}/'.format(os.path.splitext(os.path.basename(file))[0])

    if added_directory is not None:
        output_dir += added_directory + '/'

    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if add_datetime:
        output_dir += t + '/'

    return output_dir