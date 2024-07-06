import os
import sys
import random
import errno
import time
import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from datetime import timedelta


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


def setup_system(seed, cudnn_benchmark=True, cudnn_deterministic=True) -> None:
    '''
    Set seeds for for reproducible training
    '''
    # python
    random.seed(seed)
    
    # numpy
    np.random.seed(seed)
    
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn_benchmark_enabled = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
      
        
def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def sec_to_min(seconds):
    
    seconds = int(seconds)
    minutes = seconds // 60
    seconds_remaining = seconds % 60
    
    if seconds_remaining < 10:
        seconds_remaining = '0{}'.format(seconds_remaining)
    
    return '{}:{}'.format(minutes, seconds_remaining)


def sec_to_time(seconds):
    return "{:0>8}".format(str(timedelta(seconds=int(seconds))))


def print_time_stats(t_train_start, t_epoch_start, epochs_remaining, steps_per_epoch):
    
    elapsed_time = time.time() - t_train_start
    speed_epoch = time.time() - t_epoch_start 
    speed_batch = speed_epoch / steps_per_epoch
    eta = speed_epoch * epochs_remaining
        
    print("Elapsed {}, {} time/epoch, {:.2f} s/batch, remaining {}".format(
                sec_to_time(elapsed_time), sec_to_time(speed_epoch), speed_batch, sec_to_time(eta)))
    

def apply_viridis_colormap(array):
    """Applies the Viridis colormap to a 2D array and returns an RGB array."""
    norm = plt.Normalize(vmin=-80, vmax=0) # -80 and 0
    cmap = cm.get_cmap('viridis')
    rgba_array = cmap(norm(array)).astype(np.float32)
    rgb_array = rgba_array[:, :, :3]
    return rgb_array


class Customizable:
    """Descriptor class for making class methods customizable on a per-instance basis."""
    def __init__(self, method):
        self.method = method
    def __set_name__(self, owner, name):
        self.name = name
    def __get__(self, instance, owner):
        return instance.__dict__.get(self.name, self.method.__get__(instance, owner))