from itertools import tee
import math


def to_list(x):
    return x if type(x) == list else [x]


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def get_batch_info(dataloader):
    n_obs = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    n_batch_per_epoch = math.ceil(n_obs / float(batch_size))
    return n_obs, batch_size, n_batch_per_epoch
