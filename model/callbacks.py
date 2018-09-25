import torch
import torch.nn as nn
import torch.nn.functional as F

class CallBacks():
    def __init__(self, n_epoch, n_batch_iter):
        self.n_epoch = n_epoch
        self.n_batch_iter = n_batch_iter

    def on_train_begin(self): pass

    def on_train_end(self): raise NotImplementedError

    def on_epoch_begin(self):pa

    def on_epoch_end(self):
        raise NotImplementedErrorss

    def on_batch_begin(self):
        raise NotImplementedError

    def on_batch_end(self):
        raise NotImplementedError

class CLR(CallBacks):
    def __init__(self, n_epoch, n_batch_iter, max):
