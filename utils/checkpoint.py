import shutil
import os
import torch
from utils.envs import model_cp_path

# TODO not sure whether working correctly
def save_checkpoint(
    idx, model, optimizer, is_best=False, cp_fname="cp", md_fname="best"
):
    full_cp_fname = "{}_{}_model.pth".format(idx, cp_fname)
    full_cp_optim_fname = "{}_{}_optim.pth".format(idx, cp_fname)
    full_md_fname = "{}_model.pth".format(md_fname)
    full_md_optim_fname = "{}_optim.pth".format(md_fname)
    cp_path = os.path.join(model_cp_path, full_cp_fname)
    cp_optim_path = os.path.join(model_cp_path, full_cp_optim_fname)
    md_path = os.path.join(model_cp_path, full_md_fname)
    md_optim_path = os.path.join(model_cp_path, full_md_optim_fname)
    torch.save(model.state_dict(), cp_path)
    torch.save(optimizer.state_dict(), cp_optim_path)
    if is_best:
        torch.save(model.state_dict(), md_path)
        torch.save(optimizer.state_dict(), md_optim_path)


def load_cp_model(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)


def load_cp_optim(optimizer, checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    optimizer.load_state_dict(state_dict)
