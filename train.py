import os
import random
import numpy as np
import torch
from data import make_data_loader
from loss import make_loss
from model import bulid_MGN_resnet
from processor import Trainer, make_optimizer, make_scheduler
from configs import args

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def train(args):
    
    output_dir = args.output_dir
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_loader, val_loader, class_num, cam_num, view_num = make_data_loader(args)

    if class_num != args.num_classes:
        raise Exception('The number of dataset classes is not equal to the model')

    model = bulid_MGN_resnet(args)

    tri_loss, soft_loss = make_loss(args, class_num)
    
    loss_fn = [tri_loss, soft_loss]

    optimizer = make_optimizer(args, model)
    
    scheduler = make_scheduler(args, optimizer)

    Trainer(args, model, train_loader, val_loader, optimizer, scheduler, loss_fn)

if __name__ == '__main__':
    
    train(args)