import os
import torch
import argparse
from configs import args
from data import make_data_loader
from model import bulid_MGN_resnet
from processor import inference

def test(args):

    print('Start Testing ------')
    train_loader, val_loader, class_num, _, _, num_query = make_data_loader(args)

    device = torch.device(args.cuda)

    model = bulid_MGN_resnet(args)

    model.to(device)

    model.load_param(args.test_weight)


    inference(args, model, val_loader, num_query, device)


if __name__ == '__main__':
    
    test(args)