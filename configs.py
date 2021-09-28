import argparse

parser = argparse.ArgumentParser(description='MGN')

parser.add_argument("--cuda", type=int, default=3, help='dataset directory')

# Data
parser.add_argument("--root", type=str, default="/home/wangt/mgn_master/dataset", help='dataset directory')
parser.add_argument('--dataset', type=str, default='market1501', help='train dataset name')
parser.add_argument('--resize', type=tuple, default=(384, 128), help='resize')
parser.add_argument('--filp', type=float, default=0.5, help='filp')
parser.add_argument('--random_crop', type=tuple, default=(384, 128), help='')
parser.add_argument("--random_erase", default=True, help='')
parser.add_argument("--erase_prob", type=float, default=0.3, help='')
parser.add_argument("--padding", type=int, default=10, help='')



#Model 

parser.add_argument('--feats', type=int, default=256, help='number of feature maps')
parser.add_argument('--num_classes', type=int, default=751, help='')

# train
parser.add_argument("--epochs", type=int, default=80, help='number of epochs to train')

parser.add_argument("--batch_size", type=int, default=64, help='the batch for id')
parser.add_argument("--num_instance", type=int, default=16, help='the batch of per id')
parser.add_argument("--batchtest", type=int, default=64, help='input batch size for test')
parser.add_argument("--num_worker", type=int, default=8, help='input batch size for test')
parser.add_argument("--label_smooth",type=str, default='on', help='input batch size for test')



# optim

parser.add_argument('--optimizer', default='SGD', help='')
parser.add_argument('--momentum', type=float, default=0.9, help='')
parser.add_argument('--dampening',type=float, default=0, help='')
parser.add_argument('--nesterov', action='store_true', help='')
parser.add_argument('--lr', type=float, default=0.001, help='')
parser.add_argument('--gamma', type=float, default=0.1, help='')
parser.add_argument('--lr_decay', type=int, default=60, help='')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='')
parser.add_argument('--decay_type', type=str, default='step', help='')
parser.add_argument('--no_margin', default= True , help='')
    # if no_margin is False then margin works 
parser.add_argument('--margin',type=float, default=0.3 , help='')


#log
parser.add_argument('--output_dir', default='', help='')
parser.add_argument('--log_period', default=50, help='')
parser.add_argument('--checkpoint_period', type=int, default=40, help='')
parser.add_argument('--eval_period', type=int, default=40, help='')


args = parser.parse_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False