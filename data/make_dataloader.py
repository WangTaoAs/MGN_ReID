'''
@author: Tao Wang
@email: taowang@stu.pku.edu.cn
'''
import torch
from PIL import Image
import os.path as osp
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T
from torch.utils.data import DataLoader, dataloader, sampler
from timm.data.random_erasing import RandomErasing
from torchvision.transforms import transforms

from .market1501 import Market1501
from .occ_duke import OCC_DukeMTMCreID
from .sampler import RandomIdentitySampler


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, trackid,img_path.split('/')[-1]


def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,

def make_data_loader(args):
    # train transform
    transform_list_train = [
        T.Resize(args.resize, interpolation=3),
        T.RandomHorizontalFlip(args.filp),
        T.Pad(args.padding),
        T.RandomCrop(args.random_crop),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # COCO pic mean and std?
    ]

    if args.random_erase:
        transform_list_train.append(RandomErasing(probability=args.erase_prob, mode='pixel', max_count=1, device='cpu'))

    train_list = T.Compose(transform_list_train)

    # val transform
    transform_list_val = [
        T.Resize(args.resize, interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ]
    val_list = T.Compose(transform_list_val)

    datasets = args.dataset
    if datasets == 'market1501':
        dst = Market1501(args.root)
    elif datasets == 'occ_duke':
        dst = OCC_DukeMTMCreID(args.root)
    else:
        print('Do not support this dataset......')
    
    train_set = ImageDataset(dst.train, train_list)
    class_num = dst.num_train_pids
    cam_num = dst.num_train_cams
    view_num = dst.num_train_vids

    train_loader = DataLoader(train_set, 
                              batch_size=args.batch_size, 
                              sampler=RandomIdentitySampler(dst.train, args.batch_size, args.num_instance),
                              shuffle=False,
                              num_workers=args.num_worker,
                              collate_fn=train_collate_fn)

    val_set = ImageDataset(dst.query + dst.gallery, val_list)

    val_loader = DataLoader(val_set, 
                              batch_size=args.batch_size, 
                              sampler=RandomIdentitySampler(dst.train, args.batch_size, args.num_instance),
                              num_workers=args.num_worker)

    return train_loader, val_loader, class_num, cam_num, view_num