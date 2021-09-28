import torch
import os
from .triplet_loss import TripletLoss
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth

def make_loss(args, num_classes, use_cuda=True):
    if args.no_margin:
        triplet = TripletLoss()
        print("using soft triplet loss for training")

    else:
        triplet = TripletLoss(args.margin)
        print("using triplet loss with margin:{}".format(args.margin))

    if args.label_smooth== 'on':
        xent = CrossEntropyLabelSmooth(args, num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)
    else:
        xent = F.cross_entropy()
        print("label smooth off, numclasses:", num_classes)

    return triplet, xent


    # if args.label_smooth:
    #     # ID_LOSS = xent(score, target)