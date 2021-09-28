# import torch
# import torch.nn as nn


# class BasicBlock(nn.Module):
#     """Basic Block for resnet 18 and resnet 34
#     """

#     #BasicBlock and BottleNeck block
#     #have different output size
#     #we use class attribute expansion
#     #to distinct
#     expansion = 1

#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()

#         #residual function
#         self.residual_function = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels * BasicBlock.expansion)
#         )

#         #shortcut
#         self.shortcut = nn.Sequential()

#         #the shortcut output dimension is not the same with residual function
#         #use 1*1 convolution to match the dimension
#         if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels * BasicBlock.expansion)
#             )

#     def forward(self, x):
#         return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

# class BottleNeck(nn.Module):
#     """Residual block for resnet over 50 layers
#     """
#     expansion = 4
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#         self.residual_function = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels * BottleNeck.expansion),
#         )

#         self.shortcut = nn.Sequential()

#         if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(out_channels * BottleNeck.expansion)
#             )

#     def forward(self, x):
#         return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

# class ResNet(nn.Module):

#     def __init__(self, block, num_block, num_classes=100):
#         super().__init__()

#         self.in_channels = 64

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True))
#         #we use a different inputsize than the original paper
#         #so conv2_x's stride is 1
#         self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
#         self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
#         self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
#         self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         # self.fc = nn.Linear(512 * block.expansion, num_classes)

#     def _make_layer(self, block, out_channels, num_blocks, stride):
#         """make resnet layers(by layer i didnt mean this 'layer' was the
#         same as a neuron netowork layer, ex. conv layer), one layer may
#         contain more than one residual block
#         Args:
#             block: block type, basic block or bottle neck block
#             out_channels: output depth channel number of this layer
#             num_blocks: how many blocks per layer
#             stride: the stride of the first block of this layer
#         Return:
#             return a resnet layer
#         """

#         # we have num_block blocks per layer, the first block
#         # could be 1 or 2, other blocks would always be 1
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_channels, out_channels, stride))
#             self.in_channels = out_channels * block.expansion

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         output = self.conv1(x)
#         output = self.conv2_x(output)
#         output = self.conv3_x(output)
#         output = self.conv4_x(output)


#         output = self.conv5_x(output)


#         output = self.avg_pool(output)
#         output = output.view(output.size(0), -1)
#         # output = self.fc(output)

#         return output

# def resnet18():
#     """ return a ResNet 18 object
#     """
#     return ResNet(BasicBlock, [2, 2, 2, 2])

# def resnet34():
#     """ return a ResNet 34 object
#     """
#     return ResNet(BasicBlock, [3, 4, 6, 3])

# def resnet50():
#     """ return a ResNet 50 object
#     """
#     return ResNet(BottleNeck, [3, 4, 6, 3])

# def resnet101():
#     """ return a ResNet 101 object
#     """
#     return ResNet(BottleNeck, [3, 4, 23, 3])

# def resnet152():
#     """ return a ResNet 152 object
#     """
#     return ResNet(BottleNeck, [3, 8, 36, 3])



# num_layer = {
#     18:[2, 2, 2, 2],
#     34:[3, 4, 6, 3],
#     50:[3, 4, 6, 3],
#     101:[3, 4, 23, 3],
#     152:[3, 8, 36, 3]
# }


import copy

import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet50, Bottleneck

def make_model(args):
    return MGN(args)

class bulid_MGN_resnet(nn.Module):
    def __init__(self, args):
        super(bulid_MGN_resnet, self).__init__()
        num_classes = args.num_classes

        resnet = resnet50(pretrained=True)

        self.backone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )

        res_conv4 = nn.Sequential(*resnet.layer3[1:])

        res_g_conv5 = resnet.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        
        pool2d = nn.MaxPool2d

        self.maxpool_zg_p1 = pool2d(kernel_size=(12, 4))
        self.maxpool_zg_p2 = pool2d(kernel_size=(24, 8))
        self.maxpool_zg_p3 = pool2d(kernel_size=(24, 8))
        self.maxpool_zp2 = pool2d(kernel_size=(12, 8))
        self.maxpool_zp3 = pool2d(kernel_size=(8, 8))

        reduction = nn.Sequential(nn.Conv2d(2048, args.feats, 1, bias=False), nn.BatchNorm2d(args.feats), nn.ReLU())

        self._init_reduction(reduction)
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)
        self.reduction_6 = copy.deepcopy(reduction)
        self.reduction_7 = copy.deepcopy(reduction)

        #self.fc_id_2048_0 = nn.Linear(2048, num_classes)
        self.fc_id_2048_0 = nn.Linear(args.feats, num_classes)
        self.fc_id_2048_1 = nn.Linear(args.feats, num_classes)
        self.fc_id_2048_2 = nn.Linear(args.feats, num_classes)

        self.fc_id_256_1_0 = nn.Linear(args.feats, num_classes)
        self.fc_id_256_1_1 = nn.Linear(args.feats, num_classes)
        self.fc_id_256_2_0 = nn.Linear(args.feats, num_classes)
        self.fc_id_256_2_1 = nn.Linear(args.feats, num_classes)
        self.fc_id_256_2_2 = nn.Linear(args.feats, num_classes)

        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)

        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        #nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        #nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):

        x = self.backone(x)

        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        zg_p1 = self.maxpool_zg_p1(p1)
        zg_p2 = self.maxpool_zg_p2(p2)
        zg_p3 = self.maxpool_zg_p3(p3)

        zp2 = self.maxpool_zp2(p2)
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]

        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]
        
        fg_p1 = self.reduction_0(zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction_1(zg_p2).squeeze(dim=3).squeeze(dim=2)
        fg_p3 = self.reduction_2(zg_p3).squeeze(dim=3).squeeze(dim=2)
        f0_p2 = self.reduction_3(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f1_p2 = self.reduction_4(z1_p2).squeeze(dim=3).squeeze(dim=2)
        f0_p3 = self.reduction_5(z0_p3).squeeze(dim=3).squeeze(dim=2)
        f1_p3 = self.reduction_6(z1_p3).squeeze(dim=3).squeeze(dim=2)
        f2_p3 = self.reduction_7(z2_p3).squeeze(dim=3).squeeze(dim=2)

        '''
        l_p1 = self.fc_id_2048_0(zg_p1.squeeze(dim=3).squeeze(dim=2))
        l_p2 = self.fc_id_2048_1(zg_p2.squeeze(dim=3).squeeze(dim=2))
        l_p3 = self.fc_id_2048_2(zg_p3.squeeze(dim=3).squeeze(dim=2))
        '''
        l_p1 = self.fc_id_2048_0(fg_p1)
        l_p2 = self.fc_id_2048_1(fg_p2)
        l_p3 = self.fc_id_2048_2(fg_p3)
        
        l0_p2 = self.fc_id_256_1_0(f0_p2)
        l1_p2 = self.fc_id_256_1_1(f1_p2)
        l0_p3 = self.fc_id_256_2_0(f0_p3)
        l1_p3 = self.fc_id_256_2_1(f1_p3)
        l2_p3 = self.fc_id_256_2_2(f2_p3)

        predict = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)

        return fg_p1, fg_p2, fg_p3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3

# class bulid_MGN_resnet(nn.Module):

#     def __init__(self, args, layer_nums=50):
#         super(bulid_MGN_resnet, self).__init__()

#         num_classes = args.num_classes
#         self.layer_nums = layer_nums

#         self.model = ResNet(BottleNeck, num_layer[layer_nums]) #default 50

#         # print(self.model)
#         self.layer1 = self.model.conv1
#         self.layer2 = self.model.conv2_x
#         self.layer3 = self.model.conv3_x
#         self.layer4 = self.model.conv4_x
#         self.layer5 = self.model.conv5_x
#         # print(self.layer5[0])

#         # self.layer1 = self.model.conv1
#         self.brach_1_max_pooling = nn.MaxPool2d(kernel_size=(12,4))
#         self.brach_2_max_pooling = nn.MaxPool2d(kernel_size=(24,8))
#         self.brach_3_max_pooling = nn.MaxPool2d(kernel_size=(24,8))

#         self.brach_2_split1_max_pooling = nn.MaxPool2d(kernel_size=(12,8))
#         # self.brach_2_split2_max_pooling = nn.AdaptiveMaxPool2d((12,8))

#         self.brach_3_split1_max_pooling = nn.MaxPool2d(kernel_size=(8,8))
#         # self.brach_3_split2_max_pooling = nn.AdaptiveMaxPool2d((8,8))
#         # self.brach_3_split3_max_pooling = nn.AdaptiveMaxPool2d((8,8))

#         # self.branch_2_layer_5 = self.model._make_layer(BottleNeck, 512, num_layer[50][3], 1)
#         # BottleNeck
#         self.branch_2_layer_5 = BottleNeck(1024, 512, stride=1)
#         self.branch_3_layer_5 = BottleNeck(1024, 512, stride=1)

#         # print(self.branch_2_layer_5)


#         self.conv1_1 = nn.Conv2d(2048, 256, 1, stride=1,bias=False)
#         self.bn1 = nn.BatchNorm2d(256)
#         self.relu1 = nn.ReLU(inplace=True)
#         nn.init.kaiming_normal_(self.conv1_1.weight, mode="fan_in")
#         nn.init.normal_(self.bn1.weight, mean=1., std=0.02)
#         nn.init.constant_(self.bn1.bias, 0.)

#         self.conv2_1 = nn.Conv2d(2048, 256, 1, stride=1,bias=False)
#         self.bn2 = nn.BatchNorm2d(256)
#         self.relu2 = nn.ReLU(inplace=True)
#         nn.init.kaiming_normal_(self.conv2_1.weight, mode="fan_in")
#         nn.init.normal_(self.bn2.weight, mean=1., std=0.02)
#         nn.init.constant_(self.bn2.bias, 0.)

#         self.conv2_1_1 = nn.Conv2d(2048, 256, 1, stride=1,bias=False)
#         self.bn2_1 = nn.BatchNorm2d(256)
#         self.relu2_1 = nn.ReLU(inplace=True)
#         nn.init.kaiming_normal_(self.conv2_1_1.weight, mode="fan_in")
#         nn.init.normal_(self.bn2_1.weight, mean=1., std=0.02)
#         nn.init.constant_(self.bn2_1.bias, 0.)

#         self.conv2_1_2 = nn.Conv2d(2048, 256, 1, stride=1,bias=False)
#         self.bn2_2 = nn.BatchNorm2d(256)
#         self.relu2_2 = nn.ReLU(inplace=True)
#         nn.init.kaiming_normal_(self.conv2_1_2.weight, mode="fan_in")
#         nn.init.normal_(self.bn2_2.weight, mean=1., std=0.02)
#         nn.init.constant_(self.bn2_2.bias, 0.)

#         self.conv3_1 = nn.Conv2d(2048, 256, 1, stride=1,bias=False)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.relu3 = nn.ReLU(inplace=True)
#         nn.init.kaiming_normal_(self.conv3_1.weight, mode="fan_in")
#         nn.init.normal_(self.bn3.weight, mean=1., std=0.02)
#         nn.init.constant_(self.bn3.bias, 0.)

#         self.conv3_1_1 = nn.Conv2d(2048, 256, 1, stride=1,bias=False)
#         self.bn3_1 = nn.BatchNorm2d(256)
#         self.relu3_1 = nn.ReLU(inplace=True)
#         nn.init.kaiming_normal_(self.conv3_1_1.weight, mode="fan_in")
#         nn.init.normal_(self.bn3_1.weight, mean=1., std=0.02)
#         nn.init.constant_(self.bn3_1.bias, 0.)

#         self.conv3_1_2 = nn.Conv2d(2048, 256, 1, stride=1,bias=False)
#         self.bn3_2 = nn.BatchNorm2d(256)
#         self.relu3_2 = nn.ReLU(inplace=True)
#         nn.init.kaiming_normal_(self.conv3_1_2.weight, mode="fan_in")
#         nn.init.normal_(self.bn3_2.weight, mean=1., std=0.02)
#         nn.init.constant_(self.bn3_2.bias, 0.)

#         self.conv3_1_3 = nn.Conv2d(2048, 256, 1, stride=1,bias=False)
#         self.bn3_3 = nn.BatchNorm2d(256)
#         self.relu3_3 = nn.ReLU(inplace=True)
#         nn.init.kaiming_normal_(self.conv3_1_3.weight, mode="fan_in")
#         nn.init.normal_(self.bn3_3.weight, mean=1., std=0.02)
#         nn.init.constant_(self.bn3_3.bias, 0.)

#         self.fc_id_2048_0 = nn.Linear(args.feats, num_classes)
#         self.fc_id_2048_1 = nn.Linear(args.feats, num_classes)
#         self.fc_id_2048_2 = nn.Linear(args.feats, num_classes)

#         self.fc_id_256_1_0 = nn.Linear(args.feats, num_classes)
#         self.fc_id_256_1_1 = nn.Linear(args.feats, num_classes)
#         self.fc_id_256_2_0 = nn.Linear(args.feats, num_classes)
#         self.fc_id_256_2_1 = nn.Linear(args.feats, num_classes)
#         self.fc_id_256_2_2 = nn.Linear(args.feats, num_classes)

#         self._init_fc(self.fc_id_2048_0)
#         self._init_fc(self.fc_id_2048_1)
#         self._init_fc(self.fc_id_2048_2)

#         self._init_fc(self.fc_id_256_1_0)
#         self._init_fc(self.fc_id_256_1_1)
#         self._init_fc(self.fc_id_256_2_0)
#         self._init_fc(self.fc_id_256_2_1)
#         self._init_fc(self.fc_id_256_2_2)

#     @staticmethod
#     def _init_fc(fc):
#         nn.init.kaiming_normal_(fc.weight, mode='fan_out')
#         #nn.init.normal_(fc.weight, std=0.001)
#         nn.init.constant_(fc.bias, 0.)

#     def forward(self, x):
        
#         bs = x.shape[0]
#         # shared backbone
#         output = self.layer1(x)
#         output = self.layer2(output)
#         output = self.layer3(output)

#         # MGN 3 branches

#         # 1-global branch
#         branch_1 = self.layer4(output)
#         # print(branch_1.shape)
#         branch_1 = self.layer5(branch_1)

#         branch_1 = self.brach_1_max_pooling(branch_1)
#         triplet_1 = branch_1
#         branch_1 = self.conv1_1(branch_1)
#         branch_1 = self.bn1(branch_1)
#         sft_branch_1 = self.relu1(branch_1)
 

#         # 2-local branch
#         branch_2 = self.layer4(output)
#         # branch_2 = self.branch_2_layer_5(branch_2)
#         branch_2 = self.layer5(branch_2)


#         gobal_max_2 = self.brach_2_max_pooling(branch_2)
#         triplet_2 = gobal_max_2
#         gobal_max_2 = self.conv2_1(gobal_max_2)
#         gobal_max_2 = self.bn2(gobal_max_2)
#         sft_branch_2 = self.relu2(gobal_max_2)

#         branch2_split_pooling1 = self.brach_2_split1_max_pooling(branch_2)
#         branch2_split1 = branch2_split_pooling1[:,:, 0:1, :]
#         branch2_split2 = branch2_split_pooling1[:,:, 1:2, :]

#         # split1
#         branch2_split1 = self.conv2_1_1(branch2_split1)
#         branch2_split1 = self.bn2_1(branch2_split1)
#         sft_branch_2_split1 = self.relu2_1(branch2_split1)
#         # split2 
#         branch2_split2 = self.conv2_1_2(branch2_split2)
#         branch2_split2 = self.bn2_2(branch2_split2)
#         sft_branch_2_split2 = self.relu2_2(branch2_split2)



#         # 3-local branch
#         branch_3 = self.layer4(output)
#         branch_3 = self.layer5(branch_3)

#         # branch_3 = self.branch_3_layer_5(branch_3)

#         gobal_max_3 = self.brach_3_max_pooling(branch_3)
#         triplet_3 = gobal_max_3
#         gobal_max_3 = self.conv3_1(gobal_max_3)
#         gobal_max_3 = self.bn3(gobal_max_3)
#         sft_branch_3 = self.relu3(gobal_max_3)


#         branch3_split_pooling1 = self.brach_3_split1_max_pooling(branch_3)

#         branch3_split1 = branch3_split_pooling1[:,:, 0:1, :]
#         branch3_split2 = branch3_split_pooling1[:,:, 1:2, :]
#         branch3_split3 = branch3_split_pooling1[:,:, 2:3, :]


#         branch3_split1 = self.conv3_1_1(branch3_split1)
#         branch3_split1 = self.bn3_1(branch3_split1)
#         sft_branch_3_split1 = self.relu3_1(branch3_split1)

#         branch3_split2 = self.conv3_1_2(branch3_split2)
#         branch3_split2 = self.bn3_2(branch3_split2)
#         sft_branch_3_split2 = self.relu3_2(branch3_split2)

#         branch3_split3 = self.conv3_1_3(branch3_split3)
#         branch3_split3 = self.bn3_3(branch3_split3)
#         sft_branch_3_split3 = self.relu3_3(branch3_split3)
        
  

#         print(sft_branch_1.shape)
#         l_p1 = self.fc_id_2048_0(sft_branch_1.squeeze(dim=3).squeeze(dim=2))
#         l_p2 = self.fc_id_2048_1(sft_branch_2.squeeze(dim=3).squeeze(dim=2))
#         l_p3 = self.fc_id_2048_2(sft_branch_3.squeeze(dim=3).squeeze(dim=2))
        
#         l0_p2 = self.fc_id_256_1_0(sft_branch_2_split1.squeeze(dim=3).squeeze(dim=2))
#         l1_p2 = self.fc_id_256_1_1(sft_branch_2_split2.squeeze(dim=3).squeeze(dim=2))
#         l0_p3 = self.fc_id_256_2_0(sft_branch_3_split1.squeeze(dim=3).squeeze(dim=2))
#         l1_p3 = self.fc_id_256_2_1(sft_branch_3_split2.squeeze(dim=3).squeeze(dim=2))
#         l2_p3 = self.fc_id_256_2_2(sft_branch_3_split3.squeeze(dim=3).squeeze(dim=2))        

#         return triplet_1, triplet_2, triplet_3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3

    
if __name__ == '__main__':

    model = bulid_MGN_resnet()
    x = torch.randn(1, 3, 384, 128)
    model(x)