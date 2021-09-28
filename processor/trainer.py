import logging
import os
import time
import torch
import torch.nn as nn
# from utils.metrics import R1_mAP_eval
from torch.cuda import amp
from loss.make_loss import make_loss

class AverageMeter(object):
    """Computes and stores the average and current value"""

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

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def Trainer(args,
             model,
             train_loader,
             val_loader,
             optimizer,
             scheduler,
             loss_fn
             ):

    log_period = args.log_period
    checkpoint_period = args.checkpoint_period
    eval_period = args.eval_period



    device = torch.device(args.cuda)
    epochs = args.epochs
    logger = logging.getLogger("MGN.train")
    logger.info('start training')

    loss_meter = AverageMeter()
    # acc_meter = AverageMeter()

    if device:
        model.to(device)

    scaler = amp.GradScaler()

    print('-------------- Start Training ------------')
    # train
    for epoch in range(1, epochs+1):
        start_time = time.time()
        loss_meter.reset()
        # acc_meter.reset()
        scheduler.step(epoch)
        model.train()
        for ind, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            loss_t = 0
            loss_s = 0
            with amp.autocast(enabled=True):

                triplet_1, triplet_2, triplet_3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3 = model(img)

                # triplet loss

                loss_tri = [triplet_1, triplet_2, triplet_3]

                for i in loss_tri:
                    # print(loss_fn[0])
                    # print(loss_fn[0](i, target)[0])
                    loss_t = loss_t + loss_fn[0](i, target)[0]
                
                loss_t = loss_t / len(loss_tri)

                loss_soft = [l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3]

                for j in loss_soft:
                    loss_s = loss_s + loss_fn[1](j, target)

                loss_s = loss_s / len(loss_soft)
                
                loss = loss_t + loss_s
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()



            loss_meter.update(loss.item(), img.shape[0])
            if (ind + 1) % log_period == 0:
                print("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (ind + 1), len(train_loader),
                                    loss_meter.avg, scheduler.get_last_lr()[0]))
    
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (ind + 1)

        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(),
                os.path.join(args.output_dir, 'MGN' + '_{}.pth'.format(epoch)))
