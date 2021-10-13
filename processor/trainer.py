import os
import time
from numpy import divide
import torch
import torch.nn as nn
from utils import R1_mAP_eval
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
             loss_fn,
             num_query
             ):

    log_period = args.log_period
    checkpoint_period = args.checkpoint_period
    eval_period = args.eval_period



    device = torch.device(args.cuda)
    epochs = args.epochs
    print("MGN.train")
    # print('start training')

    loss_meter = AverageMeter()
    # acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=True)


    if device:
        model.to(device)

    scaler = amp.GradScaler()

    print('-------------- Start Training ------------')
    # train
    for epoch in range(1, epochs+1):
        start_time = time.time()
        loss_meter.reset()
        # acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for ind, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            # loss_t = 0
            # loss_s = 0
            with amp.autocast(enabled=True):

                triplet_1, triplet_2, triplet_3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3 = model(img)

                # triplet loss

                loss_tri = [triplet_1, triplet_2, triplet_3]

                loss_t = [loss_fn[0](i, target)[0] for i in loss_tri]
                loss_t = sum(loss_t) / len(loss_t)
                # for i in loss_tri:
                #     # print(loss_fn[0])
                #     # print(loss_fn[0](i, target)[0])
                #     loss_t = loss_t + loss_fn[0](i, target)[0]
                
                # loss_t = loss_t / len(loss_tri)

                loss_soft = [l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3]

                loss_s = [loss_fn[1](j, target) for j in loss_soft]
                loss_s = sum(loss_s) / len(loss_s)
                # for j in loss_soft:
                #     loss_s = loss_s + loss_fn[1](j, target)

                # loss_s = loss_s / len(loss_soft)
                
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

        print("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(),
                os.path.join(args.output_dir, 'MGN' + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            model.eval()
            for iters, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)
                    feat = model(img)
                    evaluator.update((feat, pid, camid))
            
            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            print("Validation Results - Epoch: {}".format(epoch))
            print('mAP: {:.1%}'.format(mAP))
            for r in [1, 5, 10]:
                print("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            torch.cuda.empty_cache()            


def inference(args, model, val_loader, num_query, device):


    print("Test----")
    print('query num', num_query)
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=True)

    evaluator.reset()

    model.eval() #model 切换到concat特征
    

    for iter, (img, pid, camid, camids, target_view, imgpath)in enumerate(val_loader):
        # print(pid)
        with torch.no_grad():

            img = img.to(device)
            feat = model(img)
            evaluator.update((feat, pid, camid))
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    print("Validation Results")
    print('mAP: {:.1%}'.format(mAP))
    for r in [1, 5, 10]:
        print("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))