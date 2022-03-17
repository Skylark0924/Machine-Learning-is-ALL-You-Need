import os
import sys

sys.path.append("./")
import argparse
import time
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from datasets import dataset
from model import vggnet
from loss import CrossEntropyLoss
from flops import print_model_parm_flops
from transfroms import Padding, RandomCrop, RandomFlip, Cutout

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--log_file', type=str, default='log.txt')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
###dataset
parser.add_argument('--dataset', default='CIFAR10LT', type=str)
parser.add_argument('--net_cfg', default="A", type=str)
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
### random seed
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--calculate-flops', default=False, type=bool)

# You cannot change the following parameters in the first four tasks. 
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=2e-2, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')


def log(args, text):
    open("Logs/" + args.log_file, "a+").write(text + '\n')
    print(text)


def Train():
    ### SEED
    if args.seed is not None:
        SEED = args.seed
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)
        torch.backends.cudnn.deterministic = True

    ##### dataset
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    normalize = transforms.Normalize(mean, std)

    # TODO: task 3
    transform_train = transforms.Compose([
        Padding(padding=4),
        RandomCrop(size=32),
        RandomFlip(),
        transforms.ToTensor(),
        Cutout(1, 8),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    data = getattr(dataset, args.dataset)(batch_size=args.batch_size, transform_train=transform_train,
                                          transform_test=transform_test)

    #####
    model = vggnet(cfg=args.net_cfg, num_classes=args.num_classes)
    model = nn.DataParallel(model)
    cudnn.benchmark = True
    model = model.cuda()

    if args.calculate_flops:
        print_model_parm_flops(model)
        return

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                                nesterov=True)

    ##### lr multi step schedule
    def lr_schedule_multistep(epoch):
        if epoch < 5:
            factor = (epoch + 1) / 5.0
            lr = args.lr * (1 / 3.0 * (1 - factor) + factor)
        elif epoch < 80:
            lr = args.lr
        elif epoch < 90:
            lr = args.lr * 0.1
        else:
            lr = args.lr * 0.001

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    # finetune
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    #####Train
    end_epoch = args.epochs
    best_acc = 0.0

    print(model)

    cls_num_list = data.img_num_list

    for epoch in range(start_epoch, end_epoch):
        #####adjust learning rate every epoch begining
        lr = lr_schedule_multistep(epoch)
        #####
        model.train()
        total = 0.0
        correct = 0.0

        idx = epoch // 160
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], cls_num_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()

        loss_function = CrossEntropyLoss()
        # loss_function = nn.CrossEntropyLoss()

        for i, (inputs, target) in enumerate(data.train):
            input, target = inputs.cuda(), target.cuda()
            logit = model(input)
            loss = loss_function(logit, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predicted = logit.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        acc_n = logit.max(dim=1)[1].eq(target).sum().item()
        log(args, 'Train acc_n %.5f lr %.5f' % (acc_n / input.size(0), lr))

        model.eval()
        total = 0.0
        class_num = torch.zeros(args.num_classes).cuda()
        correct = torch.zeros(args.num_classes).cuda()
        for i, (inputs, target) in enumerate(data.test):
            input, target = inputs.cuda(), target.cuda()
            logit = model(input)

            _, predicted = logit.max(1)
            total += target.size(0)
            target_one_hot = F.one_hot(target, args.num_classes)
            predict_one_hot = F.one_hot(predicted, args.num_classes)
            class_num = class_num + target_one_hot.sum(dim=0).to(torch.float)
            correct = correct + (target_one_hot + predict_one_hot == 2).sum(dim=0).to(torch.float)

        acc = correct.sum() / total

        if best_acc < acc:
            best_acc = acc
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, "model_best.pth")

        log(args, "Test epoch=%d  acc=%.5f best_acc=%.5f" % (epoch, correct.sum() / total, best_acc))
        log(args, "Test " + str(correct / class_num))


if __name__ == '__main__':
    args = parser.parse_args()
    Train()
