import ast
import json
import os
import time
import random
import argparse
import numpy as np
from cv2 import transform
import glob
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.utils as utils
import torchvision.transforms as transforms
from model1 import AttnVGG_before
from model2 import AttnVGG_after
from utilities import *
from utilities import _worker_init_fn_

from nih_dataset import XRAY,PacemakerDataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


_PACEMAKER = True
_XRAY = not _PACEMAKER
_CIFAR = not _XRAY and not _PACEMAKER

log_path = {
    'CIFAR': 'CIFAR_logs',
    'XRAY': 'logs',
    'PACEMAKER': 'pacemaker_logs'
}

if _CIFAR:
    d_logs = log_path['CIFAR']
elif _XRAY:
    d_logs = log_path['XRAY']
elif _PACEMAKER:
    train_base_line = 'log_train_185_base_feature_16_drop015_stable0_72'
    d_logs = log_path['PACEMAKER']

for d in log_path.values():  # , d_train, d_test]:
    if not os.path.exists(d):
        os.mkdir(d)

def main():
    ## load data
    print('\nloading the dataset ...\n')
    if _CIFAR:
        parser = argparse.ArgumentParser(description="LearnToPayAttn-CIFAR100")
        parser.add_argument("--batch_size", type=int, default=8, help="batch size")
        parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
        parser.add_argument("--lr", type=float, default=0.1, help="initial learning rate")
        parser.add_argument("--outf", type=str, default="logs", help='path of log files')
        parser.add_argument("--attn_mode", type=str, default="after",
                            help='insert attention modules before OR after maxpooling layers')

        parser.add_argument("--normalize_attn", action='store_true',
                            help='if True, attention map is normalized by softmax; otherwise use sigmoid')
        parser.add_argument("--no_attention", action='store_true', help='turn down attention')
        parser.add_argument("--log_images", action='store_true', help='log images and (is available) attention maps')

        parser.add_argument("--pre-train", action='store_true', help='use pre train model from outf path')

        opt = parser.parse_args()
        # CIFAR-100: 500 training images and 100 testing images per class
        num_aug = 3
        im_size = 32
        transform_train = transforms.Compose([
            transforms.RandomCrop(im_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        trainset = torchvision.datasets.CIFAR100(root='CIFAR100_data', train=True, download=True,
                                                 transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=8,
                                                  worker_init_fn=_worker_init_fn_)
        testset = torchvision.datasets.CIFAR100(root='CIFAR100_data', train=False, download=True,
                                                transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=5)

        num_of_class = 100
        device_ids = [0, 1]
        criterion = nn.CrossEntropyLoss()
    elif _XRAY or _PACEMAKER:
        if _XRAY:
            parser = argparse.ArgumentParser(description="LearnToPayAttn-XRAY")
            opt = argparse.Namespace(attn_mode='before', batch_size=8, epochs=50, log_images=True, lr=1e-3,
                                     no_attention=False, base_feature_size=16, image_size=128,
                                     momentum=0.9, weight_decay=1e-3, dropout=0.15,
                                     normalize_attn=True, outf=d_logs, pre_train=True)
        else:
            pre_train = True
            if pre_train:
                chest_xray_pretrain_path = "/home/matan/pycharm_projects/xray/LearnToPayAttention/log_train_185_base_feature_16_drop015_stable0_72/net.pth"
                print("setting arguments for PACEMAKER, assuming previous train was done")
                parser = argparse.ArgumentParser(description="LearnToPayAttn-XRAY")
                opt = argparse.Namespace(attn_mode='before', batch_size=8, epochs=100, log_images=True, lr=1e-4,
                                         no_attention=False, base_feature_size=16, image_size=128,
                                         momentum=0.9, weight_decay=1e-3, dropout=0.15,
                                         normalize_attn=True, outf=d_logs, pre_train=True,
                                         chest_xray_pretrain_path=chest_xray_pretrain_path)

            else:
                chest_xray_pretrain_path = 'gazim'
                print("setting arguments for PACEMAKER, assuming previous train was done")
                parser = argparse.ArgumentParser(description="LearnToPayAttn-XRAY")
                opt = argparse.Namespace(attn_mode='before', batch_size=8, epochs=150, log_images=True, lr=1e-3,
                                         no_attention=False, base_feature_size=16, image_size=128,
                                         momentum=0.9, weight_decay=1e-3, dropout=0.15,
                                         normalize_attn=True, outf=d_logs, pre_train=True,
                                         chest_xray_pretrain_path=chest_xray_pretrain_path)
        num_aug = 1
        raw_size = 1024
        im_size = opt.image_size

        transform_train = transforms.Compose([
            transforms.Resize(im_size),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # 0.498, std: 0.185
            transforms.Normalize((0.5,), (0.185,))
        ])

        transform_test = transforms.Compose([
            transforms.Resize(im_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.185,))
        ])

        if _XRAY:
            xray = XRAY(transform_train, transform_test, force_pre_process=False)
            trainset = xray.train_set
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=8,
                                                      worker_init_fn=_worker_init_fn_)
            testset = xray.test_set
            testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=5)
            class_to_index = xray.class_to_index()
        else:
            trainset = PacemakerDataset(transform=transform_train, is_train=True)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=8,
                                                      worker_init_fn=_worker_init_fn_)

            testset = PacemakerDataset(transform=transform_test, is_train=False)
            testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=5)
            class_to_index = trainset.class_to_index()

        num_of_class = len(class_to_index.keys())
        device_ids = [0, 1]
        # criterion = nn.BCEWithLogitsLoss()
        criterion = nn.CrossEntropyLoss()

    else:
        raise Exception("how da hell did you get here ????")

    print(opt)
    print('done num_of_classes: %s , post crop size: %s' % (num_of_class, im_size))

    ## load network
    print('\nloading the network ...\n')
    # use attention module?
    if not opt.no_attention:
        print('\nturn on attention ...\n')
    else:
        print('\nturn off attention ...\n')
    # (linear attn) insert attention befroe or after maxpooling?
    # (grid attn only supports "before" mode)
    if opt.attn_mode == 'before':
        print('\npay attention before maxpooling layers...\n')
        net = AttnVGG_before(im_size=im_size, num_classes=num_of_class,
                             attention=not opt.no_attention, normalize_attn=opt.normalize_attn, init='xavierUniform',
                             _base_features=opt.base_feature_size, dropout=opt.dropout)
    elif opt.attn_mode == 'after':
        print('\npay attention after maxpooling layers...\n')
        net = AttnVGG_after(im_size=im_size, num_classes=num_of_class,
                            attention=not opt.no_attention, normalize_attn=opt.normalize_attn, init='xavierUniform')
    else:
        raise NotImplementedError("Invalid attention mode!")

    print('done')

    ## move to GPU
    print('\nmoving to GPU ...\n')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_pre_trained = None
    record = None

    if opt.pre_train:
        try:
            if os.path.exists(opt.chest_xray_pretrain_path):
                model_pre_trained = torch.load(opt.chest_xray_pretrain_path)
                record = None
        except AttributeError:
            record_path = os.path.join(opt.outf, 'record')
            if os.path.exists(record_path):
                with open(os.path.join(opt.outf, 'record'), 'r') as frecord:
                    # contents = record.read()
                    record = json.load(frecord)
                # record = ast.literal_eval(contents)

                if os.path.exists(record['model']):
                    model_pre_trained = torch.load(record['model'])

                print("found pre trained data: ", record)

    if model_pre_trained is not None:
        model = nn.DataParallel(net, device_ids=device_ids)
        model.load_state_dict(model_pre_trained)
        model = model.to(device)
    else:
        model = nn.DataParallel(net, device_ids=device_ids).to(device)

    criterion.to(device)
    print('done')

    if record is None:
        lr = opt.lr
        first_epoch = 0
        step = 0
    else:
        lr = record['lr']
        first_epoch = record['epoch']
        step = record['step']

    ### optimizer
    if _CIFAR:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        lr_lambda = lambda epoch: np.power(0.5, int(epoch / 25))
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif _XRAY or _PACEMAKER:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
        lr_lambda = lambda epoch: np.power(0.5, int(epoch / 25))
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # training
    start = time.time()
    print('\nstart training [%s]...\n' % start)

    running_avg_accuracy = 0
    writer = SummaryWriter(opt.outf)
    for epoch in range(first_epoch, first_epoch + opt.epochs):
        images_disp = []
        # adjust learning rate
        scheduler.step()
        writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        print("\nepoch %d learning rate %f\n" % (epoch, optimizer.param_groups[0]['lr']))
        # run for one epoch
        for aug in range(num_aug):
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # warm up
                model.train()
                model.zero_grad()
                optimizer.zero_grad()
                if (aug == 0) and (i == 0):  # archive images in order to save to logs
                    images_disp.append(inputs[0:36, :, :, :])
                # forward
                pred, __, __, __ = model(inputs)
                # backward
                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()
                # display results
                if i % 10 == 0:
                    model.eval()
                    pred, __, __, __ = model(inputs)
                    predict = torch.argmax(pred, 1)
                    # print("predict: ", predict.shape, "pred: ", pred.shape, "label: ", labels.shape, "input: ", inputs.shape)
                    total = labels.size(0)
                    correct = torch.eq(predict, labels).sum().double().item()
                    accuracy = correct / total
                    # print("accuracy:%s = correct:%s [pred:%s, predict:%s, labels:%s] / total:%s" % (accuracy, correct, pred, predict, labels, total))
                    running_avg_accuracy = 0.9 * running_avg_accuracy + 0.1 * accuracy
                    writer.add_scalar('train/loss', loss.item(), step)
                    writer.add_scalar('train/accuracy', accuracy, step)
                    writer.add_scalar('train/running_avg_accuracy', running_avg_accuracy, step)
                    print("[epoch %d][aug %d/%d][%d/%d] loss %.4f accuracy %.2f%% running avg accuracy %.2f%%"
                          % (epoch, aug, num_aug - 1, i, len(trainloader) - 1, loss.item(), (100 * accuracy),
                             (100 * running_avg_accuracy)))
                step += 1
        # the end of each epoch: test & log
        print('\none epoch done [took: %s], saving records ...\n' % (time.time() - start))
        state = os.path.join(opt.outf, 'net.pth')
        torch.save(model.state_dict(), state)
        with open(os.path.join(opt.outf, 'record'), 'w') as record:
            srecord = {"lr": optimizer.param_groups[0]['lr'], "epoch": epoch, "model": state, "step": step, 'global_arg': str(opt)}
            json.dump(srecord, record)

        if epoch == opt.epochs / 2:
            torch.save(model.state_dict(), os.path.join(opt.outf, 'net%d.pth' % epoch))
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            # log scalars
            images_disp.append(inputs[0:36, :, :, :])
            if not _PACEMAKER: # TODO not needed, remove if
                for i, data in enumerate(testloader, 0):
                    images_test, labels_test = data
                    images_test, labels_test = images_test.to(device), labels_test.to(device)

                    pred_test, __, __, __ = model(images_test)
                    predict = torch.argmax(pred_test, 1)
                    total += labels_test.size(0)
                    correct += torch.eq(predict, labels_test).sum().double().item()

                writer.add_scalar('test/accuracy', correct / total, epoch)
                print("\n[epoch %d] accuracy on test data: %.2f%%\n" % (epoch, 100 * correct / total))
            else:
                print("\n[epoch %d] log images for pacemaker" % epoch)
            # log images
            if opt.log_images:
                print('\nlog images ...\n')
                I_train = utils.make_grid(images_disp[0], nrow=6, normalize=True, scale_each=True)
                writer.add_image('train/image', I_train, epoch)
                #if epoch == 0:
                if epoch == first_epoch:
                    I_test = utils.make_grid(images_disp[1], nrow=6, normalize=True, scale_each=True)
                    writer.add_image('test/image', I_test, epoch)
            if opt.log_images and (not opt.no_attention):
                print('\nlog attention maps ...\n')
                # base factor
                if opt.attn_mode == 'before':
                    min_up_factor = 1
                else:
                    min_up_factor = 2
                # sigmoid or softmax
                if opt.normalize_attn:
                    vis_fun = visualize_attn_softmax
                else:
                    vis_fun = visualize_attn_sigmoid
                # training data
                __, c1, c2, c3 = model(images_disp[0])
                if c1 is not None:
                    attn1 = vis_fun(I_train, c1, up_factor=min_up_factor, nrow=6)
                    writer.add_image('train/attention_map_1', attn1, epoch)
                if c2 is not None:
                    attn2 = vis_fun(I_train, c2, up_factor=min_up_factor * 2, nrow=6)
                    writer.add_image('train/attention_map_2', attn2, epoch)
                if c3 is not None:
                    attn3 = vis_fun(I_train, c3, up_factor=min_up_factor * 4, nrow=6)
                    writer.add_image('train/attention_map_3', attn3, epoch)
                # test data
                __, c1, c2, c3 = model(images_disp[1])
                if c1 is not None:
                    attn1 = vis_fun(I_test, c1, up_factor=min_up_factor, nrow=6)
                    writer.add_image('test/attention_map_1', attn1, epoch)
                if c2 is not None:
                    attn2 = vis_fun(I_test, c2, up_factor=min_up_factor * 2, nrow=6)
                    writer.add_image('test/attention_map_2', attn2, epoch)
                if c3 is not None:
                    attn3 = vis_fun(I_test, c3, up_factor=min_up_factor * 4, nrow=6)
                    writer.add_image('test/attention_map_3', attn3, epoch)

        start = time.time()


if __name__ == "__main__":
    main()
