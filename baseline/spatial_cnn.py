import numpy as np
import pickle
from PIL import Image
import time
import tqdm
import shutil
from random import randint
import argparse

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from util import *
from network import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='PyTorch UCF101 spatial stream video level training')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', default=1e-3, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

def main():

    global args
    args = parser.parse_args()

    #Prepare DataLoader
    data_loader = Data_Loader(
                        BATCH_SIZE=arg.batch_size,
                        num_workers=4,
                        data_path=,
                        dic_path=, 
                        )
    
    train_loader = data_loader.train()
    test_loader = data_loader.test()

    spatial_cnn = Spatial_CNN(
                        nb_epochs=arg.epochs
                        lr=arg.lr
                        batch_size=arg.batch_size
                        resume=arg.resume
                        start_epoch=arg.start_epoch
                        evaluate=arg.evaluate
                        train_loader=train_loader
                        test_loader=test_loader
    )

    spatial_cnn.build_model_and_optimizer()
    spatial_cnn.run()





class Spatial_CNN():

    def __init__(self, nb_epochs, lr, batch_size, resume, start_epoch, evaluate, train_loader, test_loader):

        self.nb_epochs=nb_epochs
        self.lr=lr
        self.batch_size=batch_size
        self.resume=resume
        self.start_epoch=start_epoch
        self.evaluate=evaluate
        self.train_loader=train_loader
        self.test_loader=test_loader

    def build_model_and_optimizer(self):
        self.model = network.ResNet18(pretrained= true)
        #Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.LR, momentum=0.9, weight_decay=1e-6)
    
    def run(self):
        self.best_prec1=0
        cudnn.benchmark = True

        if self.resume:
            if os.path.isfile(args.resume):
                print("==> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("==> loaded checkpoint '{}' (epoch {})"
                  .format(self.resume, checkpoint['epoch']))
            else:
                print("==> no checkpoint found at '{}'".format(self.resume))
        if self.evaluate:
            prec1, val_loss = self.validate_1epoch()

        for self.epoch in range(self.start_epoch, self.nb_epochs):
            print('==> Epoch:[{0}/{1}][training stage]'.format(epoch, self.nb_epochs))
            self.train_1epoch()
            print('==> Epoch:[{0}/{1}][validation stage]'.format(epoch, self.nb_epochs))
            prec1, val_loss = self.validate_1epoch()

            is_best = top1 > self.best_prec1
            self.best_prec1 = max(prec1, self.best_prec1)
            save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_prec1,
                'optimizer' : self.optimizer.state_dict()
            },is_best)

    def train_1epoch(self):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        #switch to train mode
        self.model.train()    
        end = time.time()
        # mini-batch training
        progress = tqdm.tqdm(self.train_loader)
        for i, (data,label) in enumerate(progress):
    
            # measure data loading time
            data_time.update(time.time() - end)
            
            label = label.cuda(async=True)
            input_var = Variable(data).cuda()
            target_var = Variable(label).cuda()

            # compute output
            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
            losses.update(loss.data[0], data.size(0))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Data Time':[round(data_time.avg,3)],
                'Loss':[losses.avg],
                'Prec@1':[top1.avg],
                'Prec@5':[top5.avg]}

        record_info(info, 'record/training.csv')

    def validate_1epoch(self):

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()

        progress = tqdm.tqdm(self.test_loader)
        for i, (data,label) in enumerate(progress):
            
            label = label.cuda(async=True)
            data_var = Variable(data, volatile=True).cuda(async=True)
            label_var = Variable(label, volatile=True).cuda(async=True)

            # compute output
            output = self.model(data_var)
            loss = self.criterion(output, label_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
            losses.update(loss.data[0], data.size(0))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Data Time':[round(data_time.avg,3)],
                'Loss':[losses.avg],
                'Prec@1':[top1.avg],
                'Prec@5':[top5.avg]}

        record_info(info, 'record/testing.csv')

        return prec1, losses.avg


class Data_Loader():
    def __init__(self, BATCH_SIZE, num_workers, data_path, dic_path):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers = num_workers
        self.data_dir=data_dir

        #load data dictionary
        with open(dic_path+'/sub_jhmdb_train_video.pickle','rb') as f:
            dic_train=pickle.load(f)
        f.close()

        with open(dic_path+'/sub_jhmdb_test_video.pickle','rb') as f:
            dic_test=pickle.load(f)
        f.close()

        self.training_set = JHMDB_rgb_data(dic=dic_training, root_dir=self.data_path, transform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        self.validation_set = UCF101_rgb_data(dic=dic_testing, root_dir=self.data_path ,transform = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))

    def train(self):
        train_loader = DataLoader(
            dataset=self.training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers)
        return train_loader

    def validate(self):
        test_loader = DataLoader(
            dataset=self.testing_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)
        return test_loader